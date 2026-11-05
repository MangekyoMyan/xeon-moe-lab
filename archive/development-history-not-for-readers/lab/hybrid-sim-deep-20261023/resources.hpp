#pragma once

// Deterministic fluid-resource event engine. Bytes are decimal bytes, rates GB/s,
// and time milliseconds. This is a resource model, not a hardware emulator.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sim2 {

struct Config {
    double host_total_GBs = 48;
    double cpu_read_GBs = 45;
    double gpu_total_GBs = 346;
    double h2d_GBs = 12;
    double d2h_GBs = 12;
    double weight_chunk_bytes = 1024 * 1024;
    // Charged at every background-weight chunk. The explicit latency arguments
    // of add_d2h/add_h2d_urgent already contain their entire setup cost.
    double copy_setup_us = 0;
};

struct Work {
    double bytes = 0;
    double compute_ms = 0;
    double delay_ms = 0;
    // Zero selects cpu_read_GBs for CPU, gpu_total_GBs for GPU.
    double read_GBs = 0;
};

struct Stats {
    double cpu_busy_ms = 0, gpu_busy_ms = 0;
    double h2d_busy_ms = 0, d2h_busy_ms = 0;
    double cpu_bytes = 0, gpu_bytes = 0;
    double h2d_bytes = 0, d2h_bytes = 0, weight_bytes = 0;
    double host_bytes = 0, vram_bytes = 0;
    double peak_host_GBs = 0, peak_gpu_GBs = 0;
    std::uint64_t weight_chunks = 0, weights_completed = 0;
};

class Engine {
    static constexpr double TEPS = 1e-10;
    static constexpr double BEPS = 1e-5;
    struct Job { Work work; std::function<void()> done; };
    struct Transfer { std::uint64_t id; double bytes, delay_ms; };
    struct Weight { int key; double bytes; };
    enum class HKind { none, urgent, weight };

    std::optional<Job> cpu_, gpu_;
    std::deque<Job> gpu_queue_;
    std::deque<Transfer> urgent_queue_, d2h_queue_;
    std::optional<Transfer> urgent_, d2h_;
    std::deque<Weight> weights_;
    std::optional<Weight> weight_;
    HKind hkind_ = HKind::none;
    double chunk_bytes_ = 0, chunk_delay_ms_ = 0;
    std::uint64_t next_id_ = 1;
    std::unordered_set<std::uint64_t> finished_;
    bool settling_ = false;

    static bool positive_finite(double x) { return std::isfinite(x) && x > 0; }
    static void validate_work(const Work &w) {
        if (!std::isfinite(w.bytes) || !std::isfinite(w.compute_ms) ||
            !std::isfinite(w.delay_ms) || !std::isfinite(w.read_GBs) ||
            w.bytes < 0 || w.compute_ms < 0 || w.delay_ms < 0 || w.read_GBs < 0)
            throw std::invalid_argument("invalid Work");
    }
    static bool done(const Work &w) {
        return w.delay_ms <= TEPS && w.compute_ms <= TEPS && w.bytes <= BEPS;
    }
    static bool transfer_done_now(const Transfer &t) {
        return t.delay_ms <= TEPS && t.bytes <= BEPS;
    }

    void settle() {
        if (settling_) return;
        settling_ = true;
        struct Reset { bool &flag; ~Reset() { flag = false; } } reset{settling_};
        for (int guard = 0; ; ++guard) {
            if (guard > 100000) throw std::runtime_error("zero-time callback loop");
            bool changed = false;
            if (cpu_ && done(cpu_->work)) {
                auto cb = std::move(cpu_->done); cpu_.reset(); changed = true;
                if (cb) cb();
            }
            if (gpu_ && done(gpu_->work)) {
                auto cb = std::move(gpu_->done); gpu_.reset(); changed = true;
                if (cb) cb();
            }
            if (d2h_ && transfer_done_now(*d2h_)) {
                finished_.insert(d2h_->id); d2h_.reset(); changed = true;
            }
            if (hkind_ == HKind::urgent && transfer_done_now(*urgent_)) {
                finished_.insert(urgent_->id); urgent_.reset();
                hkind_ = HKind::none; changed = true;
            }
            if (hkind_ == HKind::weight && chunk_delay_ms_ <= TEPS && chunk_bytes_ <= BEPS) {
                hkind_ = HKind::none; changed = true;
                if (weight_->bytes <= BEPS) {
                    const int key = weight_->key; weight_.reset();
                    ++stats.weights_completed;
                    if (on_complete) on_complete(key);
                }
            }
            if (!gpu_ && !gpu_queue_.empty()) {
                gpu_ = std::move(gpu_queue_.front()); gpu_queue_.pop_front(); changed = true;
            }
            if (!d2h_ && !d2h_queue_.empty()) {
                d2h_ = d2h_queue_.front(); d2h_queue_.pop_front(); changed = true;
            }
            if (hkind_ == HKind::none) {
                if (!urgent_queue_.empty()) {
                    urgent_ = urgent_queue_.front(); urgent_queue_.pop_front();
                    hkind_ = HKind::urgent; changed = true;
                } else {
                    if (!weight_) {
                        auto it = std::find_if(weights_.begin(), weights_.end(),
                            [&](const Weight &w) { return !can_start || can_start(w.key); });
                        if (it != weights_.end()) {
                            weight_ = *it; weights_.erase(it); changed = true;
                            // The owner marks its victim unreadable here, not at enqueue.
                            if (on_start) on_start(weight_->key);
                        }
                    }
                    if (weight_) {
                        chunk_bytes_ = cfg.weight_chunk_bytes > 0
                            ? std::min(cfg.weight_chunk_bytes, weight_->bytes) : weight_->bytes;
                        chunk_delay_ms_ = cfg.copy_setup_us * .001;
                        hkind_ = HKind::weight; ++stats.weight_chunks; changed = true;
                    }
                }
            }
            if (!changed) break;
        }
    }

    // Weighted max-min allocation. Each flow's weight equals its unconstrained
    // demand. CPU uses host DRAM, GPU uses VRAM, both DMA directions use both.
    // Progressive filling redistributes spare capacity when another resource
    // freezes a flow, rather than applying two independent scaling factors.
    std::array<double,4> rates() const {
        std::array<double,4> demand{};
        if (cpu_ && cpu_->work.delay_ms <= TEPS && cpu_->work.bytes > BEPS)
            demand[0] = cpu_->work.read_GBs > 0 ? cpu_->work.read_GBs : cfg.cpu_read_GBs;
        if (gpu_ && gpu_->work.delay_ms <= TEPS && gpu_->work.bytes > BEPS)
            demand[1] = gpu_->work.read_GBs > 0 ? gpu_->work.read_GBs : cfg.gpu_total_GBs;
        if ((hkind_ == HKind::urgent && urgent_->delay_ms <= TEPS && urgent_->bytes > BEPS) ||
            (hkind_ == HKind::weight && chunk_delay_ms_ <= TEPS && chunk_bytes_ > BEPS))
            demand[2] = cfg.h2d_GBs;
        if (d2h_ && d2h_->delay_ms <= TEPS && d2h_->bytes > BEPS) demand[3] = cfg.d2h_GBs;
        const std::array<std::array<bool,4>,2> uses{{{{true,false,true,true}},{{false,true,true,true}}}};
        const std::array<double,2> capacity{cfg.host_total_GBs,cfg.gpu_total_GBs};
        std::array<double,4> rate{};
        std::array<bool,4> active{};
        for (int i=0;i<4;++i) active[i] = demand[i] > 0;
        for (int pass=0;pass<8;++pass) {
            double increment = std::numeric_limits<double>::infinity();
            bool any = false;
            for (int i=0;i<4;++i) if (active[i]) {
                any = true; increment = std::min(increment,(demand[i]-rate[i])/demand[i]);
            }
            if (!any) break;
            for (int resource=0;resource<2;++resource) {
                double used=0, slope=0;
                for (int i=0;i<4;++i) if (uses[resource][i]) {
                    used += rate[i]; if (active[i]) slope += demand[i];
                }
                if (slope > 0) increment = std::min(increment,std::max(0.0,capacity[resource]-used)/slope);
            }
            for (int i=0;i<4;++i) if (active[i]) rate[i] += increment*demand[i];
            for (int i=0;i<4;++i) if (active[i] && rate[i] >= demand[i]*(1-1e-12)) active[i]=false;
            for (int resource=0;resource<2;++resource) {
                double used=0; for (int i=0;i<4;++i) if (uses[resource][i]) used += rate[i];
                if (used >= capacity[resource]*(1-1e-12))
                    for (int i=0;i<4;++i) if (uses[resource][i]) active[i]=false;
            }
        }
        return rate;
    }

    double next_dt(const std::array<double,4> &rate) const {
        double dt = std::numeric_limits<double>::infinity();
        auto job = [&](const std::optional<Job> &j, double r) {
            if (!j) return;
            const Work &w=j->work;
            if (w.delay_ms > TEPS) dt=std::min(dt,w.delay_ms);
            else {
                if (w.compute_ms > TEPS) dt=std::min(dt,w.compute_ms);
                if (w.bytes > BEPS && r > 0) dt=std::min(dt,w.bytes/(r*1e6));
            }
        };
        auto transfer = [&](double bytes, double delay, double r) {
            if (delay > TEPS) dt=std::min(dt,delay);
            else if (bytes > BEPS && r > 0) dt=std::min(dt,bytes/(r*1e6));
        };
        job(cpu_,rate[0]); job(gpu_,rate[1]);
        if (hkind_ == HKind::urgent) transfer(urgent_->bytes,urgent_->delay_ms,rate[2]);
        if (hkind_ == HKind::weight) transfer(chunk_bytes_,chunk_delay_ms_,rate[2]);
        if (d2h_) transfer(d2h_->bytes,d2h_->delay_ms,rate[3]);
        return dt;
    }

    void advance(double dt, const std::array<double,4> &rate) {
        if (!std::isfinite(dt) || dt < 0) throw std::runtime_error("invalid event interval");
        const double host=rate[0]+rate[2]+rate[3], vram=rate[1]+rate[2]+rate[3];
        if (host > cfg.host_total_GBs*(1+1e-10) || vram > cfg.gpu_total_GBs*(1+1e-10))
            throw std::runtime_error("resource bandwidth exceeded");
        stats.peak_host_GBs=std::max(stats.peak_host_GBs,host);
        stats.peak_gpu_GBs=std::max(stats.peak_gpu_GBs,vram);
        auto job = [&](std::optional<Job> &j,double r,double &busy,double &bytes) {
            if (!j) return;
            busy += dt;
            Work &w=j->work;
            if (w.delay_ms > TEPS) w.delay_ms=std::max(0.0,w.delay_ms-dt);
            else {
                w.compute_ms=std::max(0.0,w.compute_ms-dt);
                const double moved=std::min(w.bytes,r*1e6*dt); w.bytes-=moved; bytes+=moved;
            }
        };
        const double before_cpu=stats.cpu_bytes, before_gpu=stats.gpu_bytes;
        const double before_h2d=stats.h2d_bytes, before_d2h=stats.d2h_bytes;
        job(cpu_,rate[0],stats.cpu_busy_ms,stats.cpu_bytes);
        job(gpu_,rate[1],stats.gpu_busy_ms,stats.gpu_bytes);
        if (hkind_ != HKind::none) {
            stats.h2d_busy_ms += dt;
            double &delay=hkind_==HKind::urgent ? urgent_->delay_ms : chunk_delay_ms_;
            double &bytes=hkind_==HKind::urgent ? urgent_->bytes : chunk_bytes_;
            if (delay > TEPS) delay=std::max(0.0,delay-dt);
            else {
                const double moved=std::min(bytes,rate[2]*1e6*dt); bytes-=moved; stats.h2d_bytes+=moved;
                if (hkind_==HKind::weight) { weight_->bytes-=moved; stats.weight_bytes+=moved; }
            }
        }
        if (d2h_) {
            stats.d2h_busy_ms += dt;
            if (d2h_->delay_ms > TEPS) d2h_->delay_ms=std::max(0.0,d2h_->delay_ms-dt);
            else { const double moved=std::min(d2h_->bytes,rate[3]*1e6*dt); d2h_->bytes-=moved; stats.d2h_bytes+=moved; }
        }
        const double dma=(stats.h2d_bytes-before_h2d)+(stats.d2h_bytes-before_d2h);
        stats.host_bytes += stats.cpu_bytes-before_cpu+dma;
        stats.vram_bytes += stats.gpu_bytes-before_gpu+dma;
        now_ms += dt;
    }

    std::uint64_t add_transfer(double bytes,double latency_us,bool h2d) {
        if (!std::isfinite(bytes) || !std::isfinite(latency_us) || bytes < 0 || latency_us < 0)
            throw std::invalid_argument("invalid transfer");
        const std::uint64_t id=next_id_++;
        (h2d ? urgent_queue_ : d2h_queue_).push_back({id,bytes,latency_us*.001});
        settle(); return id;
    }

public:
    Config cfg;
    double now_ms = 0;
    Stats stats;
    // Callbacks may enqueue work, but must not recursively advance time.
    std::function<bool(int)> can_start;
    std::function<void(int)> on_start, on_complete;

    explicit Engine(Config config = {}):cfg(config) {
        if (!positive_finite(cfg.host_total_GBs) || !positive_finite(cfg.cpu_read_GBs) ||
            !positive_finite(cfg.gpu_total_GBs) || !positive_finite(cfg.h2d_GBs) ||
            !positive_finite(cfg.d2h_GBs) || !std::isfinite(cfg.weight_chunk_bytes) ||
            !std::isfinite(cfg.copy_setup_us) || cfg.weight_chunk_bytes < 0 || cfg.copy_setup_us < 0)
            throw std::invalid_argument("invalid resource Config");
    }
    bool cpu_idle() const { return !cpu_; }
    bool gpu_idle() const { return !gpu_ && gpu_queue_.empty(); }
    // Snapshot of the active job only. Its byte/compute/delay fields are actual
    // remaining work after earlier contention, not initial work minus elapsed
    // wall time. Queued GPU jobs are intentionally excluded. Returning a copy
    // prevents a scheduling heuristic from modifying the event engine.
    std::optional<Work> gpu_remaining_work() const {
        return gpu_ ? std::optional<Work>(gpu_->work) : std::nullopt;
    }
    std::size_t pending_weights() const { return weights_.size()+(weight_ ? 1 : 0); }
    bool transfer_done(std::uint64_t id) const { return finished_.count(id) != 0; }
    bool idle() const {
        return cpu_idle() && gpu_idle() && !d2h_ && d2h_queue_.empty() &&
            hkind_==HKind::none && urgent_queue_.empty() && pending_weights()==0;
    }
    void start_cpu(Work work,std::function<void()> done_callback = {}) {
        validate_work(work); if (cpu_) throw std::logic_error("CPU job already active");
        cpu_=Job{work,std::move(done_callback)}; settle();
    }
    void enqueue_gpu(Work work,std::function<void()> done_callback = {}) {
        validate_work(work); gpu_queue_.push_back({work,std::move(done_callback)}); settle();
    }
    std::uint64_t add_d2h(double bytes,double latency_us = 0) { return add_transfer(bytes,latency_us,false); }
    std::uint64_t add_h2d_urgent(double bytes,double latency_us = 0) { return add_transfer(bytes,latency_us,true); }
    void enqueue_weight(int key,double bytes) {
        if (!std::isfinite(bytes) || bytes < 0) throw std::invalid_argument("invalid weight bytes");
        weights_.push_back({key,bytes}); settle();
    }
    void wait_until(const std::function<bool()> &condition) {
        if (settling_) throw std::logic_error("cannot advance time from a callback");
        for (std::uint64_t guard=0; ; ++guard) {
            if (guard > 100000000) throw std::runtime_error("event iteration limit");
            settle(); if (condition()) return;
            const auto r=rates(); const double dt=next_dt(r);
            if (!std::isfinite(dt)) throw std::runtime_error("no event can satisfy wait condition (blocked resource/cache?)");
            advance(dt,r);
        }
    }
    // Useful for deterministic arrivals and explicit idle gaps. Background work
    // progresses even when the caller has no foreground task to await.
    void run_for(double duration_ms) {
        if (settling_ || !std::isfinite(duration_ms) || duration_ms < 0)
            throw std::invalid_argument("invalid run_for");
        const double end=now_ms+duration_ms;
        while (now_ms < end-TEPS) {
            settle(); const auto r=rates(); advance(std::min(end-now_ms,next_dt(r)),r);
        }
        settle();
    }
    void drain() { wait_until([&] { return idle(); }); }
};

} // namespace sim2
