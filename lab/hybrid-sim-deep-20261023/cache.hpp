#pragma once

// Expert residency with physical-slot ownership. A reservation never destroys
// its victim: only the copy engine's on_start transition does so. The caller
// must snapshot and lock every resident expert of a routed layer BEFORE asking
// for any LRU admissions or starting work that may progress the copy engine.
#include "resources.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace cache2 {

constexpr int key_count = 40 * 384;
constexpr double expert_bytes = 18800640.0;

struct Config {
    int slots = 342, policy = 3, budget = 8, queue = 32;
    double decay = .97, margin = 1.15, min_heat = 1.5;
    double expert_bytes = cache2::expert_bytes;
};

struct Stats {
    std::uint64_t submitted = 0, rejected = 0, completed = 0;
    std::uint64_t overwrites = 0, heat_normalizations = 0;
    int peak_pending = 0;
};

class Cache {
    struct Slot {
        int ready_key = -1;
        int pending_key = -1;
        bool writing = false;
    };
    struct Entry { double score; std::uint64_t stamp; int key; };
    struct MinEntry {
        bool operator()(const Entry &a,const Entry &b) const {
            if (a.score != b.score) return a.score > b.score;
            if (a.stamp != b.stamp) return a.stamp > b.stamp;
            return a.key > b.key;
        }
    };
    struct MaxEntry {
        bool operator()(const Entry &a,const Entry &b) const {
            if (a.score != b.score) return a.score < b.score;
            return a.key > b.key;
        }
    };

    sim2::Engine &engine_;
    std::vector<Slot> slots_;
    std::vector<int> free_, ready_slot_, pending_slot_;
    std::vector<std::uint32_t> readers_;
    std::vector<double> heat_;
    std::vector<std::uint64_t> last_;
    std::priority_queue<Entry,std::vector<Entry>,MinEntry> victims_;
    std::priority_queue<Entry,std::vector<Entry>,MaxEntry> candidates_;
    double scale_ = 1;
    std::uint64_t tick_ = 0;
    int pending_ = 0;

    static void validate_key(int key) {
        if (key < 0 || key >= key_count) throw std::out_of_range("expert key outside 40x384");
    }
    static void unique_keys(const std::vector<int> &keys) {
        for (std::size_t i=0;i<keys.size();++i) {
            validate_key(keys[i]);
            if (std::find(keys.begin(),keys.begin()+i,keys[i]) != keys.begin()+i)
                throw std::invalid_argument("duplicate reader lock key");
        }
    }
    bool frequency() const { return cfg.policy == 1 || cfg.policy == 3; }
    double victim_value(int key) const { return frequency() ? heat_[key] : double(last_[key]); }
    bool available_candidate(int key) const {
        return ready_slot_[key] < 0 && pending_slot_[key] < 0;
    }
    bool available_victim(int key) const {
        const int index=ready_slot_[key];
        return index >= 0 && slots_[index].pending_key < 0 && !slots_[index].writing;
    }
    void push_victim(int key) {
        if (available_victim(key)) victims_.push({victim_value(key),last_[key],key});
    }
    void push_candidate(int key) {
        if (available_candidate(key) && heat_[key] > 0)
            candidates_.push({heat_[key],last_[key],key});
    }
    void rebuild() {
        victims_ = {}; candidates_ = {};
        for (int key=0;key<key_count;++key) { push_victim(key); push_candidate(key); }
    }
    void compact() {
        if (victims_.size() > std::size_t(std::max(128,8*cfg.slots)) ||
            candidates_.size() > 8*key_count) rebuild();
    }
    int take_victim() {
        while (!victims_.empty()) {
            const Entry e=victims_.top(); victims_.pop();
            if (available_victim(e.key) && e.score == victim_value(e.key) && e.stamp == last_[e.key])
                return e.key;
        }
        return -1;
    }
    int best_candidate() {
        while (!candidates_.empty()) {
            const Entry &e=candidates_.top();
            if (available_candidate(e.key) && e.score == heat_[e.key] && e.stamp == last_[e.key])
                return e.key;
            candidates_.pop();
        }
        return -1;
    }

    bool can_start(int key) const {
        validate_key(key);
        const int index=pending_slot_[key];
        if (index < 0) throw std::logic_error("weight copy has no cache reservation");
        const Slot &s=slots_[index];
        if (s.pending_key != key || s.writing) throw std::logic_error("invalid copy-start state");
        return s.ready_key < 0 || readers_[s.ready_key] == 0;
    }
    void on_start(int key) {
        if (!can_start(key)) throw std::logic_error("copy started over a live GPU reader");
        Slot &s=slots_[pending_slot_[key]];
        if (s.ready_key >= 0) {
            const int old=s.ready_key;
            if (ready_slot_[old] != pending_slot_[key]) throw std::logic_error("victim mapping mismatch");
            ready_slot_[old]=-1; s.ready_key=-1; ++stats.overwrites;
            push_candidate(old);
        }
        s.writing=true;
    }
    void on_complete(int key) {
        validate_key(key);
        const int index=pending_slot_[key];
        if (index < 0) throw std::logic_error("completion without cache reservation");
        Slot &s=slots_[index];
        if (s.pending_key != key || !s.writing || s.ready_key >= 0 || ready_slot_[key] >= 0)
            throw std::logic_error("invalid cache completion state");
        s.pending_key=-1; s.writing=false; s.ready_key=key;
        pending_slot_[key]=-1; ready_slot_[key]=index;
        --pending_; ++stats.completed; push_victim(key);
    }
    bool admit(int key,bool check_heat) {
        validate_key(key);
        if (!available_candidate(key)) return false;
        if (!cfg.slots || pending_ >= cfg.queue) { ++stats.rejected; return false; }
        int old=-1,index=-1;
        if (!free_.empty()) index=free_.back();
        else {
            old=take_victim();
            if (old < 0) { ++stats.rejected; return false; }
            if (check_heat && heat_[key] <= heat_[old]*cfg.margin) {
                push_victim(old); return false;
            }
            index=ready_slot_[old];
        }
        Slot &s=slots_[index];
        if (s.pending_key >= 0 || s.writing) throw std::logic_error("double cache-slot reservation");
        if (old < 0) free_.pop_back();
        s.pending_key=key; pending_slot_[key]=index; ++pending_;
        ++stats.submitted; stats.peak_pending=std::max(stats.peak_pending,pending_);
        // enqueue_weight may synchronously call on_start, so the complete
        // reservation and all counters must exist before handing over ownership.
        engine_.enqueue_weight(key,cfg.expert_bytes);
        return true;
    }
    void promote(int budget) {
        for (int i=0;i<budget;++i) {
            const int key=best_candidate();
            if (key < 0 || heat_[key]*scale_ < cfg.min_heat) break;
            // Discard this heap record before enqueueing: on_start can add the
            // victim as a candidate and change the heap top synchronously.
            const Entry saved=candidates_.top(); candidates_.pop();
            if (!admit(key,true)) {
                if (available_candidate(key)) candidates_.push(saved);
                break;
            }
        }
    }

public:
    const Config cfg;
    Stats stats;

    explicit Cache(Config config,sim2::Engine &engine):engine_(engine),
        slots_(config.slots >= 0 && config.slots <= key_count ? config.slots : 0),
        ready_slot_(key_count,-1),pending_slot_(key_count,-1),readers_(key_count,0),
        heat_(key_count,0),last_(key_count,0),cfg(config) {
        if (cfg.slots < 0 || cfg.slots > key_count || cfg.policy < 0 || cfg.policy > 4 ||
            cfg.budget < 0 || cfg.queue < 1 || !std::isfinite(cfg.decay) || cfg.decay <= 0 || cfg.decay > 1 ||
            !std::isfinite(cfg.margin) || cfg.margin < 1 || !std::isfinite(cfg.min_heat) || cfg.min_heat < 0 ||
            !std::isfinite(cfg.expert_bytes) || cfg.expert_bytes <= 0)
            throw std::invalid_argument("invalid cache Config");
        if (engine_.can_start || engine_.on_start || engine_.on_complete || engine_.pending_weights())
            throw std::invalid_argument("cache requires exclusive weight-copy callbacks");
        for (int i=cfg.slots-1;i>=0;--i) free_.push_back(i);
        engine_.can_start=[this](int key){return can_start(key);};
        engine_.on_start=[this](int key){on_start(key);};
        engine_.on_complete=[this](int key){on_complete(key);};
    }
    Cache(const Cache &)=delete;
    Cache &operator=(const Cache &)=delete;
    Cache(Cache &&)=delete;
    Cache &operator=(Cache &&)=delete;
    ~Cache() {
        engine_.can_start={}; engine_.on_start={}; engine_.on_complete={};
    }

    void begin_token() {
        scale_*=cfg.decay;
        // A single extreme decay may underflow, which correctly discards past
        // heat. Normalizing here bounds the reciprocal used by future observes.
        if (scale_ < 1e-60) {
            for (double &h:heat_) h*=scale_;
            scale_=1; ++stats.heat_normalizations; rebuild();
        }
    }
    bool ready(int key) const { validate_key(key); return ready_slot_[key] >= 0; }
    bool pending(int key) const { validate_key(key); return pending_slot_[key] >= 0; }
    int pending_count() const { return pending_; }
    double effective_heat(int key) const { validate_key(key); return heat_[key]*scale_; }

    void observe_layer(const std::vector<int> &keys) {
        for (int key:keys) validate_key(key);
        for (int key:keys) {
            if (tick_ == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("cache observation counter exhausted");
            heat_[key]+=1/scale_; last_[key]=++tick_;
            push_candidate(key); push_victim(key);
        }
        compact();
    }
    void lock(const std::vector<int> &keys) {
        unique_keys(keys);
        for (int key:keys) if (!ready(key) || readers_[key] == std::numeric_limits<std::uint32_t>::max())
            throw std::logic_error("reader lock requires a ready expert");
        for (int key:keys) ++readers_[key];
    }
    void unlock(const std::vector<int> &keys) {
        unique_keys(keys);
        for (int key:keys) if (!readers_[key] || !ready(key))
            throw std::logic_error("unbalanced cache reader unlock");
        for (int key:keys) --readers_[key];
        // Do not advance the Engine here: this method is also valid inside a
        // GPU-completion callback. The enclosing/next settle starts freed slots.
    }
    void admit_lru(const std::vector<int> &keys) {
        if (cfg.policy != 2) return;
        for (int key:keys) validate_key(key);
        for (int key:keys) if (available_candidate(key)) admit(key,false);
        compact();
    }
    void finish_token(bool decode) {
        if (cfg.policy == 3 || (cfg.policy == 1 && !decode)) promote(cfg.budget);
        compact();
    }
    void check() const {
        int occupied=0,ready_count=0,pending_count=0;
        std::vector<bool> free_seen(slots_.size(),false);
        for (int index:free_) {
            if (index < 0 || index >= int(slots_.size()) || free_seen[index])
                throw std::logic_error("invalid free-slot list");
            free_seen[index]=true;
        }
        for (int i=0;i<int(slots_.size());++i) {
            const Slot &s=slots_[i];
            if (s.ready_key < 0 && s.pending_key < 0) {
                if (!free_seen[i] || s.writing) throw std::logic_error("unlisted free slot");
                continue;
            }
            ++occupied;
            if (free_seen[i]) throw std::logic_error("occupied free slot");
            if (s.ready_key >= 0) {
                ++ready_count;
                if (s.writing || ready_slot_[s.ready_key] != i)
                    throw std::logic_error("ready-slot mapping mismatch");
            }
            if (s.pending_key >= 0) {
                ++pending_count;
                if (pending_slot_[s.pending_key] != i || ready_slot_[s.pending_key] >= 0)
                    throw std::logic_error("pending-slot mapping mismatch");
            } else if (s.writing) throw std::logic_error("writing without pending expert");
        }
        int mapped_ready=0,mapped_pending=0;
        for (int k=0;k<key_count;++k) {
            mapped_ready+=ready_slot_[k]>=0; mapped_pending+=pending_slot_[k]>=0;
            if (ready_slot_[k]>=0 && slots_[ready_slot_[k]].ready_key!=k)
                throw std::logic_error("reverse ready mapping mismatch");
            if (pending_slot_[k]>=0 && slots_[pending_slot_[k]].pending_key!=k)
                throw std::logic_error("reverse pending mapping mismatch");
            if (readers_[k] && ready_slot_[k]<0) throw std::logic_error("reader owns overwritten bytes");
            if (!std::isfinite(heat_[k]) || heat_[k]<0) throw std::logic_error("invalid heat");
        }
        if (occupied+int(free_.size())!=cfg.slots || ready_count!=mapped_ready ||
            pending_count!=mapped_pending || pending_count!=pending_ || pending_>cfg.queue ||
            std::size_t(pending_)!=engine_.pending_weights() || stats.completed+pending_!=stats.submitted)
            throw std::logic_error("cache occupancy/copy accounting mismatch");
    }
};

} // namespace cache2
