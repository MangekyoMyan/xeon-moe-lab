#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

constexpr int NL = 40, NE = 384, NK = NL*NE;
constexpr double EXPERT_BYTES = 18800640.0;
constexpr double EPS = 1e-9;
struct Record { uint32_t step, pos; std::array<uint16_t,240> ids; std::array<uint8_t,40> hits; int decode=0, turn=0; };
struct Config {
    std::string id, card;
    int policy=3, slots=342, budget=4, queue=32;
    double decay=.97, margin=1.15, min_heat=1.5;
    double cpu_scale=1, host_GBs=45, gpu_GBs=400, common_ms=15, pcie_GBs=12;
    double contention=.5, gpu_contention=1, sync_us=40, launch_us=10, parallel_common=.2;
    int instant=0, detail=0;
};
// 0: no cache; 1: prefill-only admission; 2: async LRU; 3: limited decayed frequency; 4: synchronous LRU.
struct Entry { double score; uint64_t stamp; int key; };
struct MinEntry { bool operator()(const Entry &a,const Entry &b) const {
    if(a.score!=b.score)return a.score>b.score;
    if(a.stamp!=b.stamp)return a.stamp>b.stamp;
    return a.key>b.key;
}};
struct MaxEntry { bool operator()(const Entry &a,const Entry &b) const {
    if(a.score!=b.score)return a.score<b.score;
    return a.key>b.key;
}};
struct Copy { int key; double remaining; };
struct Metrics {
    double now=0, moved=0, cpu_busy=0, gpu_busy=0, copy_wait=0;
    uint64_t submitted=0, completed=0, rejected=0;
    int peak_queue=0;
};

class Simulator {
public:
    Config c;
    Metrics m;
    std::array<double,7> cpu;
    std::array<double,7> cpu_floor;
    std::vector<uint8_t> state; // absent, loading, ready
    std::vector<double> heat;
    std::vector<uint64_t> last;
    std::vector<int> locked;
    std::priority_queue<Entry,std::vector<Entry>,MinEntry> victims;
    std::priority_queue<Entry,std::vector<Entry>,MaxEntry> candidates;
    std::deque<Copy> copies;
    int occupancy=0, epoch=0;
    uint64_t tick=0;
    double scale=1;
    Simulator(Config cfg,std::array<double,7> times,std::array<double,7> floors={}):c(cfg),cpu(times),cpu_floor(floors),state(NK),heat(NK),last(NK),locked(NK,-1) {
        assert(c.slots>=0&&c.slots<=NK&&c.queue>=1&&c.cpu_scale>0&&c.pcie_GBs>0);
        assert(c.decay>0&&c.decay<=1&&c.contention>=0&&c.contention<=1);
        assert(c.parallel_common>=0&&c.parallel_common<=1&&c.host_GBs>0&&c.gpu_GBs>0);
    }
    bool frequency() const { return c.policy==1||c.policy==3; }
    double value(int key) const { return frequency()?heat[key]:double(last[key]); }
    void push_victim(int key) { if(state[key]==2)victims.push({value(key),last[key],key}); }
    void observe(int key) {
        heat[key]+=1/scale; last[key]=++tick;
        if(state[key]==0)candidates.push({heat[key],last[key],key});
        push_victim(key);
    }
    int victim() {
        std::vector<Entry> held;
        int result=-1;
        while(!victims.empty()) {
            auto e=victims.top();victims.pop();
            if(state[e.key]!=2||e.score!=value(e.key)||e.stamp!=last[e.key])continue;
            if(locked[e.key]==epoch){held.push_back(e);continue;}
            result=e.key;break;
        }
        for(auto &e:held)victims.push(e);
        return result;
    }
    bool admit(int key,bool check_heat) {
        if(!c.slots||state[key]!=0)return false;
        if(int(copies.size())>=c.queue){++m.rejected;return false;}
        int old=-1;
        if(occupancy==c.slots) {
            old=victim();
            if(old<0)return false;
            if(check_heat&&heat[key]<=heat[old]*c.margin) { push_victim(old);return false; }
        }
        if(old>=0) {
            state[old]=0;candidates.push({heat[old],last[old],old});
        }else ++occupancy;
        state[key]=1;
        ++m.submitted;
        if(c.instant) {
            m.moved+=EXPERT_BYTES;++m.completed;state[key]=2;push_victim(key);
        }else {
            copies.push_back({key,EXPERT_BYTES});
            m.peak_queue=std::max(m.peak_queue,int(copies.size()));
        }
        assert(occupancy<=c.slots);
        return true;
    }
    void advance(double cpu_work,double gpu_work,bool drain=false,double cpu_compute=0) {
        assert(cpu_work>=0&&gpu_work>=0&&cpu_compute>=0);
        int guard=0;
        while(cpu_work>EPS||cpu_compute>EPS||gpu_work>EPS||(drain&&!copies.empty())) {
            assert(++guard<1000000);
            const bool active=!copies.empty();
            const double tx=active?std::min(c.pcie_GBs,c.host_GBs*.8):0;
            const double cr=1-c.contention*tx/c.host_GBs;
            const double gr=1-c.gpu_contention*std::min(tx/c.gpu_GBs,.8);
            double dt=std::numeric_limits<double>::infinity();
            if(cpu_work>EPS)dt=std::min(dt,cpu_work/cr);
            if(cpu_compute>EPS)dt=std::min(dt,cpu_compute);
            if(gpu_work>EPS)dt=std::min(dt,gpu_work/gr);
            if(active)dt=std::min(dt,copies.front().remaining/(tx*1e6));
            assert(std::isfinite(dt)&&dt>=0);
            if(cpu_work>EPS||cpu_compute>EPS)m.cpu_busy+=dt;
            if(gpu_work>EPS)m.gpu_busy+=dt;
            cpu_work=std::max(0.0,cpu_work-dt*cr);
            cpu_compute=std::max(0.0,cpu_compute-dt);
            gpu_work=std::max(0.0,gpu_work-dt*gr);
            if(active) {
                const double moved=std::min(copies.front().remaining,dt*tx*1e6);
                copies.front().remaining-=moved;m.moved+=moved;
                if(copies.front().remaining<1e-4) {
                    const int key=copies.front().key;copies.pop_front();
                    assert(state[key]==1);state[key]=2;++m.completed;push_victim(key);
                }
            }
            m.now+=dt;
        }
    }
    void compact() {
        if(victims.size()>size_t(std::max(128,8*c.slots))) {
            victims={};for(int k=0;k<NK;++k)push_victim(k);
        }
        if(candidates.size()>8*NK) {
            candidates={};for(int k=0;k<NK;++k)if(state[k]==0&&heat[k]>0)candidates.push({heat[k],last[k],k});
        }
    }
    void promote(int budget) {
        ++epoch; // no GPU expert kernel is live at a token boundary
        for(int b=0;b<budget;++b) {
            while(!candidates.empty()) {
                const auto &e=candidates.top();
                if(state[e.key]!=0||heat[e.key]!=e.score||last[e.key]!=e.stamp)candidates.pop();else break;
            }
            if(candidates.empty())break;
            const auto e=candidates.top();
            if(e.score*scale<c.min_heat)break;
            if(!admit(e.key,true))break;
            candidates.pop();
        }
    }
    std::array<int,40> step(const Record &r) {
        std::array<int,40> hits{};
        scale*=c.decay;
        assert(scale>1e-150);
        for(int l=0;l<NL;++l) {
            advance(0,c.common_ms*(1-c.parallel_common)/NL);
            ++epoch;
            for(int j=0;j<6;++j) {
                const int key=l*NE+r.ids[6*l+j];
                assert(key>=0&&key<NK);
                const bool hit=state[key]==2;
                if(hit){++hits[l];locked[key]=epoch;}
                observe(key);
                if(!hit&&(c.policy==2||c.policy==4))admit(key,false);
            }
            const int miss=6-hits[l];
            const double cpu_ms=miss?cpu[miss]/c.cpu_scale+c.sync_us*.001:0;
            const double compute_ms=miss?cpu_floor[miss]+c.sync_us*.001:0;
            const double gpu_ms=c.common_ms*c.parallel_common/NL+
                (hits[l]?hits[l]*EXPERT_BYTES/(c.gpu_GBs*1e6)+c.launch_us*.001:0);
            advance(cpu_ms,gpu_ms,false,compute_ms);
            if(c.policy==4) {
                const double start=m.now;advance(0,0,true);m.copy_wait+=m.now-start;
            }
        }
        if(c.policy==3)promote(c.budget);
        if(c.policy==1&&!r.decode)promote(c.budget);
        compact();
        return hits;
    }
    void check() const {
        int count=0,loading=0;
        for(auto v:state){count+=v!=0;loading+=v==1;}
        assert(count==occupancy&&loading==int(copies.size())&&count<=c.slots);
    }
};

struct Summary {
    int tokens=0; uint64_t hits=0,submitted=0;
    double elapsed=0,moved=0,cpu_busy=0,gpu_busy=0,copy_wait=0;
    std::vector<double> latencies;
    void add(int hit,const Metrics &before,const Metrics &after) {
        ++tokens;hits+=hit;submitted+=after.submitted-before.submitted;
        elapsed+=after.now-before.now;moved+=after.moved-before.moved;
        cpu_busy+=after.cpu_busy-before.cpu_busy;gpu_busy+=after.gpu_busy-before.gpu_busy;
        copy_wait+=after.copy_wait-before.copy_wait;latencies.push_back(after.now-before.now);
    }
};

static std::vector<Record> read_trace(const char *path,const char *phases) {
    std::ifstream f(path,std::ios::binary),p(phases);assert(f&&p);
    std::vector<Record> rows;
    while(true) {
        Record r{};
        f.read(reinterpret_cast<char*>(&r.step),4);
        if(f.eof()&&f.gcount()==0)break;
        assert(f.gcount()==4);
        f.read(reinterpret_cast<char*>(&r.pos),4);
        f.read(reinterpret_cast<char*>(r.ids.data()),480);
        f.read(reinterpret_cast<char*>(r.hits.data()),40);assert(f);
        assert(p>>r.decode>>r.turn);
        assert((r.decode==0||r.decode==1)&&r.turn>=0&&r.turn<100);
        for(auto id:r.ids)assert(id<NE);
        rows.push_back(r);
    }
    assert(!rows.empty());return rows;
}

static void selftest() {
    const std::array<double,7> cpu{0,1,2,3,4,5,6};
    Config c;c.policy=2;c.slots=2;c.pcie_GBs=1;c.host_GBs=100;c.contention=0;c.gpu_contention=0;
    Simulator a(c,cpu);
    assert(a.admit(1,false)&&a.state[1]==1);
    a.advance(0,1);assert(a.state[1]==1&&std::abs(a.m.moved-1e6)<1);
    a.advance(0,0,true);assert(a.state[1]==2&&std::abs(a.m.now-EXPERT_BYTES/1e6)<1e-8);a.check();
    assert(a.admit(2,false));a.advance(0,0,true);a.check();
    a.observe(1);assert(a.admit(3,false));assert(a.state[2]==0);a.check();
    a.advance(0,0,true);a.check();
    Config d=c;d.contention=1;d.host_GBs=2;Simulator b(d,cpu);b.admit(0,false);
    b.advance(2,0);assert(std::abs(b.m.now-4)<1e-8);
    Simulator serial(c,cpu);serial.advance(2,3);assert(std::abs(serial.m.now-3)<1e-8);
    Simulator floor(c,cpu);floor.advance(.5,0,false,2);assert(std::abs(floor.m.now-2)<1e-8);
    Config q=c;q.queue=1;Simulator queue(q,cpu);assert(queue.admit(0,false));assert(!queue.admit(1,false));queue.check();
    Config z=c;z.slots=0;Simulator zero(z,cpu);assert(!zero.admit(0,false));zero.check();
    Config instant=c;instant.instant=1;Simulator now(instant,cpu);now.admit(0,false);assert(now.state[0]==2&&now.m.now==0);now.check();
    Config full=c;full.slots=NK;full.instant=1;full.common_ms=10;full.parallel_common=.2;
    full.sync_us=0;full.launch_us=0;full.gpu_GBs=200;
    Simulator full_cache(full,cpu);
    Record repeated{};for(int l=0;l<40;++l)for(int j=0;j<6;++j)repeated.ids[l*6+j]=j;
    full_cache.step(repeated);const double start=full_cache.m.now;
    const auto hit=full_cache.step(repeated);for(int n:hit)assert(n==6);
    const double expected=10+240*EXPERT_BYTES/(200*1e6);
    assert(std::abs(full_cache.m.now-start-expected)<1e-7);full_cache.check();
    std::cout<<"selftests passed\n";
}

int main(int argc,char **argv) {
    if(argc==2&&std::string(argv[1])=="--selftest"){selftest();return 0;}
    if(argc!=7)return 2;
    const auto records=read_trace(argv[1],argv[2]);
    std::array<double,7> cpu{};std::ifstream cpu_file(argv[3]);assert(cpu_file);
    for(auto &x:cpu)assert(cpu_file>>x);
    std::array<double,7> floors{};for(auto &x:floors)assert(cpu_file>>x);
    std::ifstream config_file(argv[4]);std::ofstream output(argv[5]);assert(config_file&&output);
    output<<"id\tcard\tturn\ttokens\tms_per_token\ttok_s\thit_fraction\tcopies_per_token\tDMA_MB_per_token\tcpu_busy_ms\tgpu_busy_ms\tcopy_wait_ms\tp50_ms\tp95_ms\tpeak_queue\tpending_at_end\n";
    int scenario=0;
    while(true) {
        Config c;
        if(!(config_file>>c.id))break;
        assert(config_file>>c.card>>c.policy>>c.slots>>c.budget>>c.queue>>c.decay>>c.margin>>c.min_heat>>
            c.cpu_scale>>c.host_GBs>>c.gpu_GBs>>c.common_ms>>c.pcie_GBs>>c.contention>>c.gpu_contention>>
            c.sync_us>>c.launch_us>>c.parallel_common>>c.instant>>c.detail);
        Simulator sim(c,cpu,floors);
        Summary all;std::array<Summary,100> per_turn;
        std::ofstream detail;
        if(c.detail){detail.open(std::string(argv[6])+"/"+c.id+".csv");assert(detail);detail<<"step,turn,decode,ms,hits,copies,DMA_MB,queue\n";}
        for(const auto &r:records) {
            const auto before=sim.m;
            auto h=sim.step(r);int hit=0;for(int n:h)hit+=n;
            if(c.instant&&c.slots==4500&&c.policy==2)for(int l=0;l<NL;++l)assert(h[l]==r.hits[l]);
            if(r.decode){all.add(hit,before,sim.m);per_turn[r.turn].add(hit,before,sim.m);}
            if(c.detail)detail<<r.step<<','<<r.turn<<','<<r.decode<<','<<sim.m.now-before.now<<','<<hit<<','<<sim.m.submitted-before.submitted<<','<<(sim.m.moved-before.moved)/1e6<<','<<sim.copies.size()<<'\n';
        }
        sim.check();
        auto write=[&](Summary s,int turn) {
            if(!s.tokens)return;
            std::sort(s.latencies.begin(),s.latencies.end());
            auto quantile=[&](double f){return s.latencies[std::min(s.latencies.size()-1,size_t(f*(s.latencies.size()-1)))];};
            output<<std::setprecision(12)<<c.id<<'\t'<<c.card<<'\t'<<turn<<'\t'<<s.tokens<<'\t'<<s.elapsed/s.tokens<<'\t'<<1000*s.tokens/s.elapsed<<'\t'<<double(s.hits)/(240*s.tokens)<<'\t'<<double(s.submitted)/s.tokens<<'\t'<<s.moved/1e6/s.tokens<<'\t'<<s.cpu_busy/s.tokens<<'\t'<<s.gpu_busy/s.tokens<<'\t'<<s.copy_wait/s.tokens<<'\t'<<quantile(.5)<<'\t'<<quantile(.95)<<'\t'<<sim.m.peak_queue<<'\t'<<sim.copies.size()<<'\n';
        };
        write(all,-1);for(int t=0;t<100;++t)write(per_turn[t],t);
        output.flush();
        std::cout<<"done "<<++scenario<<' '<<c.id<<' '<<all.elapsed/all.tokens<<" ms/token\n"<<std::flush;
    }
}
