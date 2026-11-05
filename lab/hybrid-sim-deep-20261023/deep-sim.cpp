#include "resources.hpp"
#include "cache.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

constexpr double EB=18800640;
constexpr double EF=70778880;
struct Record {uint32_t step,pos;std::array<uint16_t,240> ids;std::array<uint8_t,40> hits;int decode,turn;};
struct Layer {double pre_b,pre_f,post_b,post_f,shared_b,shared_f;};
struct Cfg {
    std::string id,card,trace,phases,shapes,cpu_file;
    int slots,policy,budget,queue,balance,context_base,candidate_first,detail;
    double decay,margin,min_heat,usable_gib,workspace_gib;
    double mem_scale,cpu_fixed_scale,cpu_compute_scale;
    double common_bw,expert_bw,gpu_peak_bw,gpu_GF,attention_GF;
    double host_bw,h2d_bw,d2h_bw,chunk,copy_setup_us,transfer_us,common_us,expert_us,merge_us,occupancy;
};
struct Totals {
    int n=0;double time=0,ready=0,gpu_used=0,cpu_used=0,copies=0,weightbytes=0,returnwait=0,inputwait=0;
    double cpu_busy=0,gpu_busy=0,h2d_busy=0,d2h_busy=0;
    double common_path=0,attention_path=0,moe_input=0,parallel=0,cpu_tail=0,gpu_tail=0,merge_path=0,other_path=0;
    std::vector<double> latency;
};

static std::vector<Record> load_trace(const std::string &path,const std::string &phase){
    std::ifstream f(path,std::ios::binary),p(phase);assert(f&&p);std::vector<Record> a;
    while(true){Record r{};f.read((char*)&r.step,4);if(f.eof()&&f.gcount()==0)break;assert(f.gcount()==4);
        f.read((char*)&r.pos,4);f.read((char*)r.ids.data(),480);f.read((char*)r.hits.data(),40);assert(f);
        assert(p>>r.decode>>r.turn);assert(r.turn>=0&&r.turn<100);for(auto e:r.ids)assert(e<384);a.push_back(r);}
    return a;
}
static std::vector<Layer> load_shapes(const std::string &path){
    std::ifstream f(path);assert(f);std::vector<Layer> a(41);
    for(auto &l:a)assert(f>>l.pre_b>>l.pre_f>>l.post_b>>l.post_f>>l.shared_b>>l.shared_f);
    return a;
}

class Model {
public:
    Cfg c;sim2::Engine e;cache2::Cache cache;
    std::vector<Layer> shapes;
    std::array<double,7> fixed{},compute{};
    double input_wait=0,return_wait=0;
    double common_path=0,attention_path=0,moe_input=0,parallel=0,cpu_tail=0,gpu_tail=0,merge_path=0,other_path=0;
    int step_ready=0,step_gpu=0,step_cpu=0;
    Model(Cfg cfg,std::vector<Layer> shape):c(cfg),e(resource_cfg(cfg)),cache(cache_cfg(cfg),e),shapes(std::move(shape)){
        std::ifstream f(c.cpu_file);assert(f);
        for(auto &v:fixed)assert(f>>v);
        for(auto &v:compute)assert(f>>v);
    }
    static sim2::Config resource_cfg(const Cfg &c){
        sim2::Config r;r.host_total_GBs=c.host_bw*c.mem_scale;r.cpu_read_GBs=45*c.mem_scale;
        r.gpu_total_GBs=c.gpu_peak_bw;r.h2d_GBs=c.h2d_bw;r.d2h_GBs=c.d2h_bw;
        r.weight_chunk_bytes=c.chunk;r.copy_setup_us=c.copy_setup_us;return r;
    }
    static cache2::Config cache_cfg(const Cfg &c){
        cache2::Config r;r.slots=c.slots;r.policy=c.policy;r.budget=c.budget;r.queue=c.queue;
        r.decay=c.decay;r.margin=c.margin;r.min_heat=c.min_heat;return r;
    }
    sim2::Work gpu(double b,double f,double delay,double bw)const{return {b,f/(c.gpu_GF*1e6),delay,bw};}
    sim2::Work cpu(int n)const{return {n*EB,compute[n]*c.cpu_compute_scale,fixed[n]*c.cpu_fixed_scale,45*c.mem_scale};}
    double cpu_time(int n)const{
        return n?fixed[n]*c.cpu_fixed_scale+std::max(n*EB/(45*c.mem_scale*1e6),compute[n]*c.cpu_compute_scale):0;
    }
    double expert_bw(int n)const{
        // Occupancy=0 uses a constant rate. Occupancy=1 gives 0.65x at one expert, saturating at six.
        const double factors[]={0,.65,.8,.9,.95,.98,1};
        return c.expert_bw*(1-c.occupancy*(1-factors[n]));
    }
    sim2::Work expert_work(int n)const{return n?gpu(n*EB,n*EF,c.expert_us*.001,expert_bw(n)):sim2::Work{};}
    double nominal(const sim2::Work &w)const{
        return w.delay_ms+std::max(w.compute_ms,w.bytes/((w.read_GBs>0?w.read_GBs:c.gpu_peak_bw)*1e6));
    }
    void run_gpu(const sim2::Work &w){e.enqueue_gpu(w);e.wait_until([&]{return e.gpu_idle();});}
    void attention(int l,int N){
        const int window=std::min(N,128);
        const int global=l<2?0:std::min(N/(l<20?2:1),512);
        int index=0;
        if(l==2||l==8||l==14)index=N/2;
        if(l==20)index=N;
        if(l==24||l==28||l==32||l==36)index=c.candidate_first?std::min(N,16384):N;
        const double flops=(window+global)*4.0*64*512+index*2.0*32*128;
        const double bytes=window*528.0+global*288.0+index*68.0;
        run_gpu({bytes,flops/(c.attention_GF*1e6),c.common_us*.001/3,c.common_bw});
    }
    void step(const Record &r){
        cache.begin_token();step_ready=step_gpu=step_cpu=0;
        common_path=attention_path=moe_input=parallel=cpu_tail=gpu_tail=merge_path=other_path=0;
        double part_start=e.now_ms;
        // Embedding and Engram table lookup stay in host RAM. Constant 0.1ms control work, no HDD tier.
        e.run_for(.1);auto emb=e.add_h2d_urgent(10240+12672,c.transfer_us);
        e.wait_until([&]{return e.transfer_done(emb);});
        other_path+=e.now_ms-part_start;
        for(int l=0;l<40;++l){
            const auto &s=shapes[l];
            part_start=e.now_ms;
            run_gpu(gpu(s.pre_b,s.pre_f,c.common_us*.001/3,c.common_bw));
            common_path+=e.now_ms-part_start;part_start=e.now_ms;
            attention(l,c.context_base+int(r.pos)+1);
            attention_path+=e.now_ms-part_start;part_start=e.now_ms;
            run_gpu(gpu(s.post_b,s.post_f,c.common_us*.001/3,c.common_bw));
            common_path+=e.now_ms-part_start;
            const auto shared=gpu(s.shared_b,s.shared_f,c.expert_us*.001,c.common_bw);
            double gpu_finish=e.now_ms;
            e.enqueue_gpu(shared,[&]{gpu_finish=e.now_ms;});
            const double input_start=e.now_ms;
            const auto input=e.add_d2h(10240+48,c.transfer_us);
            e.wait_until([&]{return e.transfer_done(input);});
            input_wait+=e.now_ms-input_start;
            moe_input+=e.now_ms-input_start;
            const double parallel_start=e.now_ms;
            std::vector<int> keys,available;
            for(int j=0;j<6;++j)keys.push_back(l*384+r.ids[l*6+j]);
            for(int key:keys)if(cache.ready(key))available.push_back(key);
            step_ready+=int(available.size());
            cache.observe_layer(keys);cache.lock(available);
            int h=int(available.size());
            if(c.balance){
                double best=1e100;
                const auto remaining=e.gpu_remaining_work();
                const double remaining_shared=remaining?nominal(*remaining):0;
                for(int n=0;n<=int(available.size());++n){
                    const double ct=cpu_time(6-n)+(n<6?c.transfer_us*.001+20480/(c.h2d_bw*1e6):0);
                    const double gt=remaining_shared+nominal(expert_work(n));
                    const double t=std::max(ct,gt);
                    if(t<best){best=t;h=n;}
                }
            }
            std::vector<int> on_gpu(available.begin(),available.begin()+h);
            std::vector<int> on_cpu(available.begin()+h,available.end());
            if(c.policy==2)cache.admit_lru(keys); // all available keys locked before replacement
            cache.unlock(on_cpu);
            if(h)e.enqueue_gpu(expert_work(h),[this,on_gpu,&gpu_finish]{cache.unlock(on_gpu);gpu_finish=e.now_ms;});
            const int misses=6-h;step_gpu+=h;step_cpu+=misses;
            if(misses){
                e.start_cpu(cpu(misses));
                e.wait_until([&]{return e.cpu_idle();});
                const double return_start=e.now_ms;
                const auto output=e.add_h2d_urgent(20480,c.transfer_us);
                e.wait_until([&]{return e.transfer_done(output);});
                return_wait+=e.now_ms-return_start;
            }
            const double cpu_finish=e.now_ms;
            e.wait_until([&]{return e.gpu_idle();});
            parallel+=std::max(0.0,std::min(cpu_finish,gpu_finish)-parallel_start);
            cpu_tail+=std::max(0.0,cpu_finish-std::max(gpu_finish,parallel_start));
            gpu_tail+=std::max(0.0,gpu_finish-std::max(cpu_finish,parallel_start));
            part_start=e.now_ms;
            run_gpu({81920,0,c.merge_us*.001,c.common_bw});
            merge_path+=e.now_ms-part_start;
        }
        cache.finish_token(r.decode);
        part_start=e.now_ms;
        const auto &head=shapes[40];run_gpu(gpu(head.pre_b,head.pre_f,.1,c.common_bw));
        common_path+=e.now_ms-part_start;part_start=e.now_ms;
        auto token=e.add_d2h(4,c.transfer_us);e.wait_until([&]{return e.transfer_done(token);});
        other_path+=e.now_ms-part_start;
        cache.check();
    }
};

int main(int argc,char**argv){
    if(argc!=4)return 2;
    std::ifstream configs(argv[1]);std::ofstream out(argv[2]);assert(configs&&out);
    out<<"id\tcard\ttrace\tturn\tsteps\tms_token\ttok_s\tready_fraction\tgpu_fraction\tcpu_experts_token\tcopy_MB_token\tcopies_token\tinput_wait_ms\treturn_wait_ms\tcpu_busy_ms\tgpu_busy_ms\tp50_ms\tp95_ms\thost_peak_GBs\tgpu_peak_GBs\tcommon_path_ms\tattention_path_ms\tmoe_input_ms\tparallel_ms\tcpu_tail_ms\tgpu_tail_ms\tmerge_path_ms\tother_path_ms\n";
    std::map<std::string,std::vector<Record>> traces;std::map<std::string,std::vector<Layer>> shapes;
    int cases=0;
    while(true){Cfg c;if(!(configs>>c.id))break;
        assert(configs>>c.card>>c.trace>>c.phases>>c.shapes>>c.cpu_file>>c.slots>>c.policy>>c.budget>>c.queue>>c.balance>>c.context_base>>c.candidate_first>>c.detail>>
            c.decay>>c.margin>>c.min_heat>>c.usable_gib>>c.workspace_gib>>c.mem_scale>>c.cpu_fixed_scale>>c.cpu_compute_scale>>
            c.common_bw>>c.expert_bw>>c.gpu_peak_bw>>c.gpu_GF>>c.attention_GF>>c.host_bw>>c.h2d_bw>>c.d2h_bw>>c.chunk>>
            c.copy_setup_us>>c.transfer_us>>c.common_us>>c.expert_us>>c.merge_us>>c.occupancy);
        const auto trace_key=c.trace+"\n"+c.phases;
        if(!traces.count(trace_key))traces[trace_key]=load_trace(c.trace,c.phases);
        if(!shapes.count(c.shapes))shapes[c.shapes]=load_shapes(c.shapes);
        Model model(c,shapes[c.shapes]);Totals total;std::array<Totals,100> turns;
        std::ofstream details;if(c.detail){details.open(std::string(argv[3])+"/"+c.id+".csv");assert(details);details<<"step,turn,decode,ms,ready,gpu,cpu,copies,weight_MB,input_wait_ms,return_wait_ms\n";}
        for(const auto&r:traces[trace_key]){
            const double begin=model.e.now_ms, iw=model.input_wait,rw=model.return_wait;
            const auto before=model.e.stats;const auto copies=model.cache.stats.submitted;
            model.step(r);const auto after=model.e.stats;
            const double parts=model.common_path+model.attention_path+model.moe_input+model.parallel+model.cpu_tail+model.gpu_tail+model.merge_path+model.other_path;
            assert(std::abs(parts-(model.e.now_ms-begin))<1e-6);
            auto add=[&](Totals &t){++t.n;t.time+=model.e.now_ms-begin;t.ready+=model.step_ready;t.gpu_used+=model.step_gpu;t.cpu_used+=model.step_cpu;
                t.copies+=model.cache.stats.submitted-copies;t.weightbytes+=after.weight_bytes-before.weight_bytes;t.inputwait+=model.input_wait-iw;t.returnwait+=model.return_wait-rw;
                t.cpu_busy+=after.cpu_busy_ms-before.cpu_busy_ms;t.gpu_busy+=after.gpu_busy_ms-before.gpu_busy_ms;t.latency.push_back(model.e.now_ms-begin);
                t.common_path+=model.common_path;t.attention_path+=model.attention_path;t.moe_input+=model.moe_input;t.parallel+=model.parallel;
                t.cpu_tail+=model.cpu_tail;t.gpu_tail+=model.gpu_tail;t.merge_path+=model.merge_path;t.other_path+=model.other_path;};
            if(r.decode){add(total);add(turns[r.turn]);}
            if(c.detail)details<<r.step<<','<<r.turn<<','<<r.decode<<','<<model.e.now_ms-begin<<','<<model.step_ready<<','<<model.step_gpu<<','<<model.step_cpu<<','<<model.cache.stats.submitted-copies<<','<<(after.weight_bytes-before.weight_bytes)/1e6<<','<<model.input_wait-iw<<','<<model.return_wait-rw<<'\n';
        }
        auto write=[&](Totals t,int turn){if(!t.n)return;std::sort(t.latency.begin(),t.latency.end());auto q=[&](double x){return t.latency[size_t(x*(t.latency.size()-1))];};
            out<<std::setprecision(12)<<c.id<<'\t'<<c.card<<'\t'<<c.trace<<'\t'<<turn<<'\t'<<t.n<<'\t'<<t.time/t.n<<'\t'<<1000*t.n/t.time<<'\t'<<t.ready/(t.n*240)<<'\t'<<t.gpu_used/(t.n*240)<<'\t'<<t.cpu_used/t.n<<'\t'<<t.weightbytes/1e6/t.n<<'\t'<<t.copies/t.n<<'\t'<<t.inputwait/t.n<<'\t'<<t.returnwait/t.n<<'\t'<<t.cpu_busy/t.n<<'\t'<<t.gpu_busy/t.n<<'\t'<<q(.5)<<'\t'<<q(.95)<<'\t'<<model.e.stats.peak_host_GBs<<'\t'<<model.e.stats.peak_gpu_GBs<<'\t'<<t.common_path/t.n<<'\t'<<t.attention_path/t.n<<'\t'<<t.moe_input/t.n<<'\t'<<t.parallel/t.n<<'\t'<<t.cpu_tail/t.n<<'\t'<<t.gpu_tail/t.n<<'\t'<<t.merge_path/t.n<<'\t'<<t.other_path/t.n<<'\n';};
        write(total,-1);for(int t=0;t<100;++t)write(turns[t],t);out.flush();
        std::cout<<"done "<<++cases<<' '<<c.id<<' '<<total.time/total.n<<" ms/token\n"<<std::flush;
    }
}
