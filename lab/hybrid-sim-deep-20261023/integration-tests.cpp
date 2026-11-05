// Exercise the production Model and resource/cache classes without modifying
// their source. The renamed CLI main is never called by these tests.
#define main deep_sim_cli_main_for_integration
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#include "deep-sim.cpp"
#pragma GCC diagnostic pop
#undef main
#include <cstdlib>

static void near_value(double a,double b,const char *message) {
    if(std::abs(a-b)>1e-9*std::max(1.0,std::abs(b))) {
        std::cerr<<message<<": got "<<std::setprecision(16)<<a<<", expected "<<b<<'\n';std::abort();
    }
}
static Cfg fixture(const char *cpu_file) {
    Cfg c{};c.id="integration";c.card="P40";c.cpu_file=cpu_file;
    c.slots=0;c.policy=0;c.budget=8;c.queue=256;c.balance=1;c.candidate_first=1;
    c.decay=.97;c.margin=1.15;c.min_heat=1.5;c.usable_gib=24;c.workspace_gib=2;
    c.mem_scale=c.cpu_fixed_scale=c.cpu_compute_scale=1;
    c.common_bw=210;c.expert_bw=210;c.gpu_peak_bw=346;c.gpu_GF=5000;c.attention_GF=1000;
    c.host_bw=48;c.h2d_bw=c.d2h_bw=12;c.chunk=1048576;c.copy_setup_us=3;
    c.transfer_us=20;c.common_us=100;c.expert_us=20;c.merge_us=5;c.occupancy=1;
    return c;
}
static Record record(int step=0) {
    Record r{};r.step=step;r.pos=step;r.decode=1;r.turn=0;
    for(int l=0;l<40;++l)for(int j=0;j<6;++j)r.ids[l*6+j]=j;
    return r;
}
static double dynamic_read_bytes(const Cfg &c,const Record &r) {
    double sum=0;const int N=c.context_base+int(r.pos)+1;
    for(int l=0;l<40;++l) {
        const int w=std::min(N,128),g=l<2?0:std::min(N/(l<20?2:1),512);
        int i=0;
        if(l==2||l==8||l==14)i=N/2;
        if(l==20)i=N;
        if(l==24||l==28||l==32||l==36)i=c.candidate_first?std::min(N,16384):N;
        sum+=w*528.0+g*288.0+i*68.0;
    }
    return sum;
}
static double common_read_bytes(const std::vector<Layer> &shapes) {
    double bytes=0;for(const auto &s:shapes)bytes+=s.pre_b+s.post_b+s.shared_b;return bytes;
}
static void check_limits(const Model &m) {
    const auto &s=m.e.stats;
    assert(s.peak_host_GBs<=m.e.cfg.host_total_GBs*(1+1e-10));
    assert(s.peak_gpu_GBs<=m.e.cfg.gpu_total_GBs*(1+1e-10));
    assert(s.host_bytes<=m.e.cfg.host_total_GBs*1e6*m.e.now_ms*(1+1e-10));
    assert(s.vram_bytes<=m.e.cfg.gpu_total_GBs*1e6*m.e.now_ms*(1+1e-10));
}
static void checked_step(Model &m,const Record &r) {
    const double before=m.e.now_ms;m.step(r);
    const std::array<double,8> pieces{m.common_path,m.attention_path,m.moe_input,m.parallel,
        m.cpu_tail,m.gpu_tail,m.merge_path,m.other_path};
    double sum=0;for(double piece:pieces){assert(piece>=0);sum+=piece;}
    near_value(sum,m.e.now_ms-before,"critical-path partition equals token time");
}
static void warm_all(Model &m) {
    std::vector<int> keys;
    for(int l=0;l<40;++l)for(int j=0;j<6;++j)keys.push_back(l*384+j);
    m.cache.observe_layer(keys);m.cache.admit_lru(keys);m.e.drain();m.cache.check();
    for(int k:keys)assert(m.cache.ready(k));
}

int main(int argc,char **argv) {
    if(argc!=9) {std::cerr<<"usage: integration-tests CPU_TSV SHAPES_TSV FULL_MS_1 ... FULL_MS_6\n";return 2;}
    const auto shapes=load_shapes(argv[2]);int checks=0;
    for(int n=1;n<=6;++n) {
        Model m(fixture(argv[1]),shapes);
        const double full=std::stod(argv[2+n]);
        near_value(m.cpu_time(n),full,"CPU nominal calibration reconstruction");
        m.e.start_cpu(m.cpu(n));m.e.wait_until([&]{return m.e.cpu_idle();});
        near_value(m.e.now_ms,full,"CPU event calibration reconstruction");
        near_value(m.e.stats.cpu_bytes,n*EB,"CPU expert byte count");check_limits(m);++checks;
    }
    for(double read_ref:{40.0,45.0,50.0,55.0})for(int n=1;n<=6;++n) {
        auto c=fixture(argv[1]);c.mem_scale=read_ref/45;c.host_bw=48/c.mem_scale;
        Model m(c,shapes);const double full=std::stod(argv[2+n]);
        // Keep the same measured six totals and cache-compute proxy while
        // changing the otherwise unidentifiable read/fixed decomposition.
        m.fixed[n]=full-std::max(n*EB/(read_ref*1e6),m.compute[n]);
        assert(m.fixed[n]>=0);near_value(m.e.cfg.host_total_GBs,48,"refit physical host cap");
        near_value(m.e.cfg.cpu_read_GBs,read_ref,"refit read-rate demand");
        near_value(m.cpu_time(n),full,"refit nominal CPU reconstruction");
        m.e.start_cpu(m.cpu(n));m.e.wait_until([&]{return m.e.cpu_idle();});
        near_value(m.e.now_ms,full,"refit Engine CPU reconstruction");
        check_limits(m);++checks;
    }
    {
        // With no expert residency, every routed payload is read once by CPU.
        // Common GPU weights, sparse attention and merge are distinct reads.
        Model m(fixture(argv[1]),shapes);const auto r=record(31);checked_step(m,r);
        assert(m.step_ready==0 && m.step_gpu==0 && m.step_cpu==240);
        const double h2d=10240+12672+40*20480,d2h=40*(10240+48)+4;
        const double gpu=common_read_bytes(shapes)+dynamic_read_bytes(m.c,r)+40*81920;
        near_value(m.e.stats.cpu_bytes,240*EB,"uncached routed CPU reads");
        near_value(m.e.stats.gpu_bytes,gpu,"common GPU reads counted once");
        near_value(m.e.stats.h2d_bytes,h2d,"embedding and 40 CPU result copies");
        near_value(m.e.stats.d2h_bytes,d2h,"40 routing/input copies and sampled token");
        near_value(m.e.stats.weight_bytes,0,"no background weight copies");
        near_value(m.e.stats.host_bytes,240*EB+h2d+d2h,"host read/write ledger");
        near_value(m.e.stats.vram_bytes,gpu+h2d+d2h,"VRAM read/write ledger");
        check_limits(m);++checks;
    }
    for(bool force_cpu:{false,true}) {
        auto c=fixture(argv[1]);c.slots=240;c.policy=2;c.balance=force_cpu?1:0;
        if(force_cpu)c.expert_bw=.05;
        Model m(c,shapes);warm_all(m);const auto before=m.e.stats;const auto r=record(31);checked_step(m,r);
        assert(m.step_ready==240);
        assert(m.step_gpu==(force_cpu?0:240) && m.step_cpu==(force_cpu?240:0));
        const double h2d=10240+12672+(force_cpu?40*20480:0),d2h=40*(10240+48)+4;
        const double cpu=force_cpu?240*EB:0;
        const double gpu=common_read_bytes(shapes)+dynamic_read_bytes(c,r)+40*81920+(force_cpu?0:240*EB);
        near_value(m.e.stats.cpu_bytes-before.cpu_bytes,cpu,"resident routed CPU assignment bytes");
        near_value(m.e.stats.gpu_bytes-before.gpu_bytes,gpu,"resident routed GPU assignment bytes");
        near_value(m.e.stats.weight_bytes-before.weight_bytes,0,"warm residents need no extra DMA");
        near_value(m.e.stats.h2d_bytes-before.h2d_bytes,h2d,"warm assignment H2D ledger");
        near_value(m.e.stats.d2h_bytes-before.d2h_bytes,d2h,"host routing design D2H ledger");
        near_value(m.e.stats.host_bytes-before.host_bytes,cpu+h2d+d2h,"warm assignment host ledger");
        near_value(m.e.stats.vram_bytes-before.vram_bytes,gpu+h2d+d2h,"warm assignment VRAM ledger");
        check_limits(m);m.cache.check();++checks;
    }
    {
        auto a=fixture(argv[1]);a.slots=64;a.policy=3;
        auto b=a;b.card="V100";b.id="name_only_changed";
        Model x(a,shapes),y(b,shapes);
        for(int t=0;t<16;++t) {
            auto r=record(t);checked_step(x,r);checked_step(y,r);
            near_value(x.e.now_ms,y.e.now_ms,"card name must not change resource behavior");
            assert(x.step_ready==y.step_ready && x.step_gpu==y.step_gpu && x.step_cpu==y.step_cpu);
            near_value(x.e.stats.host_bytes,y.e.stats.host_bytes,"card name host bytes");
            near_value(x.e.stats.vram_bytes,y.e.stats.vram_bytes,"card name VRAM bytes");
            assert(x.cache.stats.submitted==y.cache.stats.submitted);
            check_limits(x);check_limits(y);
        }
        ++checks;
    }
    std::cout<<"integration tests passed: "<<checks<<'\n';
}
