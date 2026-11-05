#include "resources.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using sim2::Config;
using sim2::Engine;
using sim2::Work;

static void near(double got,double expected,const char *label) {
    if (std::abs(got-expected) > 1e-8*std::max(1.0,std::abs(expected))) {
        std::cerr << label << ": got " << got << ", expected " << expected << '\n';
        std::abort();
    }
}
static Config free_bus() {
    Config c; c.host_total_GBs=100; c.cpu_read_GBs=8; c.gpu_total_GBs=100;
    c.h2d_GBs=1; c.d2h_GBs=1; c.weight_chunk_bytes=0; return c;
}

int main() {
    int tests=0;
    // All byte sizes below are decimal MB; 1 GB/s transfers 1 MB per ms.
    for (const auto &row : std::vector<std::pair<double,double>>{{0,8.25},{1e6,2.25},{.25e6,1.75}}) {
        auto c=free_bus(); c.weight_chunk_bytes=row.first; Engine e(c);
        int started=0,completed=0;
        e.on_start=[&](int key){assert(key==7);++started;};
        e.on_complete=[&](int key){assert(key==7);++completed;};
        e.enqueue_weight(7,8e6); e.run_for(1.3);
        const auto id=e.add_h2d_urgent(.25e6);
        e.wait_until([&]{return e.transfer_done(id);});
        near(e.now_ms,row.second,"priority short H2D completion");
        e.drain(); near(e.now_ms,8.25,"all H2D bytes conserved");
        near(e.stats.weight_bytes,8e6,"weight payload");
        near(e.stats.h2d_bytes,8.25e6,"total H2D payload");
        assert(started==1 && completed==1); ++tests;
    }
    {
        Engine e(free_bus()); e.enqueue_weight(1,8e6); e.run_for(2);
        auto id=e.add_d2h(1e6); e.wait_until([&]{return e.transfer_done(id);});
        near(e.now_ms,3,"independent D2H"); e.drain(); near(e.now_ms,8,"independent H2D"); ++tests;
    }
    {
        // A's slot is reserved at time zero but remains readable while the
        // callback denies overwrite. A GPU reader extends the lock to 6 ms.
        Engine e(free_bus()); bool readable=true; double writing=-1,ready=-1;
        e.can_start=[&](int){return e.now_ms>=6;};
        e.on_start=[&](int){readable=false;writing=e.now_ms;};
        e.on_complete=[&](int){ready=e.now_ms;};
        e.enqueue_weight(3,2e6); e.run_for(4); assert(readable);
        e.enqueue_gpu(Work{0,2,0}); e.wait_until([&]{return e.gpu_idle();});
        near(writing,6,"overwrite after reader"); e.drain(); near(ready,8,"replacement ready"); ++tests;
    }
    for (double gpu_time : {4.0,9.0}) {
        Engine e(free_bus()); e.run_for(2); e.enqueue_gpu(Work{0,gpu_time,0});
        auto input=e.add_d2h(.5e6); e.wait_until([&]{return e.transfer_done(input);});
        near(e.now_ms,2.5,"CPU input dependency"); e.start_cpu(Work{0,5,0});
        e.wait_until([&]{return e.cpu_idle();}); near(e.now_ms,7.5,"CPU computation");
        auto output=e.add_h2d_urgent(.25e6);
        e.wait_until([&]{return e.transfer_done(output) && e.gpu_idle();});
        near(e.now_ms,gpu_time==4 ? 7.75 : 11,"layer join");
        e.run_for(1); near(e.now_ms,gpu_time==4 ? 8.75 : 12,"serial suffix"); ++tests;
    }
    {
        auto c=free_bus(); c.host_total_GBs=8; c.cpu_read_GBs=8; c.h2d_GBs=4;
        Engine e(c); e.start_cpu(Work{12e6,0,0}); auto id=e.add_h2d_urgent(4e6);
        // Fair proportional sharing gives CPU:DMA=8:4, hence 16/3 and 8/3
        // GB/s initially. DMA ends at 1.5 ms; CPU completes at 2 ms.
        e.wait_until([&]{return e.transfer_done(id);}); near(e.now_ms,1.5,"fair DMA completion");
        e.drain(); near(e.now_ms,2,"host conserved completion");
        near(e.stats.host_bytes,16e6,"host bytes"); near(e.stats.peak_host_GBs,8,"host peak"); ++tests;
    }
    {
        auto c=free_bus(); c.gpu_total_GBs=8; c.h2d_GBs=4;
        Engine e(c); e.enqueue_gpu(Work{12e6,0,0,8}); auto id=e.add_h2d_urgent(4e6);
        e.wait_until([&]{return e.transfer_done(id);}); near(e.now_ms,1.5,"fair VRAM DMA completion");
        e.drain(); near(e.now_ms,2,"VRAM conserved completion");
        near(e.stats.vram_bytes,16e6,"VRAM bytes"); near(e.stats.peak_gpu_GBs,8,"VRAM peak"); ++tests;
    }
    for (double bandwidth : {8.0,16.0}) {
        auto c=free_bus(); c.host_total_GBs=bandwidth; c.cpu_read_GBs=bandwidth; c.h2d_GBs=4;
        Engine e(c); e.start_cpu(Work{12e6,3,0}); e.add_h2d_urgent(4e6);
        e.drain(); near(e.now_ms,3,"independent arithmetic floor"); ++tests;
    }
    {
        auto c=free_bus(); c.host_total_GBs=10; c.gpu_total_GBs=2; c.cpu_read_GBs=8; c.h2d_GBs=8;
        Engine e(c); e.start_cpu(Work{8e6,0,0}); e.add_h2d_urgent(2e6);
        // VRAM freezes DMA at 2. CPU may then use the unused 8 host GB/s.
        e.drain(); near(e.now_ms,1,"progressive refill of spare host bandwidth");
        near(e.stats.peak_host_GBs,10,"two resource host limit"); near(e.stats.peak_gpu_GBs,2,"two resource GPU limit"); ++tests;
    }
    {
        Engine e(free_bus()); e.start_cpu(Work{8e6,3,2}); e.drain();
        near(e.now_ms,5,"serial delay followed by concurrent memory and compute"); ++tests;
    }
    {
        auto c=free_bus(); c.weight_chunk_bytes=1e6; c.copy_setup_us=100;
        Engine e(c); e.enqueue_weight(0,2e6); e.drain(); near(e.now_ms,2.2,"per chunk setup");
        auto id=e.add_h2d_urgent(1e6,200); e.wait_until([&]{return e.transfer_done(id);});
        near(e.now_ms,3.4,"explicit urgent latency not double counted"); ++tests;
    }
    {
        Engine e(free_bus()); bool allow=false; std::vector<int> order;
        e.can_start=[&](int key){return key!=1 || allow;};
        e.on_start=[&](int key){order.push_back(key);};
        e.enqueue_weight(1,1e6); e.enqueue_weight(2,1e6);
        e.run_for(1); assert(order==std::vector<int>{2}); allow=true;
        e.drain(); assert((order==std::vector<int>{2,1})); near(e.now_ms,2,"skip blocked pending weight"); ++tests;
    }
    {
        Engine e(free_bus()); int completed=0;
        e.enqueue_gpu(Work{0,1,0},[&]{++completed;e.enqueue_gpu(Work{0,2,0});});
        e.drain(); near(e.now_ms,3,"callback enqueues next GPU work"); assert(completed==1); ++tests;
    }
    {
        auto c=free_bus(); c.host_total_GBs=10; c.gpu_total_GBs=6;
        c.cpu_read_GBs=8; c.h2d_GBs=5; c.d2h_GBs=4;
        c.weight_chunk_bytes=.2e6; c.copy_setup_us=5;
        Engine e(c); int completed=0;
        e.on_complete=[&](int){++completed;};
        e.start_cpu(Work{18e6,4,.2});
        e.enqueue_gpu(Work{21e6,3,.3,5}); e.enqueue_gpu(Work{5e6,2,0,4});
        for (int k=1;k<=4;++k) e.enqueue_weight(k,k*1e6);
        const auto first=e.add_h2d_urgent(1e6,10), reverse=e.add_d2h(2e6,20);
        e.run_for(1); const auto second=e.add_h2d_urgent(.5e6,15); e.drain();
        assert(e.transfer_done(first) && e.transfer_done(second) && e.transfer_done(reverse));
        assert(completed==4 && e.pending_weights()==0);
        near(e.stats.cpu_bytes,18e6,"mixed CPU bytes"); near(e.stats.gpu_bytes,26e6,"mixed GPU bytes");
        near(e.stats.weight_bytes,10e6,"mixed weight bytes"); near(e.stats.h2d_bytes,11.5e6,"mixed H2D bytes");
        near(e.stats.d2h_bytes,2e6,"mixed D2H bytes"); near(e.stats.host_bytes,31.5e6,"mixed host bytes");
        near(e.stats.vram_bytes,39.5e6,"mixed VRAM bytes");
        assert(e.stats.peak_host_GBs <= c.host_total_GBs*(1+1e-10));
        assert(e.stats.peak_gpu_GBs <= c.gpu_total_GBs*(1+1e-10));
        assert(e.stats.host_bytes <= c.host_total_GBs*1e6*e.now_ms*(1+1e-10));
        assert(e.stats.vram_bytes <= c.gpu_total_GBs*1e6*e.now_ms*(1+1e-10)); ++tests;
    }
    {
        Engine e(free_bus()); e.can_start=[](int){return false;}; e.enqueue_weight(1,1e6);
        bool caught=false;
        try { e.drain(); } catch (const std::runtime_error &) { caught=true; }
        assert(caught); ++tests;
    }
    {
        auto c=free_bus(); c.gpu_total_GBs=8; c.h2d_GBs=4;
        Engine e(c); assert(!e.gpu_remaining_work());
        e.enqueue_gpu(Work{12e6,0,.5,8}); e.add_h2d_urgent(4e6); e.run_for(1);
        // The first 0.5 ms is setup. The next 0.5 ms shares VRAM 8:4, so
        // 8/3 MB of GPU data have completed, not the uncontented 4 MB.
        auto remaining=e.gpu_remaining_work(); assert(remaining);
        near(remaining->bytes,(12-8.0/3)*1e6,"actual remaining GPU bytes");
        near(remaining->delay_ms,0,"remaining GPU delay"); near(remaining->read_GBs,8,"remaining GPU demand");
        remaining->bytes=0;
        near(e.gpu_remaining_work()->bytes,(12-8.0/3)*1e6,"GPU snapshot is independent");
        e.drain(); assert(!e.gpu_remaining_work()); ++tests;
    }
    std::cout << "resource tests passed: " << tests << '\n';
}
