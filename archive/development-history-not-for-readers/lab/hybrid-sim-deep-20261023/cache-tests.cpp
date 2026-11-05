#include "cache.hpp"
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

static void near(double a,double b,double tolerance=1e-8) { assert(std::abs(a-b)<tolerance); }
template<class F> static void throws(F f) {
    bool caught=false;try { f(); } catch (const std::exception &) { caught=true; }assert(caught);
}
static sim2::Config resources() {
    sim2::Config c;c.host_total_GBs=100;c.cpu_read_GBs=50;c.gpu_total_GBs=100;
    c.h2d_GBs=1;c.d2h_GBs=1;c.weight_chunk_bytes=100000;c.copy_setup_us=0;return c;
}
static cache2::Config config(int slots=2,int policy=2) {
    cache2::Config c;c.slots=slots;c.policy=policy;c.queue=8;c.expert_bytes=1000000;return c;
}
static void fill(cache2::Cache &c,sim2::Engine &e,const std::vector<int> &keys) {
    c.begin_token();c.observe_layer(keys);c.admit_lru(keys);e.drain();c.check();
    for(int k:keys)assert(c.ready(k));
}

int main() {
    int count=0;
    auto test=[&](const char *name,const std::function<void()> &body){body();++count;std::cout<<"PASS "<<name<<'\n';};
    test("reservation-writing-ready and old mapping",[]{
        sim2::Engine e(resources());cache2::Cache c(config(1),e);fill(c,e,{10});
        c.lock({10});c.observe_layer({20});c.admit_lru({20});
        assert(c.ready(10)&&!c.ready(20)&&c.pending(20));c.check();
        e.run_for(.4);assert(c.ready(10)&&c.pending(20));
        c.unlock({10});e.run_for(0);
        assert(!c.ready(10)&&!c.ready(20)&&c.pending(20));c.check();
        e.run_for(.5);assert(!c.ready(20));c.check();
        e.drain();assert(c.ready(20)&&!c.pending(20)&&!c.ready(10));
        near(e.now_ms,2.4);assert(c.stats.overwrites==1);c.check();
    });
    test("queued victim remains readable before DMA start",[]{
        sim2::Engine e(resources());cache2::Cache c(config(1),e);fill(c,e,{10});
        auto id=e.add_h2d_urgent(1000000);c.observe_layer({20});c.admit_lru({20});
        assert(c.ready(10)&&!c.ready(20));e.run_for(.5);assert(c.ready(10));
        e.wait_until([&]{return e.transfer_done(id);});
        assert(!c.ready(10)&&!c.ready(20));e.drain();assert(c.ready(20));c.check();
    });
    test("entire six-expert snapshot is locked before LRU admission",[]{
        sim2::Engine e(resources());cache2::Cache c(config(1),e);fill(c,e,{5});
        // The hit appears LAST in routing order. A one-pass implementation
        // would enqueue expert 0 and destroy expert 5 before counting its hit.
        const std::vector<int> route{0,1,2,3,4,5};std::vector<int> hits;
        for(int k:route)if(c.ready(k))hits.push_back(k);
        assert(hits==std::vector<int>{5});c.lock(hits);c.observe_layer(route);c.admit_lru(route);
        assert(c.ready(5)&&c.stats.submitted==2&&c.pending_count()==1);c.check();
        e.enqueue_gpu({0,.75,0,0},[&]{assert(c.ready(5));c.unlock(hits);});
        e.wait_until([&]{return e.gpu_idle();});assert(!c.ready(5));
        e.drain();assert(c.ready(0));c.check();
    });
    test("six resident experts cannot be invalidated during common GPU work",[]{
        sim2::Engine e(resources());cache2::Cache c(config(6),e);fill(c,e,{0,1,2,3,4,5});
        const std::vector<int> hits{0,1,2,3,4,5};c.lock(hits);c.observe_layer({6,7,8,9,10,11});c.admit_lru({6,7,8,9,10,11});
        assert(c.pending_count()==6);e.enqueue_gpu({0,.5,0,0},[&]{for(int k:hits)assert(c.ready(k));c.unlock(hits);});
        e.wait_until([&]{return e.gpu_idle();});c.check();e.drain();c.check();
        for(int k:hits)assert(!c.ready(k));
        for(int k:{6,7,8,9,10,11})assert(c.ready(k));
    });
    test("queue and slot limits and duplicate pending key",[]{
        auto cfg=config(2);cfg.queue=1;sim2::Engine e(resources());cache2::Cache c(cfg,e);
        c.observe_layer({0,1});c.admit_lru({0,0,1});
        assert(c.stats.submitted==1&&c.stats.rejected==1&&c.stats.peak_pending==1);c.check();
        e.drain();c.admit_lru({1});e.drain();assert(c.ready(0)&&c.ready(1));
        c.lock({0,1});c.observe_layer({2,3});c.admit_lru({2,3});assert(c.pending_count()==1);
        c.unlock({0,1});e.drain();c.check();
    });
    test("reserved physical slots cannot be reserved twice",[]{
        sim2::Engine e(resources());cache2::Cache c(config(1),e);fill(c,e,{0});
        c.lock({0});c.observe_layer({1,2});c.admit_lru({1,2,1});
        assert(c.stats.submitted==2&&c.pending(1)&&!c.pending(2));c.check();
        c.unlock({0});e.drain();assert(c.ready(1));c.check();
    });
    test("copy engine bypasses a locked queued victim",[]{
        sim2::Engine e(resources());cache2::Cache c(config(2),e);fill(c,e,{0,1});
        c.lock({0});c.observe_layer({2,3});c.admit_lru({2,3});
        assert(c.ready(0)&&!c.ready(1)&&c.pending(2)&&c.pending(3));
        e.run_for(1);assert(c.ready(3)&&c.ready(0)&&!c.ready(2));c.check();
        c.unlock({0});e.drain();assert(c.ready(2)&&c.ready(3));c.check();
    });
    test("reference-counted nested readers",[]{
        sim2::Engine e(resources());cache2::Cache c(config(1),e);fill(c,e,{0});
        c.lock({0});c.lock({0});c.observe_layer({1});c.admit_lru({1});
        c.unlock({0});e.run_for(.1);assert(c.ready(0));
        c.unlock({0});e.drain();assert(c.ready(1));c.check();
    });
    test("reader API failures are atomic",[]{
        sim2::Engine e(resources());cache2::Cache c(config(2),e);fill(c,e,{0,1});
        throws([&]{c.lock({0,2});});throws([&]{c.unlock({0});});
        throws([&]{c.lock({0,0});});throws([&]{c.unlock({0});});
        c.lock({0,1});throws([&]{c.unlock({0,0});});c.unlock({0,1});c.check();
        throws([&]{c.ready(-1);});throws([&]{c.observe_layer({0,cache2::key_count});});
    });
    test("frequency min heat, admission budget, margin",[]{
        auto cfg=config(2,3);cfg.budget=1;cfg.decay=1;sim2::Engine e(resources());cache2::Cache c(cfg,e);
        c.begin_token();c.observe_layer({0,1});c.finish_token(true);assert(c.stats.submitted==0);
        c.begin_token();c.observe_layer({0,1});c.finish_token(true);assert(c.stats.submitted==1);e.drain();assert(c.ready(0));
        c.finish_token(true);e.drain();assert(c.ready(1));
        c.observe_layer({2,2});c.finish_token(true);assert(!c.pending(2));
        c.observe_layer({2});c.finish_token(true);assert(c.pending(2));e.drain();assert(c.ready(2));c.check();
    });
    test("frequency promotion survives synchronous victim candidate insertion",[]{
        auto cfg=config(1,3);cfg.decay=1;cfg.min_heat=0;cfg.budget=1;
        sim2::Engine e(resources());cache2::Cache c(cfg,e);
        c.observe_layer({0});c.finish_token(true);e.drain();
        c.observe_layer({1,1});c.finish_token(true);assert(c.pending(1));e.drain();
        c.observe_layer({0,0,0});c.finish_token(true);assert(c.pending(0));e.drain();assert(c.ready(0));c.check();
    });
    test("prefill-only, no-cache, fixed-no-admission policies",[]{
        for(int policy:{0,1,4}){
            auto cfg=config(2,policy);cfg.min_heat=0;sim2::Engine e(resources());cache2::Cache c(cfg,e);
            c.begin_token();c.observe_layer({0});c.admit_lru({0});c.finish_token(true);
            assert(c.stats.submitted==0);c.finish_token(false);
            assert(c.stats.submitted==std::uint64_t(policy==1));e.drain();c.check();
        }
        sim2::Engine e(resources());cache2::Cache c(config(0),e);c.observe_layer({0});c.admit_lru({0});assert(!c.pending_count());c.check();
    });
    test("long heat history normalizes without changing EMA",[]{
        auto cfg=config(2,3);cfg.budget=0;sim2::Engine e(resources());cache2::Cache c(cfg,e);
        double a=0,b=0;
        for(int i=0;i<20000;++i){c.begin_token();a*=cfg.decay;b*=cfg.decay;
            c.observe_layer({0});a+=1;if(i%3==0){c.observe_layer({1});b+=1;}
            if(i%1000==0)c.check();}
        near(c.effective_heat(0),a,1e-9);near(c.effective_heat(1),b,1e-9);
        assert(c.stats.heat_normalizations>=4);c.check();
    });
    test("normalization also preserves candidate order",[]{
        auto cfg=config(1,3);cfg.decay=1e-70;cfg.min_heat=0;sim2::Engine e(resources());cache2::Cache c(cfg,e);
        c.begin_token();c.observe_layer({0,0,1});c.finish_token(true);e.drain();assert(c.ready(0));
        c.begin_token();c.observe_layer({1});c.finish_token(true);e.drain();assert(c.ready(1));
        assert(c.stats.heat_normalizations==2);c.check();
    });
    test("configuration and callback ownership validation",[]{
        sim2::Engine e(resources());auto cfg=config();cfg.slots=-1;throws([&]{cache2::Cache c(cfg,e);});
        cfg=config();cfg.queue=0;throws([&]{cache2::Cache c(cfg,e);});
        cfg=config();cfg.decay=0;throws([&]{cache2::Cache c(cfg,e);});
        {cache2::Cache c(config(),e);throws([&]{cache2::Cache second(config(),e);});}
        assert(!e.can_start&&!e.on_start&&!e.on_complete);
        {cache2::Cache c(config(),e);c.check();}
    });
    std::cout<<"All "<<count<<" cache tests passed\n";
}
