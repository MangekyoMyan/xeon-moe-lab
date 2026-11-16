#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "packed-generation.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <omp.h>
#include <sched.h>
#include <vector>

struct Capture {int threads=0;bool native=false;bool check_affinity=true;bool require_omp=false;std::array<int,24>seen{};};
static void inspect(ggml_tensor * dst,const ggml_tensor * a,int ith,int nth,void * opaque){
    auto & c=*static_cast<Capture*>(opaque);assert(nth==c.threads&&ith>=0&&ith<nth);
    if(c.require_omp)assert(omp_in_parallel()&&omp_get_num_threads()==nth);
    if(c.check_affinity){cpu_set_t mask;CPU_ZERO(&mask);assert(!sched_getaffinity(0,sizeof(mask),&mask));
        int wanted=c.native?(ith==0?23:ith-1):ith;
        assert(CPU_COUNT(&mask)==1&&CPU_ISSET(wanted,&mask)&&sched_getcpu()==wanted);
    }
    ++c.seen[ith];const auto n=ggml_nelements(a);
    for(int64_t i=ith;i<n;i+=nth)static_cast<float*>(dst->data)[i]=static_cast<const float*>(a->data)[i];
}
struct Abort {int calls=0;int stop_after=1;};
static bool abort_now(void * p){auto & a=*static_cast<Abort*>(p);return ++a.calls>=a.stop_after;}
int main(int argc,char ** argv){
    assert(argc==3);int hybrid=std::atoi(argv[2]);assert(hybrid==0||hybrid==1);
    assert(!setenv("GGML_CPU_HYBRID_POOL",argv[2],1));
    Dl_info provider{};assert(dladdr(reinterpret_cast<void*>(ggml_graph_compute),&provider));
    assert(std::filesystem::canonical(provider.dli_fname)==std::filesystem::canonical(argv[1]));
    auto mode=reinterpret_cast<int(*)(const ggml_threadpool*)>(dlsym(RTLD_DEFAULT,"xeon_threadpool_native_mode_v1"));assert(mode);
    for(uint32_t old:{0u,0x7ffe0001u,0x7fff000cu,0x80000018u,0xfffe000cu,0xffff0018u})for(int count:{1,6,12,24}){
        int transport;std::memcpy(&transport,&old,4);auto result=uint32_t(xeon_next_packed_graph(transport,count));
        assert(result==(((old+0x10000u)&0xffff0000u)|uint32_t(count)));
    }
    auto next=[](uint32_t old,int n){int i;std::memcpy(&i,&old,4);return uint32_t(xeon_next_packed_graph(i,n));};
    assert(next(0x7fff000cu,24)==0x80000018u);assert(next(0xffff0018u,24)==0x00000018u);
    auto cpu=ggml_backend_cpu_init();assert(cpu);
    auto batch_params=ggml_threadpool_params_default(24);batch_params.poll=0;batch_params.strict_cpu=true;batch_params.paused=false;
    for(int i=0;i<24;++i)batch_params.cpumask[i]=true;
    auto batch=ggml_threadpool_new(&batch_params);assert(batch&&mode(batch)==hybrid);
    auto gen_params=ggml_threadpool_params_default(12);gen_params.poll=0;gen_params.strict_cpu=true;gen_params.paused=true;
    for(int i=0;i<12;++i)gen_params.cpumask[i]=true;
    auto generation=ggml_threadpool_new(&gen_params);assert(generation&&mode(generation)==0);
    // Constructor policy is immutable, even if the environment changes afterwards.
    assert(!setenv("GGML_CPU_HYBRID_POOL",hybrid?"0":"1",1));assert(mode(batch)==hybrid&&mode(generation)==0);
    assert(!setenv("GGML_CPU_HYBRID_POOL",argv[2],1));
    auto ctx=ggml_init({8*1024*1024,nullptr,false});assert(ctx);
    constexpr int K=64,M=256,N=8;auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,M);auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);
    float * wf=ggml_get_data_f32(w),*xf=ggml_get_data_f32(x);
    for(int i=0;i<K*M;++i)wf[i]=float(i%7-3);for(int i=0;i<K*N;++i)xf[i]=float(i%5-2);
    Capture capture;auto y=ggml_map_custom1(ctx,ggml_scale(ctx,ggml_mul_mat(ctx,w,x),.5f),inspect,GGML_N_TASKS_MAX,&capture);
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto empty=ggml_new_graph(ctx);
    std::vector<float>expected(M*N);for(int j=0;j<N;++j)for(int m=0;m<M;++m){float value=0;for(int k=0;k<K;++k)value+=wf[m*K+k]*xf[j*K+k];expected[j*M+m]=value*.5f;}
    size_t compared=0;int aborts=0,graphs=0,affinity_checks=0;
    auto run=[&](ggml_threadpool * pool,int threads,bool abort,int requested=0,int abort_after=1){
        ggml_backend_cpu_set_threadpool(cpu,pool);ggml_backend_cpu_set_n_threads(cpu,requested?requested:threads);
        capture.threads=threads;capture.native=mode(pool);capture.seen.fill(0);
        Abort state;state.stop_after=abort_after;ggml_backend_cpu_set_abort_callback(cpu,abort?abort_now:nullptr,abort?&state:nullptr);
        auto result=ggml_backend_graph_compute(cpu,graph);++graphs;
        if(abort){assert(result==GGML_STATUS_ABORTED&&state.calls==state.stop_after);++aborts;
            if(abort_after==ggml_graph_n_nodes(graph))assert(!std::memcmp(y->data,expected.data(),expected.size()*4));
        }
        else{assert(result==GGML_STATUS_SUCCESS);assert(std::memcmp(y->data,expected.data(),expected.size()*4)==0);compared+=expected.size();
            for(int i=0;i<threads;++i){assert(capture.seen[i]==1);++affinity_checks;}
        }
        ggml_backend_cpu_set_abort_callback(cpu,nullptr,nullptr);
        assert(ggml_backend_graph_compute(cpu,empty)==GGML_STATUS_SUCCESS);
    };
    // Start with generation despite batch being created unpaused; then alternate
    // through backend_set_threadpool, without an explicit resume before compute.
    for(int round=0;round<8;++round){
        for(auto pool:{generation,batch,generation})for(int threads:{1,6,12,24}){
            if(pool==generation&&threads>12)continue;
            run(pool,threads,false);run(pool,threads,true);run(pool,threads,false);
            ggml_threadpool_pause(pool);run(pool,threads,false);
        }
    }
    run(generation,12,false,24); // Explicitly exercise the capacity clamp.
    for(auto pool:{generation,batch}){
        run(pool,pool==generation?12:24,true,0,ggml_graph_n_nodes(graph));
        run(pool,pool==generation?12:24,false);
    }
    size_t wrap_graphs=0;
    if(hybrid){
        ggml_backend_cpu_set_threadpool(cpu,batch);auto plan=ggml_graph_plan(empty,1,batch);
        assert(plan.n_threads==1&&plan.work_size==0);
        for(int i=0;i<70000;++i){assert(ggml_graph_compute(empty,&plan)==GGML_STATUS_SUCCESS);++wrap_graphs;}
        run(batch,24,false);run(generation,1,false);run(batch,1,false);run(generation,12,false);
    }
    // A pool omitted by the caller must remain disposable OpenMP, even at24.
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_backend_cpu_set_n_threads(cpu,24);
    capture.threads=24;capture.native=false;capture.check_affinity=false;capture.require_omp=true;capture.seen.fill(0);
    assert(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS);assert(!std::memcmp(y->data,expected.data(),expected.size()*4));
    for(int n:capture.seen)assert(n==1);
    ggml_backend_free(cpu);ggml_threadpool_free(generation);ggml_threadpool_free(batch);ggml_free(ctx);
    std::printf("{\"hybrid\":%s,\"graphs\":%d,\"comparisons\":%zu,\"abort_recoveries\":%d,\"worker_affinity_checks\":%d,\"generation_wrap_graphs\":%zu,\"passed\":true}\n",hybrid?"true":"false",graphs,compared,aborts,affinity_checks,wrap_graphs);
}
