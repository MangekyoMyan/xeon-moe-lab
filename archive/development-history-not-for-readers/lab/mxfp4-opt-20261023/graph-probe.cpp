#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

constexpr int H=5120,F=2304;
constexpr size_t MATRIX=size_t(H)*F*17/32;
static double now(){timespec t{};assert(clock_gettime(CLOCK_MONOTONIC_RAW,&t)==0);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);assert(!sched_setaffinity(0,sizeof(s),&s));}
static std::vector<int> cpus(int n,const std::string &scope){
    std::vector<int> all,out;
    for(int c=0;c<24;++c){
        if(scope=="node0" && c%12>=6)continue;
        if(scope=="node1" && c%12<6)continue;
        if(scope=="both" && n==22 && (c==0||c==12))continue;
        if(scope=="both" && n==20 && (c==0||c==6||c==12||c==18))continue;
        all.push_back(c);
    }
    assert(int(all.size())>=n);out.assign(all.begin(),all.begin()+n);return out;
}
static void placement(char *memory,size_t bytes){
    const int n=1024;std::vector<void*> pages(n);std::vector<int> status(n,-1000);
    size_t total=bytes/4096;
    for(int i=0;i<n;++i)pages[i]=memory+((size_t(i)*7919)%total)*4096;
    const long rc=syscall(SYS_move_pages,0,n,pages.data(),nullptr,status.data(),0);
    int n0=0,n1=0,other=0;
    if(rc==0)for(int x:status){if(x==0)++n0;else if(x==1)++n1;else ++other;}
    printf("{\"placement_query_rc\":%ld,\"sampled_pages\":%d,\"node0\":%d,\"node1\":%d,\"other\":%d}\n",rc,n,n0,n1,other);fflush(stdout);
}
int main(int argc,char **argv){
    if(argc<6||argc>8)return 2;
    const int threads=std::atoi(argv[1]),selected=std::atoi(argv[2]),experts=std::atoi(argv[3]);
    const std::string affinity=argv[4],scope=argc>=7?argv[6]:"both";
    const double seconds=std::atof(argv[5]);
    assert(threads>=1&&threads<=24&&selected>=1&&selected<=6&&experts>=selected&&experts<=128);
    assert(seconds>=.05&&seconds<=120);assert(affinity=="each"||affinity=="once");
    assert(scope=="both"||scope=="node0"||scope=="node1");
    const auto chosen=cpus(threads,scope);
    ggml_cpu_init();omp_set_dynamic(0);
    const size_t bytes=3*MATRIX*experts;
    auto *memory=(char*)mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(memory!=MAP_FAILED);assert(!madvise(memory,bytes,MADV_NOHUGEPAGE));
    auto *wctx=ggml_init({1024*1024,nullptr,true});assert(wctx);ggml_tensor *weights[3];
    std::vector<unsigned char> patterns[2];std::vector<float> data(H);
    auto *trait=ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    for(int s=0;s<2;++s){
        int k=s?F:H;size_t stride=ggml_row_size(GGML_TYPE_MXFP4,k);patterns[s].resize(16*stride);
        for(int p=0;p<16;++p){
            for(int j=0;j<k;++j)data[j]=std::sin((j+p*73)*.371f)*std::ldexp(1.0f,p%8-9);
            trait->from_float(data.data(),patterns[s].data()+p*stride,k);
        }
    }
    for(int phase=0;phase<3;++phase){
        int shape=phase==2,k=shape?F:H,rows=shape?H:F;
        weights[phase]=ggml_new_tensor_3d(wctx,GGML_TYPE_MXFP4,k,rows,experts);
        weights[phase]->data=memory+phase*MATRIX*experts;
        const size_t stride=weights[phase]->nb[1];
        #pragma omp parallel num_threads(24)
        {
            pin(omp_get_thread_num());
            #pragma omp for schedule(static)
            for(int e=0;e<experts;++e)for(int r=0;r<rows;++r)
                std::memcpy((char*)weights[phase]->data+e*MATRIX+r*stride,patterns[shape].data()+((r+e)%16)*stride,stride);
        }
    }
    placement(memory,bytes);
    auto *ctx=ggml_init({8*1024*1024,nullptr,false});assert(ctx);
    auto *x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,H,1,1);
    auto *ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,selected,1);
    for(int i=0;i<H;++i)((float*)x->data)[i]=std::sin(i*.173f);
    auto *gate=ggml_mul_mat_id(ctx,weights[0],x,ids);
    auto *up=ggml_mul_mat_id(ctx,weights[1],x,ids);
    auto *act=ggml_swiglu_clamp(ctx,gate,up,10.0f);
    auto *down=ggml_mul_mat_id(ctx,weights[2],act,ids);
    auto *combined=ggml_sum_rows(ctx,ggml_cont(ctx,ggml_transpose(ctx,down)));
    auto *result=ggml_scale(ctx,combined,1.0f/selected);
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
    auto set_ids=[&](uint64_t step){for(int e=0;e<selected;++e)((int32_t*)ids->data)[e]=(step*7+e)%experts;};
    std::vector<std::vector<unsigned char>> references;
    auto refparams=ggml_threadpool_params_default(1);refparams.cpumask[chosen[0]]=true;
    auto *refpool=ggml_threadpool_new(&refparams);assert(refpool);
    auto refplan=ggml_graph_plan(graph,1,refpool);std::vector<uint8_t> refwork(refplan.work_size);refplan.work_data=refwork.data();
    for(int step=0;step<3;++step){
        set_ids(step);assert(ggml_graph_compute(graph,&refplan)==GGML_STATUS_SUCCESS);
        references.emplace_back(ggml_nbytes(result));std::memcpy(references.back().data(),result->data,references.back().size());
    }
    ggml_threadpool_free(refpool);
    auto params=ggml_threadpool_params_default(threads);params.strict_cpu=true;
    if(affinity=="each")for(int c:chosen)params.cpumask[c]=true;
    auto *pool=ggml_threadpool_new(&params);assert(pool);
    auto plan=ggml_graph_plan(graph,threads,pool);std::vector<uint8_t> work(plan.work_size);plan.work_data=work.data();
    // The once mode isolates the cost of reapplying affinity every graph call.
    // Both modes keep identical per-worker CPU affinity throughout the loop.
    #pragma omp parallel num_threads(threads)
    {pin(chosen[omp_get_thread_num()]);}
    auto compute=[&](uint64_t step){set_ids(step);assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);};
    for(int step=0;step<3;++step){
        compute(step);assert(!std::memcmp(references[step].data(),result->data,references[step].size()));
        for(int i=0;i<H;++i)assert(std::isfinite(((float*)result->data)[i]));
    }
    auto check_affinity=[&](){
        #pragma omp parallel num_threads(threads)
        {
            int i=omp_get_thread_num();cpu_set_t s;CPU_ZERO(&s);assert(!sched_getaffinity(0,sizeof(s),&s));
            assert(CPU_COUNT(&s)==1&&CPU_ISSET(chosen[i],&s));assert(sched_getcpu()==chosen[i]);
        }
    };
    check_affinity();
    if(argc==8){
        FILE *out=std::fopen(argv[7],"wb");assert(out);
        for(int step=0;step<16;++step){
            compute(step);
            for(auto *tensor : {gate,up,act,down,result}){
                const size_t bytes=ggml_nbytes(tensor);
                assert(std::fwrite(tensor->data,1,bytes,out)==bytes);
            }
        }
        assert(!std::fclose(out));
    }
    double warm=now();uint64_t warm_count=0;
    while(now()-warm<1.0)compute(warm_count++);
    printf("{\"ready\":true,\"threads\":%d,\"selected\":%d,\"experts\":%d,\"weight_pool_bytes\":%zu,\"logical_bytes_per_call\":%zu,\"affinity\":\"%s\",\"scope\":\"%s\",\"seconds\":%.3f,\"validation_bit_equal\":true,\"cpus\":[",
        threads,selected,experts,bytes,3*MATRIX*selected,affinity.c_str(),scope.c_str(),seconds);
    for(size_t i=0;i<chosen.size();++i)printf("%s%d",i?",":"",chosen[i]);
    printf("]}\n");fflush(stdout);

    const double start=now();uint64_t calls=0;
    do {for(int j=0;j<16;++j)compute(calls++);} while(now()-start<seconds);
    const double elapsed=now()-start;
    printf("{\"done\":true,\"calls\":%llu,\"seconds\":%.9f,\"ms_per_call\":%.9f,\"logical_GBs\":%.9f}\n",
        (unsigned long long)calls,elapsed,elapsed*1000/calls,3*MATRIX*selected*double(calls)/elapsed/1e9);fflush(stdout);
    // Keep the PID and workers alive until observers have disabled counters.
    check_affinity();printf("{\"post_affinity_valid\":true}\n");fflush(stdout);
    ggml_threadpool_free(pool);ggml_free(ctx);ggml_free(wctx);munmap(memory,bytes);
}
