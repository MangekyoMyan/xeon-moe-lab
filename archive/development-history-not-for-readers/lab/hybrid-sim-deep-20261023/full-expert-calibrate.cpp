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
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

constexpr int H=5120,F=2304,E=64;
constexpr size_t MATRIX=size_t(H)*F*17/32;
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);assert(!sched_setaffinity(0,sizeof(s),&s));}

int main(int argc,char **argv){
    std::vector<int> thread_counts{1,12,24};
    if(argc==2){const int n=std::atoi(argv[1]);assert(n>1&&n<=24);thread_counts={1,n};}
    ggml_cpu_init();omp_set_dynamic(0);
    auto * memory=(char*)mmap(nullptr,3*MATRIX*E,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(memory!=MAP_FAILED);assert(!madvise(memory,3*MATRIX*E,MADV_NOHUGEPAGE));
    auto * wctx=ggml_init({1024*1024,nullptr,true});assert(wctx);
    ggml_tensor * weights[3];
    std::vector<unsigned char> patterns[2];
    std::vector<float> data(H);
    auto * trait=ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    for(int s=0;s<2;++s){
        int k=s?F:H;size_t stride=ggml_row_size(GGML_TYPE_MXFP4,k);patterns[s].resize(16*stride);
        for(int p=0;p<16;++p){
            for(int j=0;j<k;++j)data[j]=std::sin((j+p*73)*.371f)*std::ldexp(1.0f,p%8-9);
            trait->from_float(data.data(),patterns[s].data()+p*stride,k);
        }
    }
    for(int phase=0;phase<3;++phase){
        int shape=phase==2,k=shape?F:H,rows=shape?H:F;
        weights[phase]=ggml_new_tensor_3d(wctx,GGML_TYPE_MXFP4,k,rows,E);
        weights[phase]->data=memory+phase*MATRIX*E;
        const size_t stride=weights[phase]->nb[1];
        #pragma omp parallel num_threads(24)
        {
            pin(omp_get_thread_num());
            #pragma omp for schedule(static)
            for(int e=0;e<E;++e)for(int r=0;r<rows;++r)
                std::memcpy((char*)weights[phase]->data+e*MATRIX+r*stride,patterns[shape].data()+((r+e)%16)*stride,stride);
        }
    }
    for(int selected=1;selected<=6;++selected){
        auto * ctx=ggml_init({8*1024*1024,nullptr,false});assert(ctx);
        auto * x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,H,1,1);
        auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,selected,1);
        for(int i=0;i<H;++i)((float*)x->data)[i]=std::sin(i*.173f);
        auto * gate=ggml_mul_mat_id(ctx,weights[0],x,ids);
        auto * up=ggml_mul_mat_id(ctx,weights[1],x,ids);
        auto * act=ggml_swiglu_clamp(ctx,gate,up,10.0f);
        auto * down=ggml_mul_mat_id(ctx,weights[2],act,ids);
        auto * combined=ggml_sum_rows(ctx,ggml_cont(ctx,ggml_transpose(ctx,down)));
        auto * result=ggml_scale(ctx,combined,1.0f/selected);
        auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
        std::vector<unsigned char> reference;
        for(int threads : thread_counts){
            auto params=ggml_threadpool_params_default(threads);
            params.strict_cpu=true;
            if(threads==22){for(int i=0;i<24;++i)params.cpumask[i]=(i!=0&&i!=12);}
            else for(int i=0;i<threads;++i)params.cpumask[i]=true;
            auto * pool=ggml_threadpool_new(&params);assert(pool);
            auto plan=ggml_graph_plan(graph,threads,pool);
            std::vector<uint8_t> work(plan.work_size);plan.work_data=work.data();
            auto compute=[&](int i){
                for(int e=0;e<selected;++e)((int32_t*)ids->data)[e]=(i*7+e)%E;
                assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);
            };
            compute(0);
            for(int i=0;i<H;++i)assert(std::isfinite(((float*)result->data)[i]));
            if(threads==1){reference.resize(ggml_nbytes(result));std::memcpy(reference.data(),result->data,reference.size());}
            else assert(!std::memcmp(reference.data(),result->data,reference.size()));
            printf("{\"validation\":true,\"selected\":%d,\"threads\":%d,\"workspace_bytes\":%zu}\n",selected,threads,plan.work_size);fflush(stdout);
            if(threads>1){
                for(int round=0;round<5;++round){
                    for(int i=0;i<16;++i)compute(i);
                    double t=omp_get_wtime();for(int i=0;i<16;++i)compute(i);t=omp_get_wtime()-t;
                    int repeat=std::clamp(int(std::ceil(.3*16/t)),16,10000);
                    t=omp_get_wtime();for(int i=0;i<repeat;++i)compute(i);t=omp_get_wtime()-t;
                    printf("{\"selected\":%d,\"threads\":%d,\"round\":%d,\"repeats\":%d,\"ms_per_layer\":%.9f,\"logical_GB_s\":%.9f}\n",selected,threads,round,repeat,t*1000/repeat,3*MATRIX*double(selected)*repeat/t/1e9);fflush(stdout);
                }
            }
            ggml_threadpool_free(pool);
        }
        ggml_free(ctx);
    }
    ggml_free(wctx);munmap(memory,3*MATRIX*E);
}
