#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>
#include <vector>

static bool abort_now(void * data) { ++*static_cast<int *>(data); return true; }
int main(int argc,char ** argv) {
    assert(argc==2);Dl_info info{};char actual[PATH_MAX],wanted[PATH_MAX];
    assert(dladdr(reinterpret_cast<void *>(ggml_graph_compute),&info));
    assert(realpath(info.dli_fname,actual)&&realpath(argv[1],wanted)&&std::strcmp(actual,wanted)==0);
    ggml_init_params init = {8*1024*1024, nullptr, false};
    ggml_context * ctx = ggml_init(init); assert(ctx);
    constexpr int k=64,m=512,n=16;
    ggml_tensor * weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
    ggml_tensor * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
    float * w=ggml_get_data_f32(weights),*x=ggml_get_data_f32(input);
    for(int i=0;i<k*m;++i)w[i]=float(i%7-3);
    for(int i=0;i<k*n;++i)x[i]=float(i%5-2);
    ggml_tensor * output=ggml_scale(ctx,ggml_mul_mat(ctx,weights,input),0.5f);
    ggml_cgraph * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
    ggml_cgraph * empty=ggml_new_graph(ctx);
    std::vector<float> expected(m*n);
    for(int j=0;j<n;++j)for(int i=0;i<m;++i){float sum=0;for(int z=0;z<k;++z)sum+=w[i*k+z]*x[j*k+z];expected[j*m+i]=sum*0.5f;}
    int counts[]={6,12,24};ggml_threadpool * pools[3];
    for(int a=0;a<3;++a){auto params=ggml_threadpool_params_default(counts[a]);params.poll=0;params.paused=true;params.strict_cpu=true;for(int c=0;c<counts[a];++c)params.cpumask[c]=true;pools[a]=ggml_threadpool_new(&params);assert(pools[a]);}
    size_t comparisons=0;int aborts=0,empty_graphs=0;
    auto compute=[&](ggml_cgraph * g,int threads,ggml_threadpool * pool,bool abort){
        ggml_cplan plan=ggml_graph_plan(g,threads,pool);
        std::vector<uint8_t> work(plan.work_size+64);
        plan.work_data=reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(work.data())+63)&~uintptr_t(63));
        int calls=0;if(abort){plan.abort_callback=abort_now;plan.abort_callback_data=&calls;}
        auto status=ggml_graph_compute(g,&plan);
        if(abort){assert(status==GGML_STATUS_ABORTED&&calls>0);++aborts;}
        else {assert(status==GGML_STATUS_SUCCESS);if(g==empty){++empty_graphs;return;}float * got=ggml_get_data_f32(output);for(size_t z=0;z<expected.size();++z){assert(got[z]==expected[z]);++comparisons;}}
    };
    for(int round=0;round<24;++round){
        for(int a=0;a<3;++a){
            for(int other=0;other<3;++other)if(other!=a)ggml_threadpool_pause(pools[other]);
            ggml_threadpool_resume(pools[a]);
            for(int threads : {1,counts[a]}){
                compute(empty,threads,pools[a],false);
                compute(graph,threads,pools[a],false);
                compute(graph,threads,pools[a],true);
                compute(graph,threads,pools[a],false);
            }
        }
    }
    for(int a=0;a<3;++a){ggml_threadpool_pause(pools[a]);ggml_threadpool_free(pools[a]);}
    ggml_free(ctx);
    std::printf("{\"comparisons\":%zu,\"aborted_graphs\":%d,\"empty_graphs\":%d,\"pool_sizes\":[6,12,24],\"pass\":true}\n",comparisons,aborts,empty_graphs);
}
