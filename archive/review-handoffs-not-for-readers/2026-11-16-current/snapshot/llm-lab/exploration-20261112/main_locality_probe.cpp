#include "ggml.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>
#include <sched.h>
#include <vector>
struct capture { int calls=0; int expected=0; bool check=false; };
static bool observe(void * opaque) {
    auto * c=static_cast<capture *>(opaque);++c->calls;
    if(c->check){cpu_set_t mask;CPU_ZERO(&mask);assert(sched_getaffinity(0,sizeof(mask),&mask)==0);assert(CPU_COUNT(&mask)==1&&CPU_ISSET(c->expected,&mask));assert(sched_getcpu()==c->expected);}
    return false;
}
int main(int argc,char ** argv) {
    assert(argc==3);const bool last=std::atoi(argv[2])!=0;
    Dl_info info{};char loaded[PATH_MAX],wanted[PATH_MAX];assert(dladdr(reinterpret_cast<void *>(ggml_graph_compute),&info));assert(realpath(info.dli_fname,loaded)&&realpath(argv[1],wanted)&&std::strcmp(loaded,wanted)==0);
    ggml_init_params params={8*1024*1024,nullptr,false};auto * ctx=ggml_init(params);assert(ctx);
    constexpr int k=64,m=512,n=16;
    auto * w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
    auto * wdata=ggml_get_data_f32(w);auto * xdata=ggml_get_data_f32(x);
    for(int i=0;i<k*m;++i)wdata[i]=float(i%7-3);for(int i=0;i<k*n;++i)xdata[i]=float(i%5-2);
    auto * y=ggml_scale(ctx,ggml_mul_mat(ctx,w,x),0.5f);auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    size_t compared=0;int captures=0;
    for(int threads : {1,6,12,24}) {
        auto p=ggml_threadpool_params_default(threads);p.strict_cpu=true;p.poll=0;
        const int six[]={0,1,2,6,7,8};
        for(int i=0;i<threads;++i)p.cpumask[threads==6?six[i]:i]=true;
        auto * pool=ggml_threadpool_new(&p);assert(pool);
        for(int repeat=0;repeat<4;++repeat){
            auto plan=ggml_graph_plan(graph,threads,pool);assert(plan.n_threads==threads);
            std::vector<uint8_t> work(plan.work_size+64);plan.work_data=reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(work.data())+63)&~uintptr_t(63));
            capture c;c.expected=last?(threads==6?8:threads-1):0;c.check=threads>1;plan.abort_callback=observe;plan.abort_callback_data=&c;
            assert(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS);assert(c.calls>0);captures+=c.calls;
            float * got=ggml_get_data_f32(y);
            for(int j=0;j<n;++j)for(int i=0;i<m;++i){float expected=0;for(int z=0;z<k;++z)expected+=wdata[i*k+z]*xdata[j*k+z];assert(got[j*m+i]==expected*.5f);++compared;}
        }
        ggml_threadpool_free(pool);
    }
    ggml_free(ctx);std::printf("{\"main_last\":%s,\"comparisons\":%zu,\"affinity_callbacks\":%d,\"passed\":true}\n",last?"true":"false",compared,captures);
}
