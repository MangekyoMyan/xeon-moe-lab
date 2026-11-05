#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

int main(int argc, char ** argv) try {
    const int tokens=argc>1 ? std::stoi(argv[1]) : 512;
    const bool iq2=argc>2;
    if (tokens<1 || tokens>1024) return 1;
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);
    if (!cpu || !gpu) throw std::runtime_error("backend initialization failed");
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto dummy=ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),0);
    if (!ctx || !dummy) throw std::runtime_error("metadata allocation failed");
    ggml_backend_buffer_set_usage(dummy,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto weight=[&](const char * name,int k,int m,int experts) {
        auto t=ggml_new_tensor_3d(ctx,iq2?GGML_TYPE_IQ2_S:GGML_TYPE_MXFP4,k,m,experts);
        ggml_set_name(t,name);t->buffer=dummy;
        // Metadata-only reservation must never execute or dereference this pointer.
        t->data=reinterpret_cast<void *>(uintptr_t(0x1000));return t;
    };
    auto upw=weight("up_exps",5120,2304,384);
    auto gatew=weight("gate_exps",5120,2304,384);
    auto downw=weight("down_exps",2304,5120,384);
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,5120,1,tokens);
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,tokens);
    ggml_set_input(x);ggml_set_input(ids);
    auto up=ggml_mul_mat_id(ctx,upw,x,ids);
    auto gate=ggml_mul_mat_id(ctx,gatew,x,ids);
    auto activation=ggml_mul(ctx,ggml_silu(ctx,gate),up);
    auto down=ggml_mul_mat_id(ctx,downw,activation,ids);
    auto table=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,128);
    table->buffer=dummy;table->data=reinterpret_cast<void *>(uintptr_t(0x1000));
    auto lookup_ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tokens);
    ggml_set_input(lookup_ids);
    auto lookup=ggml_get_rows(ctx,table,lookup_ids);
    auto graph=ggml_new_graph_custom(ctx,128,false);
    ggml_build_forward_expand(graph,down);ggml_build_forward_expand(graph,lookup);
    ggml_backend_t backends[]={gpu,cpu};
    auto sched=ggml_backend_sched_new(backends,nullptr,2,128,false,true);
    ggml_backend_sched_set_tensor_backend(sched,x,gpu);
    ggml_backend_sched_set_tensor_backend(sched,ids,gpu);
    size_t sizes[2]={0,0};
    ggml_backend_sched_reserve_size(sched,graph,sizes);
    const bool up_gpu=ggml_backend_sched_get_tensor_backend(sched,up)==gpu;
    const bool down_gpu=ggml_backend_sched_get_tensor_backend(sched,down)==gpu;
    const bool lookup_cpu=ggml_backend_sched_get_tensor_backend(sched,lookup)==cpu;
    std::printf("{\"type\":\"%s\",\"tokens\":%d,\"GPU_scheduler_bytes\":%zu,\"CPU_scheduler_bytes\":%zu,\"up_GPU\":%s,\"down_GPU\":%s,\"GET_ROWS_CPU\":%s,\"computed\":false}\n",
        iq2?"IQ2_S":"MXFP4",tokens,sizes[0],sizes[1],up_gpu?"true":"false",down_gpu?"true":"false",lookup_cpu?"true":"false");
    ggml_backend_sched_free(sched);ggml_backend_buffer_free(dummy);ggml_free(ctx);
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    return lookup_cpu && up_gpu==(tokens>=(iq2?512:1024)) && down_gpu==up_gpu ? 0:2;
} catch (const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
