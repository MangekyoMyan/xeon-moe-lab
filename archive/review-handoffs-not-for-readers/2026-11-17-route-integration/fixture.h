#pragma once
#include "route_client.h"
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
struct Fixture {
    ggml_backend_t cpu=ggml_backend_cpu_init();ggml_context *ctx=nullptr;ggml_cgraph *graph=nullptr;
    ggml_backend_buffer_t buffer=nullptr;ggml_threadpool_t pool=nullptr;
    std::array<const ggml_tensor*,37> gates{};std::array<ggml_tensor*,37> ids{},out{};ggml_tensor *input=nullptr;
    Fixture(){
        need(cpu,"cpu");ctx=ggml_init({4*1024*1024,nullptr,true});need(ctx,"ctx");
        graph=ggml_new_graph_custom(ctx,256,false);input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,1,1);
        const ggml_type types[]={GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4};
        for(int i=0;i<37;++i){auto *w=ggml_new_tensor_3d(ctx,types[i%4],256,4,384);gates[i]=w;
            ids[i]=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,1);out[i]=ggml_mul_mat_id(ctx,w,input,ids[i]);ggml_build_forward_expand(graph,out[i]);}
        buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(20261117);
        for(auto *w:gates){std::vector<uint8_t>v(ggml_nbytes(w));for(auto &b:v)b=uint8_t(rng());
            for(size_t j=0;j<v.size();j+=ggml_type_size(w->type)){if(w->type==GGML_TYPE_MXFP4)v[j]=127;else{uint16_t scale=0x2000;std::memcpy(v.data()+j,&scale,2);}}
            ggml_backend_tensor_set(const_cast<ggml_tensor*>(w),v.data(),0,v.size());}
        auto p=ggml_threadpool_params_default(12);p.poll=0;p.strict_cpu=true;for(int i=0;i<12;++i)p.cpumask[i]=true;
        pool=ggml_threadpool_new(&p);need(pool,"pool");ggml_backend_cpu_set_threadpool(cpu,pool);ggml_backend_cpu_set_n_threads(cpu,12);
    }
    ~Fixture(){ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(cpu);}
    void inputs(int cycle){
        float x[256];for(int i=0;i<256;++i)x[i]=float((i*17+cycle*19)%1000-500)/1024;
        ggml_backend_tensor_set(input,x,0,sizeof(x));
        for(int layer=0;layer<37;++layer){int32_t r[6];for(int j=0;j<6;++j)r[j]=(layer*7+cycle*11+j*13)%384;ggml_backend_tensor_set(ids[layer],r,0,sizeof(r));}
    }
    std::vector<float> run(){
        int status=-999;std::thread worker([&]{status=ggml_backend_graph_compute(cpu,graph);});worker.join();need(status==GGML_STATUS_SUCCESS,"graph");
        std::vector<float>v(37*24);for(int i=0;i<37;++i)ggml_backend_tensor_get(out[i],v.data()+i*24,0,24*sizeof(float));
        for(float x:v)need(std::isfinite(x),"nonfinite fixture output");
        return v;
    }
};
