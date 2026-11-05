#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

struct test_graph {
    ggml_context * ctx;
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * output;
    ggml_cgraph * graph;
    test_graph(ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int tokens) {
        ctx=ggml_init({4*1024*1024,nullptr,true});
        x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,1,tokens);
        ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,tokens);
        ggml_set_input(x);ggml_set_input(ids);
        auto g=ggml_mul_mat_id(ctx,gate,x,ids);
        auto u=ggml_mul_mat_id(ctx,up,x,ids);
        output=ggml_mul_mat_id(ctx,down,ggml_mul(ctx,ggml_silu(ctx,g),u),ids);
        graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,output);
    }
    ~test_graph() { ggml_free(ctx); }
};

int main() try {
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);
    if (!cpu||!gpu) throw std::runtime_error("backend init");
    ggml_backend_cpu_set_n_threads(cpu,12);
    for (auto type:{GGML_TYPE_IQ2_S,GGML_TYPE_MXFP4}) {
        auto ctx=ggml_init({1024*1024,nullptr,true});
        auto up=ggml_new_tensor_3d(ctx,type,512,256,12);
        auto gate=ggml_new_tensor_3d(ctx,type,512,256,12);
        auto down=ggml_new_tensor_3d(ctx,type,256,512,12);
        auto weights=ggml_backend_alloc_ctx_tensors(ctx,cpu);
        if (!weights) throw std::runtime_error("weights allocation");
        ggml_backend_buffer_set_usage(weights,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(123);std::uniform_real_distribution<float> dist(-.02f,.02f);
        for(auto w:{gate,up,down}) {
            std::vector<float> f(ggml_nelements(w)),im(w->ne[0],1.f);
            for(auto &v:f)v=dist(rng);
            std::vector<uint8_t> q(ggml_nbytes(w));
            if (ggml_quantize_chunk(type,f.data(),q.data(),0,w->ne[1]*w->ne[2],w->ne[0],im.data())!=q.size()) throw std::runtime_error("quantize");
            ggml_backend_tensor_set(w,q.data(),0,q.size());
        }
        for (float input_scale:{.02f,.5f}) for (int tokens:{128,512,1024}) {
            test_graph ref(gate,up,down,tokens),candidate(gate,up,down,tokens);
            auto rb=ggml_backend_alloc_ctx_tensors(ref.ctx,cpu);
            ggml_backend_t backends[]={gpu,cpu};
            auto sched=ggml_backend_sched_new(backends,nullptr,2,128,false,true);
            ggml_backend_sched_set_tensor_backend(sched,candidate.x,gpu);
            ggml_backend_sched_set_tensor_backend(sched,candidate.ids,gpu);
            if (!ggml_backend_sched_alloc_graph(sched,candidate.graph)) throw std::runtime_error("graph allocate");
            std::uniform_real_distribution<float> input_dist(-input_scale,input_scale);
            std::vector<float> x(512*tokens);for(auto &v:x)v=input_dist(rng);
            std::vector<int32_t> ids(6*tokens);for(size_t i=0;i<ids.size();++i)ids[i]=i%12;
            for (auto g:{&ref,&candidate}) {
                ggml_backend_tensor_set(g->x,x.data(),0,x.size()*sizeof(float));
                ggml_backend_tensor_set(g->ids,ids.data(),0,ids.size()*sizeof(int32_t));
            }
            if (ggml_backend_graph_compute(cpu,ref.graph)!=GGML_STATUS_SUCCESS ||
                ggml_backend_sched_graph_compute(sched,candidate.graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute");
            ggml_backend_sched_synchronize(sched);ggml_backend_synchronize(cpu);
            const bool on_gpu=ggml_backend_sched_get_tensor_backend(sched,candidate.output)==gpu;
            const bool expected=tokens >= (type==GGML_TYPE_MXFP4 ? 1024:512);
            if(on_gpu!=expected)throw std::runtime_error("wrong backend selection");
            std::vector<float> a(ggml_nelements(ref.output)),b(a.size());
            ggml_backend_tensor_get(ref.output,a.data(),0,a.size()*sizeof(float));
            ggml_backend_tensor_get(candidate.output,b.data(),0,b.size()*sizeof(float));
            double err=0,norm=0;
            for(size_t i=0;i<a.size();++i){
                if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite");
                const double d=double(a[i])-b[i];err+=d*d;norm+=double(a[i])*a[i];
            }
            const double nmse=err/std::max(norm,1e-30);
            // Compare a full nonlinear expert block against independent FP64 dots.
            // Very small activations can expose the stock CPU Q8_0 FP16 scale floor.
            const auto dequant=ggml_get_type_traits(type)->to_float;
            std::vector<float> wr(512);
            double refnorm=0,cpu_error=0,gpu_error=0;
            for(int t:{0,tokens/2,tokens-1}) for(int j=0;j<6;++j) {
                const int e=ids[t*6+j];std::vector<double> hidden(256);
                for(int h=0;h<256;++h) {
                    double gv=0,uv=0;
                    dequant(static_cast<uint8_t *>(gate->data)+e*gate->nb[2]+h*gate->nb[1],wr.data(),512);
                    for(int k=0;k<512;++k)gv+=double(wr[k])*x[t*512+k];
                    dequant(static_cast<uint8_t *>(up->data)+e*up->nb[2]+h*up->nb[1],wr.data(),512);
                    for(int k=0;k<512;++k)uv+=double(wr[k])*x[t*512+k];
                    hidden[h]=gv/(1.0+std::exp(-gv))*uv;
                }
                for(int sample=0;sample<8;++sample) {
                    const int row=511*sample/7;double exact=0;
                    dequant(static_cast<uint8_t *>(down->data)+e*down->nb[2]+row*down->nb[1],wr.data(),256);
                    for(int h=0;h<256;++h)exact+=double(wr[h])*hidden[h];
                    const size_t index=(t*6+j)*512+row;
                    refnorm+=exact*exact;cpu_error+=(a[index]-exact)*(a[index]-exact);gpu_error+=(b[index]-exact)*(b[index]-exact);
                }
            }
            const double cpu_ref=cpu_error/std::max(refnorm,1e-30),gpu_ref=gpu_error/std::max(refnorm,1e-30);
            std::printf("{\"type\":\"%s\",\"input_scale\":%.3g,\"tokens\":%d,\"GPU\":%s,\"cpu_gpu_nmse\":%.12g,\"cpu_ref_nmse\":%.12g,\"candidate_ref_nmse\":%.12g}\n",ggml_type_name(type),input_scale,tokens,on_gpu?"true":"false",nmse,cpu_ref,gpu_ref);
            if(on_gpu ? gpu_ref>=5e-4 : err!=0)throw std::runtime_error("independent reference or CPU-path mismatch");
            ggml_backend_sched_free(sched);ggml_backend_buffer_free(rb);
        }
        ggml_backend_buffer_free(weights);ggml_free(ctx);
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    return 0;
} catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
