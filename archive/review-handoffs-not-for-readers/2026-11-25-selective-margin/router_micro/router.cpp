// Compare two specializations of the same installed experimental DSO on one CUDA stream.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "native_router_args.h"
#include "margin_shared.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct ggml_backend_cuda_context;
void ggml_cuda_op_topk_moe(ggml_backend_cuda_context&,const ggml_tensor*,ggml_tensor*,ggml_tensor*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,const ggml_cuda_topk_moe_args&);
extern "C" void xeon_margin_open(float);
extern "C" void xeon_margin_begin(int,int);
extern "C" void xeon_margin_receipt(margin_row*,int*);
extern "C" void xeon_margin_close();
namespace fs=std::filesystem;
static void need(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> static std::vector<T> file(const fs::path&p,size_t n){
    need(fs::file_size(p)>=n*sizeof(T),"file size");std::ifstream f(p,std::ios::binary);std::vector<T>v(n);
    f.read((char*)v.data(),n*sizeof(T));need(bool(f),"read");return v;
}
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv)try{
    need(argc==6,"capture folder, gap directory, bias F32, threshold, timing output");
    const fs::path cap=argv[1],gaps=argv[2];const float cut=std::stof(argv[4]);
    const int N=96,E=384,used=6;
    auto gpu=ggml_backend_cuda_init(0);need(gpu&&ggml_backend_is_cuda(gpu),"CUDA device");
    size_t room=0,total=0;ck(cudaMemGetInfo(&room,&total));need(room>size_t(1100)*1048576,"operator VRAM headroom");
    auto ctx=ggml_init({1024*1024,nullptr,true});
    auto logits=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,E,N);auto backing=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,E,N);
    auto ids=ggml_view_2d(ctx,backing,used,N,backing->nb[1],0);
    auto weights=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,used,N);
    auto bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,E);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,gpu);need(buf,"router buffers");
    auto b=file<float>(argv[3],E);ggml_backend_tensor_set(bias,b.data(),0,b.size()*4);
    // The DeepSeek41 graph adds epsilon to the weight sum. Its normalization
    // and scale follow the fused selection, outside this kernel boundary.
    ggml_cuda_topk_moe_args args{};args.sqrt_softplus=true;args.prob_bias=true;
    auto &native=*static_cast<ggml_backend_cuda_context*>(gpu->context);
    xeon_margin_open(cut);
    cudaEvent_t e0,e1;ck(cudaEventCreate(&e0));ck(cudaEventCreate(&e1));
    ggml_backend_event start{ggml_backend_get_device(gpu),e0},end{ggml_backend_get_device(gpu),e1};
    std::ofstream out(argv[5]);out<<"case\tpair\torder\tdecision\tcalls\tGPU_us_per_call\twall_us_per_call\n"<<std::setprecision(10);
    for(std::string name:{"JA96","PY96"}){
        auto d=cap/(name+"-early");auto x=file<float>(d/"ffn_moe_logits-0.bin",N*E);
        auto ref=file<int>(d/"ffn_moe_gate-0-ids.bin",N*used);
        auto fullref=file<margin_row>(gaps/("q_D_"+name+".gaps"),N);
        ggml_backend_tensor_set(logits,x.data(),0,x.size()*4);xeon_margin_begin(1,N);
        auto call=[&](){ggml_cuda_op_topk_moe(native,logits,weights,ids,nullptr,nullptr,bias,args);};
        std::vector<float> expected_w(N*used),got_w(N*used);std::vector<int> all(E*N);
        for(int arm=0;arm<2;arm++){
            ggml_set_name(ids,arm?"ffn_moe_topk-0":"ordinary");for(int i=0;i<100;i++)call();ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(backing,all.data(),0,all.size()*4);
            ggml_backend_tensor_get(weights,got_w.data(),0,got_w.size()*4);
            for(int t=0;t<N;t++)need(!std::memcmp(&all[t*E],&ref[t*used],used*4),"real top-six IDs");
            if(!arm)expected_w=got_w;else need(!std::memcmp(got_w.data(),expected_w.data(),got_w.size()*4),"normalized weights unchanged");
        }
        std::vector<margin_row> rows(40*128);int counts[6];xeon_margin_receipt(rows.data(),counts);
        need(!std::memcmp(rows.data(),fullref.data(),N*sizeof(margin_row)),"gaps equal to live full model");
        for(int pair=0;pair<24;pair++)for(int order=0;order<2;order++){
            const int arm=(pair+order)%2;
            ggml_set_name(ids,arm?"ffn_moe_topk-0":"ordinary");
            ggml_backend_synchronize(gpu);const int calls=128;
            auto t=ns();ggml_backend_event_record(&start,gpu);
            for(int i=0;i<calls;i++)call();
            ggml_backend_event_record(&end,gpu);ck(cudaEventSynchronize(e1));auto z=ns();
            float ms=0;ck(cudaEventElapsedTime(&ms,e0,e1));
            out<<name<<'\t'<<pair<<'\t'<<order<<'\t'<<arm<<'\t'<<calls<<'\t'<<ms*1000/calls<<'\t'<<(z-t)/1000./calls<<'\n';
        }
    }
    xeon_margin_close();ck(cudaEventDestroy(e0));ck(cudaEventDestroy(e1));ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(gpu);
    out.close();need(!out.fail(),"write");std::cout<<"two 96-token router captures reproduced; 96 timed blocks\n";
}catch(const std::exception&e){std::cerr<<"router micro: "<<e.what()<<'\n';return 1;}
