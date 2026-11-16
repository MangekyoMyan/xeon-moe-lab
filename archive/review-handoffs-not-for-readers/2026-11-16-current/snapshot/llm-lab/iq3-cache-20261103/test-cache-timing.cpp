#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "iq3-cache.h"
#include "ggml-cpu/quants.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static void need(bool b,const char *m){if(!b)throw std::runtime_error(m);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t cases=0,outputs=0;
static void run(ggml_backend_t cpu,int nth,int k,int m,int tokens,bool broadcast,bool strided,bool fp32,
                const std::vector<uint8_t> &real={}) {
    const int experts=real.empty()?7:48,used=6,slots=broadcast?1:used;
    auto ctx=ggml_init({8*1024*1024,nullptr,true});need(ctx,"ctx");
    auto w=ggml_new_tensor_3d(ctx,GGML_TYPE_IQ3_XXS,k,m,experts);
    auto xt=fp32?GGML_TYPE_F32:GGML_TYPE_Q8_K;
    const size_t stride=ggml_row_size(xt,k);
    auto backing=ggml_new_tensor_3d(ctx,xt,k,strided?slots*2:slots,strided?tokens*2+1:tokens);
    auto x=strided?ggml_view_3d(ctx,backing,k,slots,tokens,stride*2,stride*4*slots,stride):backing;
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);
    auto y=ggml_mul_mat_id(ctx,w,x,ids);
    auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buf,"buffer");
    ggml_backend_buffer_set_usage(buf,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(100+k+tokens+m);
    std::vector<uint8_t> weights(ggml_nbytes(w));
    if(real.empty()) {
        auto p=reinterpret_cast<block_iq3_xxs *>(weights.data());
        for(size_t b=0;b<weights.size()/sizeof(*p);++b){p[b].d=0x1800;for(auto &q:p[b].qs)q=rng();}
    }else {need(real.size()==weights.size(),"real weight size");weights=real;}
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());
    std::vector<uint8_t> inputs(ggml_nbytes(backing));std::vector<float> f(k);
    for(size_t off=0;off<inputs.size();off+=stride){
        for(auto &v:f)v=float(int(rng()%2001)-1000)/2048;
        if(fp32)std::memcpy(inputs.data()+off,f.data(),stride);
        else quantize_row_q8_K(f.data(),inputs.data()+off,k);
    }
    ggml_backend_tensor_set(backing,inputs.data(),0,inputs.size());
    std::vector<int32_t> routes(used*tokens);
    for(size_t i=0;i<routes.size();++i)routes[i]=int(i*3)%experts;
    if(real.empty()){routes[1]=routes[0];routes[used-1]=-1;}
    ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
    std::vector<float> base(ggml_nelements(y)),got(base.size());
    auto compute=[&](std::vector<float> &dst){
        need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"compute");
        ggml_backend_tensor_get(y,dst.data(),0,dst.size()*4);
    };
    compute(base);
    const size_t bytes=(weights.size()/sizeof(block_iq3_xxs)*144+4095)/4096*4096;
    void *small=xeon_iq3_cache_create_v1(bytes-1,0,1);need(small,"small owner");
    need(xeon_iq3_cache_add_v1(small,w)==0,"budget fallback");xeon_iq3_cache_free_v1(small);
    void *reserve=xeon_iq3_cache_create_v1(bytes,std::numeric_limits<size_t>::max(),1);
    need(xeon_iq3_cache_add_v1(reserve,w)==-1,"reserve fallback");xeon_iq3_cache_free_v1(reserve);
    void *owner=xeon_iq3_cache_create_v1(bytes,0,12);need(owner,"owner");
    need(xeon_iq3_cache_add_v1(owner,w)==1,"add");need(xeon_iq3_cache_used_v1(owner)==bytes,"accounting");
    need(xeon_iq3_cache_lookup(w),"lookup");
    ggml_tensor wrong=*w;wrong.nb[2]+=16;need(!xeon_iq3_cache_lookup(&wrong),"shape identity");
    void *other=xeon_iq3_cache_create_v1(bytes,0,1);need(other,"other owner");
    need(xeon_iq3_cache_add_v1(other,w)==0,"duplicate model pointer");xeon_iq3_cache_free_v1(other);
    uint64_t hits=xeon_iq3_cache_hits_v1();compute(got);
    need(std::memcmp(base.data(),got.data(),base.size()*4)==0,"graph bit mismatch");
    need((xeon_iq3_cache_hits_v1()>hits)==(tokens==1),"single token dispatch");
    std::vector<double> a,b;
    if(!real.empty()) {
        // Toggle the CPU reference dispatch, which uses the original single-row IQ3 dot.
        for(int repeat=0;repeat<9;++repeat) {
            for(int j=0;j<used;++j)routes[j]=(repeat*used+j)%experts;
            ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
            ggml_backend_cpu_set_use_ref(cpu,true);compute(base);
            for(bool original:{true,false,false,true}) {
            ggml_backend_cpu_set_use_ref(cpu,original);
            double start=now();compute(got);double dt=now()-start;
            need(std::memcmp(base.data(),got.data(),base.size()*4)==0,"timed graph bit mismatch");
            (original?a:b).push_back(dt);
        }
        }
        ggml_backend_cpu_set_use_ref(cpu,false);
        std::printf("{\"kind\":\"real_graph\",\"threads\":%d,\"k\":%d,\"m\":%d,\"topk\":6,\"resident_experts\":48,\"original_ms\":%.6f,\"compact_ms\":%.6f,\"speedup\":%.4f}\n",nth,k,m,med(a),med(b),med(a)/med(b));
    }
    xeon_iq3_cache_free_v1(owner);need(!xeon_iq3_cache_lookup(w),"stale lookup after free");
    compute(got);need(std::memcmp(base.data(),got.data(),base.size()*4)==0,"free fallback");
    std::vector<uint8_t> unchanged(weights.size());ggml_backend_tensor_get(w,unchanged.data(),0,unchanged.size());
    need(unchanged==weights,"original weights changed");
    // Reuse the exact tensor/data address with changed contents after owner destruction.
    reinterpret_cast<block_iq3_xxs *>(weights.data())->d=0x1c00;
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());compute(base);
    owner=xeon_iq3_cache_create_v1(bytes,0,1);need(xeon_iq3_cache_add_v1(owner,w)==1,"reload add");
    compute(got);need(std::memcmp(base.data(),got.data(),base.size()*4)==0,"reload mismatch");
    xeon_iq3_cache_free_v1(owner);
    outputs+=base.size();++cases;
    ggml_backend_buffer_free(buf);ggml_free(ctx);
}
int main(int argc,char **argv)try {
    auto cpu=ggml_backend_cpu_init();need(cpu,"CPU");
    need(!xeon_iq3_cache_create_v1((size_t(64)<<30)+1,0,1),"cap check");
    for(int nth: {1,12}) {
        ggml_backend_cpu_set_n_threads(cpu,nth);auto pp=ggml_threadpool_params_default(nth);pp.poll=0;pp.strict_cpu=true;
        for(int i=0;i<nth;++i)pp.cpumask[i]=true;
        auto pool=ggml_threadpool_new(&pp);ggml_backend_cpu_set_threadpool(cpu,pool);
        if(argc==1) {
            for(int k:{256,512,2304,5120})for(int m:{17,129})for(int tokens:{1,3})
                for(bool broadcast:{true,false})for(bool strided:{true,false})for(bool fp32:{true,false})
                    run(cpu,nth,k,m,tokens,broadcast,strided,fp32);
        } else {
            need(argc==3,"usage: [shard tensor]");
            auto g=gguf_init_from_file(argv[1],{true,nullptr});need(g,"GGUF");auto ti=gguf_find_tensor(g,argv[2]);
            need(ti>=0 && gguf_get_tensor_type(g,ti)==GGML_TYPE_IQ3_XXS,"IQ3 tensor");auto ne=gguf_get_tensor_ne(g,ti);
            need(ne[2]>=48 && ne[3]==1,"experts");int k=ne[0],m=ne[1];
            std::vector<uint8_t> weights(ggml_row_size(GGML_TYPE_IQ3_XXS,k)*m*48);
            std::ifstream f(argv[1],std::ios::binary);f.seekg(gguf_get_data_offset(g)+gguf_get_tensor_offset(g,ti));
            need(bool(f.read(reinterpret_cast<char *>(weights.data()),weights.size())),"read weights");gguf_free(g);
            run(cpu,nth,k,m,1,std::string(argv[2]).find("down")==std::string::npos,false,true,weights);
        }
        ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);
    }
    ggml_backend_free(cpu);
    std::printf("{\"cases\":%zu,\"outputs\":%zu,\"bit_mismatches\":0,\"pass\":true}\n",cases,outputs);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
