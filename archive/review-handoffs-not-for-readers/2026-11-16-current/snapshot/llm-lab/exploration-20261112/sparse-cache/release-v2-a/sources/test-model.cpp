#include "llama.h"
#include "ggml-backend.h"
#include "iq3-cache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool ok,const char * message) { if (!ok) throw std::runtime_error(message); }
int main(int argc,char ** argv) try {
    need(argc==5,"fixture full-plan sparse-plan zero-plan");
    ggml_backend_load_all_from_path("/home/codex/projects/llm-lab/deepseek41-bringup-20261101/build/bin");
    llama_backend_init();
    const auto reg=ggml_backend_reg_by_name("CPU");need(reg,"CPU registry");
    need(ggml_backend_reg_get_proc_address(reg,"xeon_iq3_cache_add_selected_v2")==reinterpret_cast<void *>(xeon_iq3_cache_add_selected_v2),"v2 backend provider");
    unsetenv("GGML_CUDA_REGISTER_HOST"); unsetenv("XEON_MOE_CENSUS_PATH");
    ggml_backend_dev_t devices[]={nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=0;
    mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;mp.use_extra_bufts=false;
    std::vector<float> reference;
    for (int pass=0;pass<5;++pass) {
        setenv("GGML_CPU_IQ3_COMPACT_MIB",pass==0?"0":"1",1);
        if(pass<2)unsetenv("GGML_CPU_IQ_PLAN");else setenv("GGML_CPU_IQ_PLAN",argv[pass],1);
        need(xeon_iq3_cache_total_v1()==0,"cache before model load");
        auto model=llama_model_load_from_file(argv[1],mp);need(model,"model load");
        const size_t expected=pass==0||pass==4?0:pass==3?368640:884736;
        need(xeon_iq3_cache_total_v1()==expected,"model plan bytes");
        auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=8;cp.n_ubatch=8;cp.n_threads=12;cp.n_threads_batch=12;
        auto ctx=llama_init_from_model(model,cp);need(ctx,"context");
        auto batch=llama_batch_init(1,512,1);batch.n_tokens=1;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=true;
        std::vector<float> output;
        for(int step=0;step<5;++step) {
            for(int i=0;i<512;++i)batch.embd[i]=float((i*17+step*23)%101-50)/100.f;
            batch.pos[0]=step;need(llama_decode(ctx,batch)==0,"decode");
            auto logits=llama_get_logits_ith(ctx,0);need(logits,"logits");
            for(int i=0;i<32;++i)need(std::isfinite(logits[i]),"nonfinite logits");
            output.insert(output.end(),logits,logits+32);
        }
        if(!pass)reference=output;
        else need(output.size()==reference.size()&&!std::memcmp(output.data(),reference.data(),output.size()*4),"model plan logits bit mismatch");
        llama_batch_free(batch);llama_free(ctx);llama_model_free(model);
        need(xeon_iq3_cache_total_v1()==0,"model cache leak");
    }
    setenv("GGML_CPU_IQ_PLAN",argv[2],1);setenv("GGML_CPU_IQ3_COMPACT_MIB","0",1);
    auto failed=llama_model_load_from_file(argv[1],mp);need(!failed,"plan accepted without budget");
    need(xeon_iq3_cache_total_v1()==0,"failed load cache leak");
    unsetenv("GGML_CPU_IQ_PLAN");llama_backend_free();
    std::puts("{\"passes\":5,\"logits_per_pass\":160,\"bit_mismatches\":0,\"budget_guard\":true,\"cache_leaks\":0,\"pass\":true}");
}catch(const std::exception & error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
