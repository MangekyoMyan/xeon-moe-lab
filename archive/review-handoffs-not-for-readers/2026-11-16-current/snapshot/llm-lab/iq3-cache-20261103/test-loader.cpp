#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "iq3-cache.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <dlfcn.h>
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(int argc,char**argv)try {
    need(argc==3,"fixture cpu|gpu");bool stream=std::strcmp(argv[2],"gpu-stream")==0;bool gpu=stream||std::strcmp(argv[2],"gpu")==0;
    // Match the server's backend discovery in its original binary directory.
    ggml_backend_load_all_from_path("/home/codex/projects/llm-lab/deepseek41-bringup-20261101/build/bin");
    auto reg=ggml_backend_reg_by_name("CPU");need(reg,"registered CPU");
    auto create=ggml_backend_reg_get_proc_address(reg,"xeon_iq3_cache_create_v1");
    need(create==reinterpret_cast<void *>(xeon_iq3_cache_create_v1),"server CPU overlay precedence");
    Dl_info info{};need(dladdr(create,&info),"CPU origin");
    std::printf("CPU cache provider: %s\n",info.dli_fname);
    llama_backend_init();auto cpu=ggml_backend_cpu_init();need(cpu,"cpu init");
    auto cuda=gpu?ggml_backend_cuda_init(0):nullptr;need(!gpu||cuda,"cuda init");
    ggml_backend_dev_t devices[]={gpu?ggml_backend_get_device(cuda):nullptr,nullptr};
    llama_model_tensor_buft_override overrides[]={
        {"blk\\.0\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    std::vector<float> baseline;
    for(int pass=0;pass<4;++pass) {
        setenv("GGML_CPU_IQ3_COMPACT_MIB",pass==0?"0":"1",1);
        if(gpu)setenv("GGML_CUDA_REGISTER_HOST","1",1);else unsetenv("GGML_CUDA_REGISTER_HOST");
        setenv("GGML_OP_OFFLOAD_MIN_BATCH","512",1);
        need(xeon_iq3_cache_total_v1()==0,"cache leaked before load");
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=gpu?99:0;
        mp.tensor_buft_overrides=overrides;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;mp.use_extra_bufts=false;
        auto model=llama_model_load_from_file(argv[1],mp);need(model,"model load");
        need(xeon_iq3_cache_total_v1()==(pass?884736:0),"loaded cache bytes");
        auto cp=llama_context_default_params();cp.n_ctx=stream?1024:128;cp.n_batch=stream?512:8;cp.n_ubatch=stream?512:8;cp.n_threads=12;cp.n_threads_batch=12;
        auto ctx=llama_init_from_model(model,cp);need(ctx,"context");
        std::vector<float> prefill_logits;
        if(stream) {
            auto prompt=llama_batch_init(512,512,1);prompt.n_tokens=512;
            for(int t=0;t<512;++t) {
                prompt.pos[t]=t;prompt.n_seq_id[t]=1;prompt.seq_id[t][0]=0;prompt.logits[t]=t==511;
                for(int i=0;i<512;++i)prompt.embd[t*512+i]=float((i*17+t*23)%101-50)/100.f;
            }
            need(llama_decode(ctx,prompt)==0,"stream prefill");auto v=llama_get_logits_ith(ctx,511);need(v,"prefill logits");
            prefill_logits.assign(v,v+32);llama_batch_free(prompt);
        }
        auto batch=llama_batch_init(1,512,1);batch.n_tokens=1;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=true;
        std::vector<float> got=prefill_logits;uint64_t before=xeon_iq3_cache_hits_v1();
        for(int step=0;step<3;++step) {
            for(int i=0;i<512;++i)batch.embd[i]=float((i*17+step*23)%101-50)/100.f;
            batch.pos[0]=step+(stream?512:0);need(llama_decode(ctx,batch)==0,"process");
            auto logits=llama_get_logits_ith(ctx,0);need(logits,"logits");got.insert(got.end(),logits,logits+32);
        }
        if(!pass)baseline=got;else need(got.size()==baseline.size()&&!std::memcmp(got.data(),baseline.data(),got.size()*4),"model logits bit mismatch");
        need((xeon_iq3_cache_hits_v1()>before)==bool(pass),"model CPU cache dispatch");
        llama_batch_free(batch);llama_free(ctx);llama_model_free(model);
        need(xeon_iq3_cache_total_v1()==0,"model cache not freed");
    }
    if(cuda)ggml_backend_free(cuda);ggml_backend_free(cpu);llama_backend_free();
    std::printf("{\"loader\":\"%s\",\"passes\":4,\"bit_mismatches\":0,\"cache_leaks\":0,\"pass\":true}\n",stream?"gpu-stream":gpu?"gpu":"cpu");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
