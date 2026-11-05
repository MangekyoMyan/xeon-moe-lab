#include "llama.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "llama-memory.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
static size_t bytes(const std::map<ggml_backend_buffer_type_t,size_t>&m){size_t n=0;for(auto &p:m)n+=p.second;return n;}
int main(int argc,char**argv)try{
 if(argc!=2)return 1;llama_backend_init();auto gpu=ggml_backend_cuda_init(0);auto cpu=ggml_backend_cpu_init();if(!gpu||!cpu)throw std::runtime_error("backend");
 ggml_backend_dev_t devices[]={ggml_backend_get_device(gpu),nullptr};auto p=llama_model_default_params();p.devices=devices;p.n_gpu_layers=99;p.no_alloc=true;p.load_mode=LLAMA_LOAD_MODE_NONE;p.lazy_mode=LLAMA_LAZY_MODE_OFF;p.use_extra_bufts=false;
 auto model=llama_model_load_from_file(argv[1],p);if(!model)throw std::runtime_error("metadata-only model");
 for(int context:{8192,32768,131072,262144,524288,1048576})for(bool real:{false,true}) {
  if(real && context!=131072)continue;
  llama_cparams c{};c.n_ctx=c.n_ctx_seq=context;c.n_seq_max=1;c.n_ubatch=1024;c.n_rs_seq=0;c.flash_attn=true;c.offload_kqv=true;c.kv_unified=false;c.ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT;
  llama_memory_params m{};m.type_k=m.type_v=GGML_TYPE_F16;m.swa_full=false;m.ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT;m.no_alloc=!real;
  std::unique_ptr<llama_memory_i> mem(model->create_memory(m,c));auto d=dynamic_cast<llama_kv_cache_dsv4*>(mem.get());if(!d)throw std::runtime_error("not DSV4 memory");
  std::printf("{\"context\":%d,\"real_KV_allocation\":%s,\"model_weights_allocated\":false,\"raw_bytes\":%zu,\"CSA_bytes\":%zu,\"HCA_bytes\":%zu,\"LID_bytes\":%zu,\"CSA_state_bytes\":%zu,\"HCA_state_bytes\":%zu,\"LID_state_bytes\":%zu,\"total_bytes\":%zu}\n",context,real?"true":"false",bytes(d->get_raw()->memory_breakdown()),bytes(d->get_csa()->memory_breakdown()),bytes(d->get_hca()->memory_breakdown()),bytes(d->get_lid()->memory_breakdown()),bytes(d->get_csa_state()->memory_breakdown()),bytes(d->get_hca_state()->memory_breakdown()),bytes(d->get_lid_state()->memory_breakdown()),bytes(d->memory_breakdown()));std::fflush(stdout);
 }
 llama_model_free(model);ggml_backend_free(gpu);ggml_backend_free(cpu);llama_backend_free();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
