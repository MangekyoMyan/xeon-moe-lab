#include "llama.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv)try {
 if(argc!=2)return 1;llama_backend_init();auto gpu=ggml_backend_cuda_init(0);auto cpu=ggml_backend_cpu_init();if(!gpu||!cpu)throw std::runtime_error("backend");
 ggml_backend_dev_t devices[]={ggml_backend_get_device(gpu),nullptr};
 llama_model_tensor_buft_override override[]={
  {"blk\\.([0-9]|[12][0-9]|3[0-6])\\.ffn_(gate_up|gate|up|down)_exps\\.weight",ggml_backend_cpu_buffer_type()},
  {"per_layer_token_embd.weight",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
 auto p=llama_model_default_params();p.devices=devices;p.n_gpu_layers=99;p.no_alloc=true;p.load_mode=LLAMA_LOAD_MODE_NONE;p.lazy_mode=LLAMA_LAZY_MODE_OFF;p.use_extra_bufts=false;p.tensor_buft_overrides=override;
 auto model=llama_model_load_from_file(argv[1],p);if(!model)throw std::runtime_error("metadata-only model");
 std::vector<ggml_backend_t>backends{gpu,cpu};std::vector<ggml_backend_buffer_type_t>bufts{ggml_backend_get_default_buffer_type(gpu),ggml_backend_get_default_buffer_type(cpu)};
 for(size_t mib:{320,640,1280,2560,2880,3200})try {
  llama_moe_cache cache(*model,true,backends,bufts,mib*1024*1024);
  size_t gpu_bytes=0,host_bytes=0;for(auto &v:cache.memory_breakdown())(ggml_backend_buft_is_host(v.first)?host_bytes:gpu_bytes)+=v.second;
  std::map<const ggml_tensor*,std::vector<int>>groups;int enabled=0;
  for(int il=0;il<37;++il)if(cache.get_slot_map(il,1,6)){
   ++enabled;auto w=cache.get_experts(model->layers[il].ffn_up_exps);if(!w)throw std::runtime_error("up cache missing");groups[w->view_src?w->view_src:w].push_back(il);
  }
  std::printf("{\"budget_MiB\":%zu,\"no_alloc\":true,\"GPU_bytes\":%zu,\"host_bytes\":%zu,\"enabled_layers\":%d,\"groups\":[",mib,gpu_bytes,host_bytes,enabled);bool first=true;
  for(auto &g:groups){std::printf("%s{\"slots\":%lld,\"minimum_distinct_slots_per_token\":%zu,\"layers\":[",first?"":",",(long long)(g.first->ne[2]-1),g.second.size()*6);first=false;
   for(size_t i=0;i<g.second.size();++i)std::printf("%s%d",i?",":"",g.second[i]);std::printf("]}");}
  std::puts("]}");
 }catch(const std::exception &e){std::printf("{\"budget_MiB\":%zu,\"no_alloc\":true,\"enabled_layers\":0,\"error\":\"%s\"}\n",mib,e.what());}
 llama_model_free(model);ggml_backend_free(cpu);ggml_backend_free(gpu);llama_backend_free();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
