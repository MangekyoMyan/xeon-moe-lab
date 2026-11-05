#include "gguf.h"
#include "ggml.h"
#include "cache-policy.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
struct item{int layer,projection;ggml_type type;std::string name;size_t bytes;};
int main(int argc,char**argv){
 std::vector<item>items;
 for(int a=1;a<argc;++a){auto g=gguf_init_from_file(argv[a],{true,nullptr});if(!g)return 1;
  for(int64_t i=0;i<gguf_get_n_tensors(g);++i){
   std::string name=gguf_get_tensor_name(g,i);int layer=-1;auto type=gguf_get_tensor_type(g,i);
   if(sscanf(name.c_str(),"blk.%d.",&layer)!=1||layer<0||layer>=37)continue;
   if(type!=GGML_TYPE_IQ2_S&&type!=GGML_TYPE_IQ3_S&&type!=GGML_TYPE_IQ3_XXS)continue;
   int projection=-1,j=0;for(const auto*s:{"ffn_gate_up_exps.weight","ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}){if(name=="blk."+std::to_string(layer)+"."+s)projection=j;++j;}if(projection<0)continue;
   auto ne=gguf_get_tensor_ne(g,i);size_t bytes=size_t(ne[0]/256)*ne[1]*ne[2]*ne[3]*144;bytes=(bytes+4095)/4096*4096;
   items.push_back({layer,projection,type,name,bytes});
  }gguf_free(g);
 }
 std::sort(items.begin(),items.end(),[](const item&a,const item&b){return a.layer<b.layer||(a.layer==b.layer&&a.projection<b.projection);});
 std::stable_sort(items.begin(),items.end(),[](const item&a,const item&b){return xeon_iq_cache_priority(a.type,a.name.c_str())<xeon_iq_cache_priority(b.type,b.name.c_str());});
 std::puts("{");bool first_budget=true;
 for(size_t budget:{size_t(64)<<30,size_t(96)<<30}){
  std::printf("%s\"%zuGiB\":{\"selected\":[",first_budget?"":",\n",budget>>30);first_budget=false;size_t used=0,count=0;bool first=true;
  for(const auto&i:items){if(i.bytes>budget-used)continue;used+=i.bytes;++count;
   std::printf("%s{\"name\":\"%s\",\"type\":\"%s\",\"additional_bytes\":%zu,\"priority\":%d}",first?"":",\n",i.name.c_str(),ggml_type_name(i.type),i.bytes,xeon_iq_cache_priority(i.type,i.name.c_str()));first=false;}
  std::printf("],\"eligible_tensors\":%zu,\"selected_tensors\":%zu,\"additional_bytes\":%zu,\"budget_bytes\":%zu}",items.size(),count,used,budget);
 }
 std::puts("\n}");
}
