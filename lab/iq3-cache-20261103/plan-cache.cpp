#include "gguf.h"
#include "ggml.h"
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
int main(int argc,char**argv){
    std::map<std::pair<int,int>,std::pair<std::string,size_t>> list;
    for(int a=1;a<argc;++a){auto g=gguf_init_from_file(argv[a],{true,nullptr});if(!g)return 1;
        for(int64_t i=0;i<gguf_get_n_tensors(g);++i){
            std::string n=gguf_get_tensor_name(g,i);int l=-1;if(sscanf(n.c_str(),"blk.%d.",&l)!=1||l<0||l>=37||gguf_get_tensor_type(g,i)!=GGML_TYPE_IQ3_XXS)continue;
            int p=-1;int j=0;for(const auto*s:{"ffn_gate_up_exps.weight","ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}){if(n=="blk."+std::to_string(l)+"."+s)p=j;++j;}if(p<0)continue;
            auto ne=gguf_get_tensor_ne(g,i);size_t bytes=size_t(ne[0]/256)*ne[1]*ne[2]*ne[3]*144;bytes=(bytes+4095)/4096*4096;list[{l,p}]={n,bytes};
        }gguf_free(g);
    }
    size_t used=0,count=0;std::puts("{\"selected\":[");bool first=true;
    for(const auto&v:list){if(v.second.second>(size_t(64)<<30)-used)continue;used+=v.second.second;++count;
        std::printf("%s{\"name\":\"%s\",\"additional_bytes\":%zu}",first?"":",\n",v.second.first.c_str(),v.second.second);first=false;}
    std::printf("\n],\"eligible_tensors\":%zu,\"selected_tensors\":%zu,\"additional_bytes\":%zu,\"budget_bytes\":%zu}\n",list.size(),count,used,size_t(64)<<30);
}
