#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
int main(int argc,char**argv){
 if(argc!=2)return 1;auto g=gguf_init_from_file(argv[1],{true,nullptr});if(!g)return 2;
 std::puts("{");bool first=true;
 for(int64_t i=0;i<gguf_get_n_kv(g);++i){
  std::string key=gguf_get_key(g,i);
  if(key!="general.architecture" && (key.find("deepseek41.")!=0 || key.find("ple.")!=std::string::npos || key.find("engram.")!=std::string::npos))continue;
  auto type=gguf_get_kv_type(g,i);
  if(type==GGUF_TYPE_ARRAY && gguf_get_arr_n(g,i)>100)continue;
  std::printf("%s\"%s\":",first?"":",\n",key.c_str());first=false;
  if(type==GGUF_TYPE_UINT32)std::printf("%u",gguf_get_val_u32(g,i));
  else if(type==GGUF_TYPE_INT32)std::printf("%d",gguf_get_val_i32(g,i));
  else if(type==GGUF_TYPE_FLOAT32)std::printf("%.9g",double(gguf_get_val_f32(g,i)));
  else if(type==GGUF_TYPE_BOOL)std::printf("%s",gguf_get_val_bool(g,i)?"true":"false");
  else if(type==GGUF_TYPE_STRING)std::printf("\"%s\"",gguf_get_val_str(g,i));
  else if(type==GGUF_TYPE_ARRAY && gguf_get_arr_type(g,i)==GGUF_TYPE_UINT32){
   auto p=static_cast<const uint32_t *>(gguf_get_arr_data(g,i));std::printf("[");for(size_t j=0;j<gguf_get_arr_n(g,i);++j)std::printf("%s%u",j?",":"",p[j]);std::printf("]");
  }else if(type==GGUF_TYPE_ARRAY && gguf_get_arr_type(g,i)==GGUF_TYPE_INT32){
   auto p=static_cast<const int32_t *>(gguf_get_arr_data(g,i));std::printf("[");for(size_t j=0;j<gguf_get_arr_n(g,i);++j)std::printf("%s%d",j?",":"",p[j]);std::printf("]");
  }else std::printf("null");
 }
 std::puts("\n}");gguf_free(g);
}
