#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "iq3-cache.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <dlfcn.h>
static bool enabled;static void(*counts)(uint64_t*)=nullptr;static size_t cases=0,values=0;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static void run(FILE*f,ggml_backend_t cpu,int k,int columns,int tokens,bool strided,bool fp32,bool duplicate,bool mixed,int rows=17){
 const int used=columns>6?columns:6,E=std::max(7,used+1),M=rows;auto ctx=ggml_init({8*1024*1024,nullptr,true});need(ctx,"context");
 auto w=ggml_new_tensor_3d(ctx,GGML_TYPE_IQ3_XXS,k,M,E);ggml_set_name(w,k==2304?"blk.0.ffn_down_exps.weight":"blk.0.ffn_up_exps.weight");
 const auto xtype=fp32?GGML_TYPE_F32:GGML_TYPE_Q8_K;const size_t row=ggml_row_size(xtype,k);
 auto backing=ggml_new_tensor_3d(ctx,xtype,k,strided?columns*2:columns,strided?tokens*2+1:tokens);auto x=strided?ggml_view_3d(ctx,backing,k,columns,tokens,row*2,row*columns*4,row):backing;
 auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);auto y=ggml_mul_mat_id(ctx,w,x,ids);auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,y);
 ggml_tensor *mw=nullptr,*mx=nullptr,*my=nullptr;
 if(mixed){mw=ggml_new_tensor_3d(ctx,GGML_TYPE_MXFP4,k,M,E);mx=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,columns,tokens);my=ggml_mul_mat_id(ctx,mw,mx,ids);ggml_build_forward_expand(graph,my);}
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
 std::mt19937 rng(27641+k+columns+tokens+int(strided)+int(fp32));std::vector<uint8_t>weights(ggml_nbytes(w));for(auto&q:weights)q=rng();
 for(size_t b=0;b<weights.size();b+=ggml_type_size(GGML_TYPE_IQ3_XXS)){uint16_t d=0x1800+16*((b/ggml_type_size(GGML_TYPE_IQ3_XXS))%8);std::memcpy(weights.data()+b,&d,2);}ggml_backend_tensor_set(w,weights.data(),0,weights.size());
 if(mixed){std::vector<block_mxfp4>v(ggml_nbytes(mw)/sizeof(block_mxfp4));for(auto &b:v){b.e=120+rng()%8;for(auto&q:b.qs)q=rng();}ggml_backend_tensor_set(mw,v.data(),0,ggml_nbytes(mw));std::vector<float>input(ggml_nelements(mx));for(auto &v:input)v=float(int(rng()%2001)-1000)/2048;ggml_backend_tensor_set(mx,input.data(),0,input.size()*4);}
 std::vector<int32_t>routes(used*tokens);for(int t=0;t<tokens;++t)for(int i=0;i<used;++i)routes[t*used+i]=(i+t+1)%E;if(duplicate)for(int t=0;t<tokens;++t){routes[t*used+1]=routes[t*used];routes[t*used+5]=-1;}ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
 std::vector<uint32_t>all(E);std::iota(all.begin(),all.end(),0);std::vector<std::vector<uint32_t>> selections={{},all,{uint32_t(E-1),3,1},{0}};
 const size_t per_expert=size_t(k/256)*M*144;
 for(size_t selection=0;selection<selections.size();++selection){
  const auto &sel=selections[selection];void*owner=nullptr;
  if(!sel.empty()){size_t bytes=(sel.size()*per_expert+4095)/4096*4096;owner=xeon_iq3_cache_create_v1(bytes,0,12);need(owner,"owner");need(xeon_iq3_cache_add_selected_v2(owner,w,sel.data(),sel.size())==1,"cache add");}
  const bool has_hit=std::any_of(routes.begin(),routes.end(),[&](int id){return id>=0&&std::find(sel.begin(),sel.end(),uint32_t(id))!=sel.end();});
  for(int phase=0;phase<3;++phase){
   std::vector<uint8_t>data(ggml_nbytes(backing));std::vector<float>floats(k);
   for(size_t offset=0;offset<data.size();offset+=row){
    if(fp32){for(auto&v:floats)v=float(int(rng()%2001)-1000)/2048;std::memcpy(data.data()+offset,floats.data(),row);}
    else {auto*q=reinterpret_cast<block_q8_K*>(data.data()+offset);for(int b=0;b<k/256;++b){q[b].d=.00390625f;for(auto &v:q[b].qs)v=int(rng()%255)-127;}
     if(phase==0)q[k/256-1].qs[255]=-128;
     if(phase==1&&(offset/row)%3==0){uint32_t nan=0x7fc05123;std::memcpy(&q[k/256-1].d,&nan,4);}
     for(int b=0;b<k/256;++b)for(int g=0;g<16;++g){int sum=0;for(int i=0;i<16;++i)sum+=q[b].qs[g*16+i];q[b].bsums[g]=sum;}
    }
   }
   ggml_backend_tensor_set(backing,data.data(),0,data.size());
   for(int threads:{12,1,24}){
    ggml_backend_cpu_set_n_threads(cpu,threads);auto pp=ggml_threadpool_params_default(threads);pp.poll=0;pp.strict_cpu=true;for(int i=0;i<threads;++i)pp.cpumask[i]=true;auto pool=ggml_threadpool_new(&pp);need(pool,"pool");ggml_backend_cpu_set_threadpool(cpu,pool);
    for(bool reference:{false,true}){
     ggml_backend_cpu_set_use_ref(cpu,reference);uint64_t before[3],after[3];counts(before);need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"compute");counts(after);
     const bool active=enabled&&!reference&&tokens==1&&columns<=64&&!sel.empty();
     if(!active)need(after[0]==before[0]&&after[1]==before[1],"disabled IQ preparation");else{
      need(after[0]>before[0],"prepare bypassed");
      if(!has_hit||(!fp32&&phase==0))need(after[1]==before[1],"unsafe or miss prepared dot");
      if(has_hit&&(fp32||phase==2))need(after[1]>before[1],"safe compact hit missed prepared dot");
     }
     std::vector<float>out(ggml_nelements(y)+(mixed?ggml_nelements(my):0));ggml_backend_tensor_get(y,out.data(),0,ggml_nbytes(y));if(mixed)ggml_backend_tensor_get(my,out.data()+ggml_nelements(y),0,ggml_nbytes(my));
     uint64_t metadata[]={uint64_t(k),uint64_t(M),uint64_t(columns),uint64_t(tokens),uint64_t(strided),uint64_t(fp32),uint64_t(duplicate),uint64_t(mixed),selection,uint64_t(phase),uint64_t(threads),uint64_t(reference),out.size()};need(std::fwrite(metadata,sizeof(metadata),1,f)==1,"metadata write");need(std::fwrite(out.data(),4,out.size(),f)==out.size(),"output write");++cases;values+=out.size();
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);
   }
  }
  xeon_iq3_cache_free_v1(owner);need(xeon_iq3_cache_total_v1()==0,"cache leaked");
 }
 ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main(int argc,char**argv)try{
 need(argc==4,"usage OUTPUT EXPECTED_CPU_LIBRARY ENABLED");enabled=std::strcmp(argv[3],"1")==0;counts=reinterpret_cast<void(*)(uint64_t*)>(dlsym(RTLD_DEFAULT,"xeon_iq_test_counters"));need(counts,"interposer missing");Dl_info info{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cpu_init),&info),"provider");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical(argv[2]),"wrong CPU library");
 auto cpu=ggml_backend_cpu_init();need(cpu,"CPU");FILE*f=std::fopen(argv[1],"wb");need(f,"output");
 for(int k:{256,512,768,2304,5120})for(int columns:{1,2,3,6})for(bool strided:{false,true})for(bool fp32:{false,true})for(bool duplicate:{false,true})run(f,cpu,k,columns,1,strided,fp32,duplicate,false);
 for(int columns:{1,2,3,6})for(bool strided:{false,true})for(bool fp32:{false,true})run(f,cpu,256,columns,3,strided,fp32,false,false);
 for(int columns:{64,65})for(bool fp32:{false,true})run(f,cpu,256,columns,1,true,fp32,false,false);
 for(int k:{256,2304,5120})for(bool fp32:{false,true})run(f,cpu,k,6,1,true,fp32,true,true);
 run(f,cpu,5120,1,1,false,true,false,false,2304);run(f,cpu,2304,6,1,false,true,false,false,5120);
 need(std::fclose(f)==0,"close");ggml_backend_free(cpu);uint64_t totals[3];counts(totals);std::printf("{\"pass\":true,\"cases\":%zu,\"outputs\":%zu,\"prepare_calls\":%llu,\"dot_calls\":%llu,\"unsafe_ranges\":%llu}\n",cases,values,(unsigned long long)totals[0],(unsigned long long)totals[1],(unsigned long long)totals[2]);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
