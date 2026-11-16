#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <dlfcn.h>

static size_t cases=0,outputs=0;
static bool enabled;
static void (*counters)(uint64_t*)=nullptr;
static void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static void run(FILE*file,ggml_backend_t cpu,int k,int m,int columns,int tokens,bool strided,bool fp32,bool duplicate){
 const int used=columns>6?columns:6,e=std::max(7,used+1);auto ctx=ggml_init({4*1024*1024,nullptr,true});need(ctx,"context");
 auto w=ggml_new_tensor_3d(ctx,GGML_TYPE_MXFP4,k,m,e);ggml_set_name(w,"blk.0.ffn_up_exps.weight");
 const auto xtype=fp32?GGML_TYPE_F32:GGML_TYPE_Q8_0;const size_t row=ggml_row_size(xtype,k);
 auto backing=ggml_new_tensor_3d(ctx,xtype,k,strided?columns*2:columns,strided?tokens*2+1:tokens);
 auto x=strided?ggml_view_3d(ctx,backing,k,columns,tokens,row*2,row*columns*4,row):backing;
 auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);auto y=ggml_mul_mat_id(ctx,w,x,ids);auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
 std::mt19937 rng(9144+k+m+columns+tokens+int(strided)+int(fp32));
 std::vector<block_mxfp4>weights(size_t(k/32)*m*e);for(auto&b:weights){b.e=116+rng()%14;for(auto&q:b.qs)q=rng();}ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
 std::vector<int32_t>routes(used*tokens);for(int t=0;t<tokens;++t)for(int i=0;i<used;++i)routes[t*used+i]=(i+t+1)%e;
 if(duplicate)for(int t=0;t<tokens;++t){routes[t*used+1]=routes[t*used];routes[t*used+5]=-1;}
 ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
 std::vector<uint8_t>activation(ggml_nbytes(backing));std::vector<float>floats(k),result(ggml_nelements(y));
 for(int phase=0;phase<3;++phase){
  for(size_t off=0;off<activation.size();off+=row){
   if(fp32){for(auto &v:floats)v=float(int(rng()%2001)-1000)/2048;std::memcpy(activation.data()+off,floats.data(),row);}
   else {auto*q=reinterpret_cast<block_q8_0*>(activation.data()+off);for(int b=0;b<k/32;++b){q[b].d=0x2400+16*(b%7);for(auto &v:q[b].qs)v=int(rng()%255)-127;}
    if(phase==0)q[(k/32)-1].qs[31]=-128;
    if(phase==1 && (off/row)%3==0)q[0].d=0x7e19;
   }
  }
  ggml_backend_tensor_set(backing,activation.data(),0,activation.size());
  for(int threads:{12,1,24,12}){
   ggml_backend_cpu_set_n_threads(cpu,threads);
   auto params=ggml_threadpool_params_default(threads);params.poll=0;params.strict_cpu=true;for(int i=0;i<threads;++i)params.cpumask[i]=true;
   auto pool=ggml_threadpool_new(&params);need(pool,"pool");ggml_backend_cpu_set_threadpool(cpu,pool);
   for(bool use_ref:{false,true}) {
    ggml_backend_cpu_set_use_ref(cpu,use_ref);uint64_t before[3],after[3];counters(before);
    need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_tensor_get(y,result.data(),0,result.size()*4);counters(after);
    if(!enabled||use_ref||tokens!=1||columns>64)need(after[0]==before[0]&&after[1]==before[1],"prepack ran when disabled");
    else {
     need(after[0]>before[0],"prepare path was bypassed");
     if(!fp32&&phase==0)need(after[1]==before[1],"unsafe input used prepared dot");
     if(!duplicate&&(fp32||phase==2))need(after[1]>before[1],"safe input did not use prepared dot");
    }
    const uint64_t meta[]={uint64_t(k),uint64_t(m),uint64_t(columns),uint64_t(tokens),uint64_t(strided),uint64_t(fp32),uint64_t(duplicate),uint64_t(phase),uint64_t(threads),uint64_t(use_ref),result.size()};
    need(std::fwrite(meta,sizeof(meta),1,file)==1,"metadata write");need(std::fwrite(result.data(),4,result.size(),file)==result.size(),"output write");++cases;outputs+=result.size();
   }
   ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);
  }
 }
 ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main(int argc,char**argv)try{
 need(argc==4,"usage: test-graph OUTPUT EXPECTED_CPU_LIBRARY ENABLED");enabled=std::strcmp(argv[3],"1")==0;counters=reinterpret_cast<void(*)(uint64_t*)>(dlsym(RTLD_DEFAULT,"xeon_mx_test_counters"));need(counters,"missing validation interposer");Dl_info info{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cpu_init),&info),"provider");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical(argv[2]),"wrong CPU provider");
 auto cpu=ggml_backend_cpu_init();need(cpu,"backend");FILE*file=std::fopen(argv[1],"wb");need(file,"output file");
 for(int k:{32,64,96,160,256,2304,5120})for(int columns:{1,2,3,6})for(bool strided:{false,true})for(bool fp32:{false,true})for(bool duplicate:{false,true})run(file,cpu,k,17,columns,1,strided,fp32,duplicate);
 for(int k:{32,256,5120})for(int columns:{1,2,3,6})for(bool strided:{false,true})for(bool fp32:{false,true})run(file,cpu,k,17,columns,3,strided,fp32,false);
 for(int columns:{64,65})for(bool fp32:{false,true})for(bool strided:{false,true})run(file,cpu,32,17,columns,1,strided,fp32,false);
 run(file,cpu,5120,2304,1,1,false,true,false);run(file,cpu,2304,5120,6,1,false,true,false);
 need(std::fclose(file)==0,"close");ggml_backend_free(cpu);uint64_t totals[3];counters(totals);std::fprintf(stderr,"prepare=%llu dot=%llu unsafe=%llu\n",(unsigned long long)totals[0],(unsigned long long)totals[1],(unsigned long long)totals[2]);std::printf("{\"pass\":true,\"cases\":%zu,\"outputs\":%zu}\n",cases,outputs);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
