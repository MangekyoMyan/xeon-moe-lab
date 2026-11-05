#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
extern "C" float ggml_table_f32_f16[65536];
#include "iq-small-kernels.inc"
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<bool two> static void native(int k,float *s,const iq_source<two> *w,const block_q8_K*x){
 if constexpr(two)ggml_vec_dot_iq2_s_q8_K(k,s,0,w,0,x,0,1);else ggml_vec_dot_iq3_s_q8_K(k,s,0,w,0,x,0,1);
}
template<bool two> static void check() {
 std::mt19937 rng(123);size_t outputs=0;
 for(int k:{256,512,768,2304,5120})for(int trial=0;trial<512;++trial){
  int blocks=k/256;std::vector<iq_source<two>>w(blocks);std::vector<iq_compact_block>c(blocks);std::vector<block_q8_K>x(blocks);
  for(int b=0;b<blocks;++b){
   auto *p=reinterpret_cast<uint8_t *>(&w[b]);for(size_t i=0;i<sizeof(w[b]);++i)p[i]=rng();
   uint16_t d;do{d=rng();}while((d&0x7c00)==0x7c00);w[b].d=d;
   x[b].d=float(int(rng()%2001)-1000)/1024;for(auto&q:x[b].qs)q=int(rng()%256)-128;
   iq_pack<two>(w[b],c[b]);
  }
  float ref;native<two>(k,&ref,w.data(),x.data());
  float direct=iq_direct_dot<two>(k,w.data(),x.data()),compact=iq_compact_dot<two>(k,c.data(),x.data());
  need(!std::memcmp(&ref,&direct,4),"direct bit mismatch");need(!std::memcmp(&ref,&compact,4),"compact bit mismatch");++outputs;
 }
 std::printf("{\"kind\":\"correctness\",\"type\":\"%s\",\"outputs\":%zu,\"bit_mismatches\":0}\n",two?"IQ2_S":"IQ3_S",outputs);
}
template<bool two> static void bench(const char*shard,const char*tensor) {
 auto g=gguf_init_from_file(shard,{true,nullptr});need(g,"GGUF");auto idx=gguf_find_tensor(g,tensor);need(idx>=0,"tensor");
 need(gguf_get_tensor_type(g,idx)==(two?GGML_TYPE_IQ2_S:GGML_TYPE_IQ3_S),"type");auto ne=gguf_get_tensor_ne(g,idx);
 need(ne[2]>=48&&ne[3]==1,"shape");int k=ne[0],m=ne[1],blocks=k/256;size_t rows=size_t(m)*48;
 std::vector<iq_source<two>>w(rows*blocks);std::vector<iq_compact_block>c(rows*blocks);
 std::ifstream file(shard,std::ios::binary);file.seekg(gguf_get_data_offset(g)+gguf_get_tensor_offset(g,idx));
 need(bool(file.read(reinterpret_cast<char*>(w.data()),w.size()*sizeof(w[0]))),"weights");gguf_free(g);
 double start=now();
 #pragma omp parallel for schedule(static) num_threads(12)
 for(size_t b=0;b<w.size();++b)iq_pack<two>(w[b],c[b]);
 double pack_ms=now()-start;
 std::mt19937 rng(77);std::vector<float>f(k*48);for(auto&v:f)v=float(int(rng()%2001)-1000)/2048;
 std::vector<block_q8_K>x(blocks*48);for(int e=0;e<48;++e)quantize_row_q8_K(f.data()+e*k,x.data()+e*blocks,k);
 bool broadcast=std::string(tensor).find("down")==std::string::npos;
 for(int threads:{1,12}) {
  omp_set_dynamic(0);omp_set_num_threads(threads);int errors=0;
  #pragma omp parallel reduction(+:errors)
  {cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(omp_get_thread_num(),&mask);errors+=sched_setaffinity(0,sizeof(mask),&mask)!=0;}
  need(!errors,"pin");
  for(int active:{6,48}) {
   const size_t count=size_t(m)*active;std::vector<float>a(count),b(count),d(count);std::vector<double>native_times,compact_times,direct_times;
   auto run=[&](int method) {
    double t=now();
    #pragma omp parallel for schedule(static)
    for(size_t row=0;row<count;++row) {
     const auto *input=x.data()+(broadcast?0:row/m)*blocks;
     if(method==0)native<two>(k,&a[row],w.data()+row*blocks,input);
     else if(method==1)b[row]=iq_compact_dot<two>(k,c.data()+row*blocks,input);
     else d[row]=iq_direct_dot<two>(k,w.data()+row*blocks,input);
    }
    return now()-t;
   };
   run(0);run(1);run(2);need(!std::memcmp(a.data(),b.data(),count*4)&&!std::memcmp(a.data(),d.data(),count*4),"real weights bit mismatch");
   for(int rep=0;rep<5;++rep)for(int method:{0,1,2,2,1,0}){
    double ms=run(method);(method==0?native_times:method==1?compact_times:direct_times).push_back(ms);
   }
   std::printf("{\"kind\":\"timing\",\"type\":\"%s\",\"direction\":\"%s\",\"threads\":%d,\"experts\":%d,\"native_ms\":%.6f,\"compact_ms\":%.6f,\"direct_ms\":%.6f,\"compact_speedup\":%.4f,\"direct_speedup\":%.4f,\"pack48_ms\":%.3f,\"bit_mismatches\":0}\n",
    two?"IQ2_S":"IQ3_S",broadcast?"up":"down",threads,active,med(native_times),med(compact_times),med(direct_times),med(native_times)/med(compact_times),med(native_times)/med(direct_times),pack_ms);std::fflush(stdout);
  }
 }
}
int main(int argc,char**argv)try{
 auto cpu=ggml_backend_cpu_init();need(cpu,"cpu");
 for(auto grid:iq2s_grid)for(int j=0;j<8;++j){int v=(grid>>(8*j))&255;need(v==8||v==25||v==43,"IQ2 grid");}
 for(auto grid:iq3s_grid)for(int j=0;j<4;++j){int v=(grid>>(8*j))&255;need(v>=1&&v<=15&&(v&1),"IQ3 grid");}
 check<true>();check<false>();
 if(argc==4){if(std::string(argv[1])=="IQ2_S")bench<true>(argv[2],argv[3]);else if(std::string(argv[1])=="IQ3_S")bench<false>(argv[2],argv[3]);else throw std::runtime_error("type");}
 else need(argc==1,"[type shard tensor]");
 ggml_backend_free(cpu);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
