#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include "iq-rhs.h"
#include "iq3-cache.h"
#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <dlfcn.h>
#include "compact-kernel.inc"
struct alignas(16) correction_block { int16_t pairs[16][8]; };
static size_t outputs=0,fallbacks=0;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static float from_bits(uint32_t b){float v;std::memcpy(&v,&b,4);return v;}
static uint32_t bits(float x){uint32_t v;std::memcpy(&v,&x,4);return v;}
static float ref(int n,const compact_block*w,const block_q8_K*x){float out;xeon_iq3_compact_dot(n,&out,0,w,0,x,0,1);return out;}
static void check(int n,const std::vector<compact_block>&w,const std::vector<block_q8_K>&x,bool unsafe){
 const float expected=ref(n,w.data(),x.data());
 for(int workers:{1,12,24}){
  std::vector<correction_block>correction(n/256);std::vector<uint8_t>safe(workers,0);
  // Reverse partition order deliberately, including empty ranges and a last-block fault.
  for(int worker=workers-1;worker>=0;--worker)safe[worker]=xeon_iq3_rhs_prepare_v1(n,x.data(),worker*(n/256)/workers,(worker+1)*(n/256)/workers,correction.data());
  bool all=std::all_of(safe.begin(),safe.end(),[](uint8_t x){return x==1;});need(all==!unsafe,"partition safety flags");float got;
  if(all)xeon_iq3_rhs_dot_v1(n,&got,w.data(),x.data(),correction.data());else{got=ref(n,w.data(),x.data());++fallbacks;}
  if(bits(expected)!=bits(got)){std::fprintf(stderr,"bit mismatch K%d workers%d expected%08x got%08x MXCSR%08x\n",n,workers,bits(expected),bits(got),_mm_getcsr());throw std::runtime_error("output mismatch");}
  ++outputs;
 }
}
int main()try{
 ggml_cpu_init();Dl_info info{};need(dladdr(reinterpret_cast<void*>(xeon_iq3_compact_dot),&info),"provider");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical("/home/codex/projects/llm-lab/exploration-20261112/sparse-cache/release-v2-a/libggml-cpu.so.0.28.0"),"wrong reference provider");
 const uint32_t saved=_mm_getcsr();std::mt19937 rng(47151);
 try{
  for(uint32_t control:{saved&~uint32_t(0x8040),saved|uint32_t(0x8040)}){
   _mm_setcsr(control);
   for(int k:{256,512,768,2304,5120})for(int it=0;it<300;++it){
    std::vector<compact_block>w(k/256);std::vector<block_q8_K>x(k/256);
    for(auto &v:w){v.d=float(int(rng()%1024)-512)/4096;for(auto &q:v.qs)q=rng();for(auto&s:v.scale)s=2*(rng()%16)+1;}
    for(auto &v:x){v.d=float(int(rng()%1024)-512)/4096;for(auto&q:v.qs)q=int(rng()%255)-127;}
    check(k,w,x,false);
   }
   const uint32_t scales[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc01234,0xffc04321,0x7f800001,0xff800003};
   for(int k:{256,512,768,2304,5120})for(uint32_t d:scales)for(int pattern=0;pattern<5;++pattern){
    std::vector<compact_block>w(k/256);std::vector<block_q8_K>x(k/256);
    for(size_t b=0;b<w.size();++b){w[b].d=pattern==0?0.0f:pattern==1?-0.0f:pattern==2?1.0f:pattern==3?65504.0f:0x1p-24f;for(auto&q:w[b].qs)q=pattern==2?0x77:pattern==3?0xff:uint8_t(rng());for(auto&s:w[b].scale)s=31;
     x[b].d=from_bits(d);for(int i=0;i<256;++i)x[b].qs[i]=pattern==0?0:pattern==1?127:pattern==2?-127:(i&1?127:-127);}
    check(k,w,x,(d&0x7f800000)==0x7f800000);
    // The fault is in the last block, not worker0's first block.
    x.back().qs[255]=-128;check(k,w,x,true);
    x.back().qs[255]=127;x.back().d=from_bits(0x7fc05555);check(k,w,x,true);
   }
   std::vector<compact_block>w(1);std::vector<block_q8_K>x(1);w[0].d=1;for(auto&s:w[0].scale)s=1;w[0].qs[0]=8;x[0].d=1;x[0].qs[0]=-128;
   need(bits(ref(256,w.data(),x.data()))==0xc3000000U,"wrong legacy PSIGNB semantics");check(256,w,x,true);
  }
 }catch(...){_mm_setcsr(saved);throw;}
 _mm_setcsr(saved);std::printf("{\"pass\":true,\"outputs\":%zu,\"fallbacks\":%zu,\"initial_mxcsr\":%u}\n",outputs,fallbacks,saved);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
