#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <dlfcn.h>
#include <filesystem>
#include "kernel.inc"

static size_t checks=0,fallbacks=0;
static void need(bool value,const char * message){if(!value)throw std::runtime_error(message);}
static uint32_t bits(float x){uint32_t v;std::memcpy(&v,&x,4);return v;}
static float reference(int n,const block_mxfp4 * w,const block_q8_0 * x){float v;ggml_vec_dot_mxfp4_q8_0(n,&v,0,w,0,x,0,1);return v;}
static float candidate(int mode,int n,const block_mxfp4 * w,const block_q8_0 * x,const mx_prepared & p){
 switch(mode){case 1:return mx_dot<1>(n,w,x,p);case 2:return mx_dot<2>(n,w,x,p);case 3:return mx_dot<3>(n,w,x,p);case 5:return mx_dot<5>(n,w,x,p);case 7:return mx_dot<7>(n,w,x,p);default:throw std::runtime_error("mode");}}
static void check(int n,const std::vector<block_mxfp4>& w,const std::vector<block_q8_0>& x,bool expect_fallback) {
 const float base=reference(n,w.data(),x.data());mx_prepared p(n/32);
 for(int mode:{1,2,3,5,7}){
  mx_prepare(n,x.data(),p,mode);need(!p.safe==expect_fallback,"fallback eligibility");
  const float got=candidate(mode,n,w.data(),x.data(),p);++checks;if(!p.safe)++fallbacks;
  if(bits(base)!=bits(got)){std::fprintf(stderr,"mismatch n=%d mode=%d ref=%08x got=%08x mxcsr=%08x\n",n,mode,bits(base),bits(got),_mm_getcsr());throw std::runtime_error("bit mismatch");}
 }
}
static void correctness(){
 const uint32_t saved=_mm_getcsr();std::mt19937 rng(90531);
 try {
  for(uint32_t control:{saved&~uint32_t(0x8040),saved|uint32_t(0x8040)}){
   _mm_setcsr(control);
   for(int n:{0,32,64,96,160,256,2304,5120}) for(int iteration=0;iteration<200;++iteration){
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(auto &b:w){b.e=110+rng()%30;for(auto &q:b.qs)q=rng();}
    for(auto &b:x){b.d=0x1000+rng()%0x2800;for(auto &q:b.qs)q=int(rng()%255)-127;}
    check(n,w,x,false);
   }
   for(int n:{32,64,96,160,2304,5120}) for(int code:{0x00,0x88,0x77,0xff}) for(int sign:{-127,127}) {
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(auto &b:w){b.e=128;for(auto &q:b.qs)q=code;}
    for(auto &b:x){b.d=0x3c00;for(auto &q:b.qs)q=sign;}
    check(n,w,x,false);
   }
   for(int n:{2304,5120}) for(bool opposite:{false,true}) {
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(size_t i=0;i<w.size();++i){w[i].e=240;for(auto &q:w[i].qs)q=opposite&&(i&1)?0xff:0x77;x[i].d=0x3c00;for(auto &q:x[i].qs)q=127;}
    check(n,w,x,false);
   }
   for(int n:{192,2304,5120}) {
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(auto &b:w)b.e=128;for(auto &b:x){b.d=0x3c00;for(auto &q:b.qs)q=1;}
    for(int b=0;b<6;++b){w[b].e=b/2==1?125:149;for(auto &q:w[b].qs)q=b/2==2?0x99:0x11;}
    check(n,w,x,false);
   }
   const uint16_t scales[]={0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff,0x7c00,0xfc00,0x7e01,0xfe51,0x7c01,0xfc03};
   for(int n:{32,64,96,160}) for(uint16_t scale:scales) for(uint8_t e:{0,1,2,64,127,128,254,255}) for(int pattern=0;pattern<5;++pattern){
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(size_t b=0;b<w.size();++b){w[b].e=e;for(int j=0;j<16;++j)w[b].qs[j]=(j&15)|((15-j)<<4);x[b].d=scale;for(int j=0;j<32;++j)x[b].qs[j]=pattern==0?0:pattern==1?127:pattern==2?-127:((j+b)&1?127:-127);}
    if(pattern==4)x.back().qs[31]=-128;
    check(n,w,x,pattern==4||((scale&0x7c00)==0x7c00));
   }
   for(int n:{32,64,96}){
    std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
    for(auto &b:w)b.e=128;for(auto &b:x)b.d=0x3c00;
    const int sentinel=n==96?2:0;w[sentinel].qs[0]=9;x[sentinel].qs[0]=-128;
    need(bits(reference(n,w.data(),x.data()))==(n==64?0xc3000000U:0x43000000U),"wrong reference AVX/tail semantics");check(n,w,x,true);
   }
  }
 }catch(...){_mm_setcsr(saved);throw;}
 _mm_setcsr(saved);
 std::printf("{\"correctness\":true,\"outputs\":%zu,\"fallback_outputs\":%zu,\"initial_mxcsr\":%u}\n",checks,fallbacks,saved);std::fflush(stdout);
}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return(x[(x.size()-1)/2]+x[x.size()/2])/2;}
static void print_vector(const char *name,const std::vector<double> &v){std::printf("\"%s\":[",name);for(size_t i=0;i<v.size();++i)std::printf("%s%.9f",i?",":"",v[i]);std::printf("]");}
static void timing(int k,int m,int experts,bool shared,int threads){
 const size_t blocks=k/32,rows=size_t(m)*experts;const int inputs=shared?1:experts;
 std::vector<block_mxfp4>w(rows*blocks);std::vector<block_q8_0>x(size_t(inputs)*blocks);std::vector<mx_prepared>prep;for(int i=0;i<inputs;++i)prep.emplace_back(blocks);
 std::vector<float>base(rows),out(rows);std::mt19937 rng(3731+k+m+experts);
 for(auto &b:w){b.e=120+rng()%8;for(auto&q:b.qs)q=rng();}
 for(auto &b:x){b.d=0x1800+rng()%0x800;for(auto&q:b.qs)q=int(rng()%255)-127;}
 auto compute=[&](int mode,std::vector<float>&result){
  if(mode)for(int i=0;i<inputs;++i)mx_prepare(k,x.data()+i*blocks,prep[i],mode);
  #pragma omp parallel num_threads(threads)
  {
   const int ith=omp_get_thread_num();cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(ith,&mask);need(sched_setaffinity(0,sizeof(mask),&mask)==0,"worker affinity");need(omp_get_num_threads()==threads,"team size");
   #pragma omp for schedule(static)
   for(size_t row=0;row<rows;++row){size_t input=shared?0:row/m;result[row]=mode?candidate(mode,k,w.data()+row*blocks,x.data()+input*blocks,prep[input]):reference(k,w.data()+row*blocks,x.data()+input*blocks);}
  }
 };
 compute(0,base);
 for(int mode:{1,2,3,5,7}){
  compute(mode,out);need(std::memcmp(base.data(),out.data(),rows*4)==0,"timing correctness");checks+=rows;
  std::vector<double>a,b,ratios;
  for(int warm=0;warm<2;++warm){compute(0,base);compute(mode,out);}
  for(int round=0;round<12;++round){double a0,b0,b1,a1,t;
   t=now();compute(0,base);a0=now()-t;t=now();compute(mode,out);b0=now()-t;t=now();compute(mode,out);b1=now()-t;t=now();compute(0,base);a1=now()-t;
   need(std::memcmp(base.data(),out.data(),rows*4)==0,"timing bit mismatch");checks+=rows;
   a.push_back(a0);a.push_back(a1);b.push_back(b0);b.push_back(b1);ratios.push_back((a0+a1)/(b0+b1));
  }
  std::printf("{\"k\":%d,\"m\":%d,\"experts\":%d,\"shared\":%s,\"threads\":%d,\"mode\":%d,\"base_ms\":%.9f,\"candidate_ms\":%.9f,\"paired_speedup\":%.9f,",k,m,experts,shared?"true":"false",threads,mode,median(a),median(b),median(ratios));print_vector("base_samples",a);std::printf(",");print_vector("candidate_samples",b);std::printf(",");print_vector("paired_ratios",ratios);std::printf("}\n");std::fflush(stdout);
 }
}
int main(int argc,char**argv)try{
 ggml_cpu_init();Dl_info info{};need(dladdr(reinterpret_cast<void *>(ggml_vec_dot_mxfp4_q8_0),&info),"reference provider");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical("/home/codex/projects/llm-lab/exploration-20261112/sparse-cache/release-v2-a/libggml-cpu.so.0.28.0"),"unexpected reference provider");std::fprintf(stderr,"Reference provider: %s\n",info.dli_fname);
 correctness();if(argc>1&&std::strcmp(argv[1],"--timing")==0)for(int threads:{1,12})for(int experts:{6,24}){timing(5120,2304,experts,true,threads);timing(2304,5120,experts,false,threads);}
 std::printf("{\"complete\":true,\"outputs_checked\":%zu}\n",checks);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
