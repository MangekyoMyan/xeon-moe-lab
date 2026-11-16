// Experimental input-abs sidecar. Current compact kernels are included unchanged.
// This is a standalone synthetic probe, not a deployed CPU backend.
#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "iq-rhs.h"
#include <immintrin.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "compact-kernel.inc"
#include "iq-small-kernels.inc"

float ggml_table_f32_f16[65536];
extern "C" float ggml_fp16_to_fp32(ggml_fp16_t h) { return GGML_COMPUTE_FP16_TO_FP32(h); }
extern "C" void ggml_abort(const char *file,int line,const char*,...) {
    std::fprintf(stderr,"assertion %s:%d\n",file,line);std::abort();
}
static void need(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static unsigned fb(float x){unsigned u;std::memcpy(&u,&x,4);return u;}
struct alignas(16) abs_rhs {uint8_t qs[256];};
struct alignas(16) bias_rhs {int16_t pairs[16][8];};
static_assert(sizeof(abs_rhs)==256&&sizeof(bias_rhs)==256,"sidecar sizes");
static bool prepare_abs(int n,const block_q8_K *x,abs_rhs *out) {
    bool safe=true;const __m128i minimum=_mm_set1_epi8(-128);
    for(int b=0;b<n/256;++b) {
        if((fb(x[b].d)&0x7f800000u)==0x7f800000u)safe=false;
        for(int h=0;h<16;++h) {
            const __m128i v=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16*h));
            if(_mm_movemask_epi8(_mm_cmpeq_epi8(v,minimum)))safe=false;
            _mm_store_si128(reinterpret_cast<__m128i *>(out[b].qs+16*h),_mm_abs_epi8(v));
        }
    }
    return safe;
}
template<int T> using weight_type=typename std::conditional<T==1,compact_block,iq_compact_block>::type;
template<int T> using source_type=typename std::conditional<T==1,block_iq3_xxs,typename std::conditional<T==2,block_iq2_s,block_iq3_s>::type>::type;
template<int T> static inline int scale(const weight_type<T> &w,int g,int h) {
    if constexpr(T==1)return w.scale[g];
    else return iq_scale<T==2>(w.scales,g,h);
}
template<int T> __attribute__((noinline)) static float abs_dot(int n,const weight_type<T> *w,const block_q8_K *x,const abs_rhs *rhs) {
    const __m128i lut=T==1?_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62):
                      T==2?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                           _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g));
            const __m128i y1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16));
            const __m128i a0=_mm_load_si128(reinterpret_cast<const __m128i *>(rhs[b].qs+32*g));
            const __m128i a1=_mm_load_si128(reinterpret_cast<const __m128i *>(rhs[b].qs+32*g+16));
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(a0,_mm_sign_epi8(q0,y0)),_mm_set1_epi16(scale<T>(w[b],g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(a1,_mm_sign_epi8(q1,y1)),_mm_set1_epi16(scale<T>(w[b],g,1))));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        const __m256 prod=_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints));
        if constexpr(T==1)acc=_mm256_add_ps(acc,prod);else acc=_mm256_add_ps(prod,acc);
    }
    if constexpr(T==1)return reduce(acc);
    else return (T==2?.125f:1.f)*iq_hsum(acc);
}
template<int T> static float current_dot(int n,const weight_type<T> *w,const block_q8_K *x) {
    if constexpr(T==1)return compact_dot(n,w,x);
    else return iq_compact_dot<T==2>(n,w,x);
}
template<int T> static void make(std::mt19937 &rng,weight_type<T> *w,size_t blocks,bool finite_bench=false) {
    for(size_t i=0;i<blocks;++i) {
        source_type<T> s;auto *p=reinterpret_cast<uint8_t *>(&s);for(size_t j=0;j<sizeof(s);++j)p[j]=uint8_t(rng());
        s.d=finite_bench?uint16_t(0x2400):uint16_t((rng()%0x7c00)|(rng()&0x8000));
        if constexpr(T==1){expanded_block expanded;expand(&s,&expanded,1);compact(expanded,w[i]);}
        else iq_pack<T==2>(s,w[i]);
    }
}
template<int T> static void test() {
    std::mt19937 rng(20261116+T);size_t checks=0,unsafe=0,bias_checks=0;
    for(int n:{256,512,768,2304,5120})for(int it=0;it<512;++it) {
        std::vector<weight_type<T>>w(n/256);std::vector<block_q8_K>x(n/256);
        std::vector<abs_rhs>a(n/256);std::vector<bias_rhs>bias(n/256);make<T>(rng,w.data(),w.size());
        const int exponents[]={-149,-145,-130,-126,-125,-24,-15,0,15};
        for(auto &v:x){v.d=std::ldexp(float(int(rng()%8191)-4095)/4096,exponents[rng()%9]);for(auto &q:v.qs)q=int(rng()%255)-127;}
        need(prepare_abs(n,x.data(),a.data()),"normal prepare rejected");
        const float old=current_dot<T>(n,w.data(),x.data());
        need(fb(old)==fb(abs_dot<T>(n,w.data(),x.data(),a.data())),"input-abs bit mismatch");++checks;
        if constexpr(T==1){need(xeon_iq3_rhs_prepare_v1(n,x.data(),0,n/256,bias.data()),"bias prepare rejected");float got;xeon_iq3_rhs_dot_v1(n,&got,w.data(),x.data(),bias.data());need(fb(old)==fb(got),"stored bias variant mismatch");++bias_checks;}
        if(it<8)for(int bad=0;bad<5;++bad) {
            const auto original=x.back();
            if(bad==0)x.back().qs[255]=-128;
            else {const unsigned u[]={0,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u};std::memcpy(&x.back().d,&u[bad],4);}
            need(!prepare_abs(n,x.data(),a.data()),"unsafe input accepted");++unsafe;x.back()=original;
        }
    }
    std::printf("{\"kind\":\"correctness\",\"type\":%d,\"mxcsr\":%u,\"bit_exact_outputs\":%zu,\"unsafe_rejections\":%zu,\"stored_bias_variant_bit_checks\":%zu}\n",T,_mm_getcsr(),checks,unsafe,bias_checks);
}
static volatile double sink=0;
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2;}
static double thread_ns(){timespec t{};need(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0,"thread CPU clock unavailable");return double(t.tv_sec)*1e9+t.tv_nsec;}
static void array(const std::vector<double>&v){std::printf("[");for(size_t i=0;i<v.size();++i)std::printf("%s%.6f",i?",":"",v[i]);std::printf("]");}
template<int T,bool BIAS=false> static void bench(int n,int rows) {
    std::mt19937 rng(202611160+T);const int blocks=n/256;
    std::vector<weight_type<T>>w(size_t(rows)*blocks);make<T>(rng,w.data(),w.size(),true);
    std::vector<block_q8_K>x(blocks);std::vector<abs_rhs>a(blocks);std::vector<bias_rhs>bias(blocks);
    for(auto &v:x){v.d=.001f;for(auto &q:v.qs)q=int(rng()%255)-127;}
    const int traversals=std::max(1,32768/rows);std::vector<double>base,candidate,ratios,cpu_base,cpu_candidate,cpu_ratios;
    for(int repeat=0;repeat<5;++repeat) {
        double local[4],cpu_local[4];
        for(int arm=0;arm<4;++arm) {
            bool alt=(arm==1||arm==2);if(repeat%2)alt=!alt;double total=0;
            const auto start=std::chrono::steady_clock::now();
            const double cpu_start=thread_ns();
            for(int tr=0;tr<traversals;++tr) {
                asm volatile(""::"r"(w.data()),"r"(x.data()):"memory");
                if(alt){if constexpr(BIAS)need(xeon_iq3_rhs_prepare_v1(n,x.data(),0,blocks,bias.data()),"bench bias unsafe");else need(prepare_abs(n,x.data(),a.data()),"bench abs unsafe");}
                for(int row=0;row<rows;++row) {
                    const auto *wp=w.data()+size_t(row)*blocks;
                    if(!alt)total+=current_dot<T>(n,wp,x.data());
                    else if constexpr(BIAS){float got;xeon_iq3_rhs_dot_v1(n,&got,wp,x.data(),bias.data());total+=got;}
                    else total+=abs_dot<T>(n,wp,x.data(),a.data());
                }
            }
            const double cpu_end=thread_ns();
            const auto end=std::chrono::steady_clock::now();sink=total;
            local[arm]=std::chrono::duration<double,std::nano>(end-start).count()/(double(rows)*traversals);
            cpu_local[arm]=(cpu_end-cpu_start)/(double(rows)*traversals);
            if(repeat>0){(alt?candidate:base).push_back(local[arm]);(alt?cpu_candidate:cpu_base).push_back(cpu_local[arm]);}
        }
        if(repeat>0)ratios.push_back(repeat%2?(local[1]+local[2])/(local[0]+local[3]):(local[0]+local[3])/(local[1]+local[2]));
        if(repeat>0)cpu_ratios.push_back(repeat%2?(cpu_local[1]+cpu_local[2])/(cpu_local[0]+cpu_local[3]):(cpu_local[0]+cpu_local[3])/(cpu_local[1]+cpu_local[2]));
    }
    std::printf("{\"kind\":\"microbench\",\"type\":%d,\"variant\":\"%s\",\"n\":%d,\"rows\":%d,\"weight_payload_bytes\":%zu,\"sidecar_bytes\":%zu,\"traversals_per_arm\":%d,\"preparation_once_per_matrix_included\":true,\"current_ns_per_dot\":",T,BIAS?"stored_iq_rhs_bias62":"experimental_input_abs",n,rows,w.size()*sizeof(weight_type<T>),a.size()*sizeof(abs_rhs),traversals);
    array(base);std::printf(",\"candidate_ns_per_dot\":");array(candidate);std::printf(",\"paired_speedups\":");array(ratios);
    std::printf(",\"paired_median_speedup\":%.6f,\"current_thread_cpu_ns_per_dot\":",median(ratios));array(cpu_base);
    std::printf(",\"candidate_thread_cpu_ns_per_dot\":");array(cpu_candidate);
    std::printf(",\"thread_cpu_paired_speedups\":");array(cpu_ratios);
    std::printf(",\"thread_cpu_paired_median_speedup\":%.6f}\n",median(cpu_ratios));
}
int main(int argc,char **argv)try {
    for(int h=0;h<65536;++h)ggml_table_f32_f16[h]=GGML_COMPUTE_FP16_TO_FP32(h);
    std::setvbuf(stdout,nullptr,_IOLBF,0);const unsigned original=_mm_getcsr();
    if(argc==1||!std::strcmp(argv[1],"test")) {
        for(unsigned flush:{0u,0x8040u})for(unsigned rounding:{0u,0x2000u,0x4000u,0x6000u}) {
            _mm_setcsr((original&~0xe040u)|flush|rounding);test<1>();test<2>();test<3>();
        }
        _mm_setcsr(original);
    } else if(!std::strcmp(argv[1],"bench")) {
        for(int n:{2304,5120})for(int rows:{256,8192}) {
            bench<1>(n,rows);bench<2>(n,rows);bench<3>(n,rows);bench<1,true>(n,rows);
        }
    } else if(!std::strcmp(argv[1],"bench_focused")) {
        bench<1>(5120,8192);bench<2>(5120,8192);bench<3>(5120,8192);
    } else throw std::runtime_error("mode must be test, bench, or bench_focused");
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
