#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "compact-kernel.inc"
#include "iq-small-kernels.inc"

template<int T> struct alignas(16) candidate_block {
    uint8_t qs[128]; float d; uint8_t scales[8];
    using correction_type = typename std::conditional<T==2,int32_t,int16_t>::type;
    correction_type correction[8];
};
static_assert(sizeof(candidate_block<1>) == 160);
static_assert(sizeof(candidate_block<2>) == 176);
static_assert(sizeof(candidate_block<3>) == 160);
struct unsigned_activation { float d; uint8_t qs[256]; };

template<int T> static int value(int code) {
    static const int xxs[16]={4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62};
    static const int two[16]={8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0};
    static const int three[16]={1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15};
    return T==1?xxs[code]:T==2?two[code]:three[code];
}
template<int T> static int scale(const uint8_t *s,int g,int h) {
    if constexpr(T==1) return s[g];
    if constexpr(T==2) return 2*((s[g]>>(4*h))&15)+1;
    return 2*((s[g/2]>>(4*(g%2)))&15)+1;
}
template<int T> static void convert(const void *src,candidate_block<T> &dst) {
    const auto *s=static_cast<const uint8_t *>(src);
    std::memcpy(dst.qs,s,128);std::memcpy(&dst.d,s+128,4);std::memcpy(dst.scales,s+132,8);
    for(int h=0;h<2;++h)for(int lane=0;lane<4;++lane) {
        int sum=0;
        for(int g=0;g<8;++g)for(int k=0;k<4;++k)
            sum+=value<T>((dst.qs[16*g+4*lane+k]>>(4*h))&15)*scale<T>(dst.scales,g,h);
        if constexpr(T==1) { if(sum%2)throw std::runtime_error("odd XXS correction");sum/=2; }
        using C=typename candidate_block<T>::correction_type;
        if(sum<std::numeric_limits<C>::min()||sum>std::numeric_limits<C>::max())throw std::runtime_error("correction overflow");
        dst.correction[h*4+lane]=sum;
    }
}
static void convert_activation(const block_q8_K *src,unsigned_activation *dst,int n) {
    for(int b=0;b<n/256;++b){dst[b].d=src[b].d;for(int i=0;i<256;++i)dst[b].qs[i]=int(src[b].qs[i])+128;}
}
template<int T> __attribute__((noinline)) static float candidate_dot(int n,const candidate_block<T> *w,const unsigned_activation *x) {
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
            const __m128i x0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g));
            const __m128i x1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16));
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(x0,q0),_mm_set1_epi16(scale<T>(w[b].scales,g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(x1,q1),_mm_set1_epi16(scale<T>(w[b].scales,g,1))));
        }
        __m128i c0,c1;
        if constexpr(T==2) {
            c0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].correction));
            c1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].correction+4));
        } else {
            const __m128i c=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].correction));
            c0=_mm_cvtepi16_epi32(c);c1=_mm_cvtepi16_epi32(_mm_srli_si128(c,8));
        }
        constexpr int shift=T==1?8:7;
        low=_mm_sub_epi32(low,_mm_slli_epi32(c0,shift));high=_mm_sub_epi32(high,_mm_slli_epi32(c1,shift));
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return (T==1?.25f:T==2?.125f:1.f)*iq_hsum(acc);
}
template<int T> using source_type=typename std::conditional<T==1,block_iq3_xxs,typename std::conditional<T==2,block_iq2_s,block_iq3_s>::type>::type;
template<int T> using accepted_type=typename std::conditional<T==1,compact_block,iq_compact_block>::type;
template<int T> static void make(std::mt19937 &rng,accepted_type<T> *a,candidate_block<T> *b,int blocks,source_type<T> *source=nullptr) {
    for(int i=0;i<blocks;++i) {
        source_type<T> s;auto *raw=reinterpret_cast<uint8_t *>(&s);for(size_t j=0;j<sizeof(s);++j)raw[j]=rng();
        s.d=uint16_t((rng()%0x7c00)|(rng()%2?0x8000:0));
        if(source)source[i]=s;
        if constexpr(T==1){expanded_block e;expand(&s,&e,1);compact(e,a[i]);}
        if constexpr(T==2)iq_pack<true>(s,a[i]);
        if constexpr(T==3)iq_pack<false>(s,a[i]);
        convert<T>(&a[i],b[i]);
    }
}
template<int T> static float original_dot(int n,const source_type<T> *w,const block_q8_K *x) {
    float s;
    if constexpr(T==1)ggml_vec_dot_iq3_xxs_q8_K(n,&s,0,w,0,x,0,1);
    if constexpr(T==2)ggml_vec_dot_iq2_s_q8_K(n,&s,0,w,0,x,0,1);
    if constexpr(T==3)ggml_vec_dot_iq3_s_q8_K(n,&s,0,w,0,x,0,1);
    return s;
}
template<int T> static float baseline(int n,const accepted_type<T> *w,const block_q8_K *x){
    if constexpr(T==1)return compact_dot(n,w,x);
    if constexpr(T==2)return iq_compact_dot<true>(n,w,x);
    if constexpr(T==3)return iq_compact_dot<false>(n,w,x);
}
template<int T> static double reference(int n,const candidate_block<T> *w,const block_q8_K *x){
    double sum=0;
    for(int b=0;b<n/256;++b)for(int g=0;g<8;++g)for(int h=0;h<2;++h)for(int j=0;j<16;++j)
        sum+=double(w[b].d)*double(x[b].d)*value<T>((w[b].qs[16*g+j]>>(4*h))&15)*scale<T>(w[b].scales,g,h)*x[b].qs[32*g+16*h+j];
    return sum*(T==1?.25:T==2?.125:1.);
}
template<int T> static void test(){
    std::mt19937 rng(271828+T);size_t cases=0,mismatch=0,original_mismatch=0,endpoints=0;double max_abs=0;
    for(int n:{256,512,768,2304,5120})for(int it=0;it<2048;++it){
        std::vector<accepted_type<T>>a(n/256);std::vector<candidate_block<T>>b(n/256);
        std::vector<source_type<T>>source(n/256);std::vector<block_q8_K>x(n/256);std::vector<unsigned_activation>u(n/256);make<T>(rng,a.data(),b.data(),n/256,source.data());
        const int exponents[]={-149,-145,-130,-126,-125,-24,-15,0,15};
        for(auto &z:x){z.d=std::ldexp(float(int(rng()%8191)-4095)/4096,exponents[rng()%9]);for(auto &q:z.qs)q=int(rng()%255)-127;}
        convert_activation(x.data(),u.data(),n);float aa=baseline<T>(n,a.data(),x.data()),bb=candidate_dot<T>(n,b.data(),u.data());
        if(std::memcmp(&aa,&bb,4))++mismatch;float oo=original_dot<T>(n,source.data(),x.data());if(std::memcmp(&aa,&oo,4))++original_mismatch;++cases;
        if(it<64){for(auto &z:x)for(auto &q:z.qs)q=-128;convert_activation(x.data(),u.data(),n);aa=baseline<T>(n,a.data(),x.data());bb=candidate_dot<T>(n,b.data(),u.data());if(aa!=bb)++endpoints;double ref=reference<T>(n,b.data(),x.data());max_abs=std::max(max_abs,std::abs(double(bb)-ref)/(1+std::abs(ref)));}
    }
    std::printf("{\"kind\":\"correctness\",\"type\":%d,\"cases\":%zu,\"bit_mismatch_q8_127\":%zu,\"original_dot_mismatch_q8_127\":%zu,\"endpoint_minus128_different_from_baseline\":%zu,\"endpoint_candidate_max_scaled_abs_error_vs_double\":%.9g,\"accepted_bytes\":%zu,\"candidate_bytes\":%zu}\n",T,cases,mismatch,original_mismatch,endpoints,max_abs,sizeof(accepted_type<T>),sizeof(candidate_block<T>));
    if(mismatch||original_mismatch)throw std::runtime_error("bit mismatch");
}
static volatile double sink=0;
template<int T> static void bench(int n,int rows){
    std::mt19937 rng(314159+T);int blocks=n/256;
    std::vector<accepted_type<T>>a(rows*blocks);std::vector<candidate_block<T>>b(rows*blocks);make<T>(rng,a.data(),b.data(),rows*blocks);
    std::vector<block_q8_K>x(blocks);std::vector<unsigned_activation>u(blocks);for(auto &z:x){z.d=.001f;for(auto &q:z.qs)q=int(rng()%255)-127;}convert_activation(x.data(),u.data(),n);
    std::vector<double> at,bt,ratios;const int traversals=std::max(1,262144/rows);
    for(int repeat=0;repeat<9;++repeat){double local[4];for(int j=0;j<4;++j){bool alt=(j==1||j==2);if(repeat%2)alt=!alt;double s=0;auto begin=std::chrono::steady_clock::now();for(int traversal=0;traversal<traversals;++traversal)for(int r=0;r<rows;++r)s+=alt?candidate_dot<T>(n,b.data()+r*blocks,u.data()):baseline<T>(n,a.data()+r*blocks,x.data());auto end=std::chrono::steady_clock::now();sink=s;double ns=std::chrono::duration<double,std::nano>(end-begin).count()/(double(rows)*traversals);if(repeat>0)(alt?bt:at).push_back(ns);local[j]=ns;}
        if(repeat>0)ratios.push_back(repeat%2?(local[1]+local[2])/(local[0]+local[3]):(local[0]+local[3])/(local[1]+local[2]));}
    auto output_array=[](const std::vector<double>&v){std::printf("[");for(size_t i=0;i<v.size();++i)std::printf("%s%.9f",i?",":"",v[i]);std::printf("]");};
    std::printf("{\"kind\":\"benchmark_raw\",\"type\":%d,\"n\":%d,\"rows\":%d,\"traversals_per_segment\":%d,\"accepted_ns_per_dot\":",T,n,rows,traversals);output_array(at);std::printf(",\"candidate_ns_per_dot\":");output_array(bt);std::printf(",\"paired_speedups\":");output_array(ratios);std::printf("}\n");
    std::sort(at.begin(),at.end());std::sort(bt.begin(),bt.end());std::sort(ratios.begin(),ratios.end());
    std::printf("{\"kind\":\"benchmark\",\"type\":%d,\"n\":%d,\"rows\":%d,\"accepted_bytes\":%zu,\"candidate_bytes\":%zu,\"accepted_ns_per_dot\":%.3f,\"candidate_ns_per_dot\":%.3f,\"paired_speedup_median\":%.5f,\"paired_speedup_min\":%.5f,\"paired_speedup_max\":%.5f,\"activation_conversion_timed\":false}\n",T,n,rows,a.size()*sizeof(a[0]),b.size()*sizeof(b[0]),at[at.size()/2],bt[bt.size()/2],ratios[ratios.size()/2],ratios.front(),ratios.back());
}
int main(int argc,char **argv)try{
    ggml_cpu_init();
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    const unsigned original_mxcsr=_mm_getcsr();
    if(argc==1||std::strcmp(argv[1],"test")==0){for(unsigned flush:{0u,0x8040u}){
        _mm_setcsr((original_mxcsr&~0x8040u)|flush);
        std::printf("{\"kind\":\"floating_point_environment\",\"mxcsr\":%u,\"ftz\":%u,\"daz\":%u,\"rounding_mode\":%u}\n",_mm_getcsr(),(_mm_getcsr()>>15)&1,(_mm_getcsr()>>6)&1,(_mm_getcsr()>>13)&3);
        test<1>();test<2>();test<3>();
    }_mm_setcsr(original_mxcsr);}
    if(argc>1&&std::strcmp(argv[1],"bench")==0){for(int n:{2304,5120})for(int rows:{256,32768}){bench<1>(n,rows);bench<2>(n,rows);bench<3>(n,rows);}}
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
