// Data-only contract probe. Links newly compiled source, not packaged ELF files.
#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "mx-rhs.h"
#include <immintrin.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

float ggml_table_f32_e8m0_half[256];
float ggml_table_f32_f16[65536];
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    std::fprintf(stderr,"GGML assertion %s:%d ",file,line);
    va_list args;va_start(args,fmt);std::vfprintf(stderr,fmt,args);va_end(args);
    std::abort();
}

static void need(bool ok,const char * message) {if(!ok)throw std::runtime_error(message);}
static unsigned bits(float x){unsigned y;std::memcpy(&y,&x,4);return y;}
static size_t checked=0,unsafe=0,partition_checks=0,shape_checks=0;
static void one(int n,std::vector<block_mxfp4>& w,std::vector<block_q8_0>& x,bool expected_safe) {
    const size_t stride=xeon_mxfp4_rhs_stride(n);
    auto * allocation=static_cast<unsigned char *>(std::aligned_alloc(64,stride+128));
    need(allocation!=nullptr,"allocation");
    std::memset(allocation,0xa5,stride+128);
    void * sidecar=allocation+64;
    float reference;ggml_vec_dot_mxfp4_q8_0(n,&reference,0,w.data(),0,x.data(),0,1);
    for(int nth:{1,3,12,24}) {
        bool safe=true;
        // Actual caller has workers write these disjoint block intervals, then a barrier.
        // This probe checks partition coverage and outputs; it is not a concurrent backend test.
        for(int ith=0;ith<nth;++ith) {
            safe = (xeon_mxfp4_rhs_prepare_v1(n,x.data(),ith*(n/32)/nth,
                    (ith+1)*(n/32)/nth,sidecar)!=0) && safe;
        }
        need(safe==expected_safe,"unexpected safe/fallback classification");
        if(safe) {
            float got;xeon_mxfp4_rhs_dot_v1(n,&got,w.data(),x.data(),sidecar);
            if(bits(got)!=bits(reference)) {
                std::fprintf(stderr,"mismatch n=%d nth=%d mxcsr=%08x ref=%08x got=%08x\n",n,nth,_mm_getcsr(),bits(reference),bits(got));
                throw std::runtime_error("bit mismatch");
            }
            ++checked;
        } else ++unsafe;
        for(size_t i=0;i<64;++i)need(allocation[i]==0xa5&&allocation[64+stride+i]==0xa5,"sidecar canary");
        ++partition_checks;
    }
    std::free(allocation);
}

int main()try {
    for(int e=0;e<256;++e)ggml_table_f32_e8m0_half[e]=GGML_E8M0_TO_FP32_HALF(e);
    for(int h=0;h<65536;++h)ggml_table_f32_f16[h]=GGML_COMPUTE_FP16_TO_FP32(h);
    std::mt19937 rng(20261116);
    const unsigned original=_mm_getcsr();
    for(unsigned flush:{0u,0x8040u})for(unsigned rounding:{0u,0x2000u,0x4000u,0x6000u}) {
        _mm_setcsr((original&~0xe040u)|flush|rounding);
        for(int n:{32,64,96,160,256,2304,5120})for(int iteration=0;iteration<128;++iteration) {
            std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
            for(size_t b=0;b<w.size();++b){
                w[b].e=uint8_t((iteration*2+b)%256);
                for(auto & q:w[b].qs)q=uint8_t(rng());
                x[b].d=uint16_t((rng()%0x7c00)|(rng()&0x8000));
                for(auto & q:x[b].qs)q=int(rng()%255)-127;
            }
            one(n,w,x,true);
        }
        for(int n:{32,64,96,2304,5120})for(int mode=0;mode<5;++mode) {
            std::vector<block_mxfp4>w(n/32);std::vector<block_q8_0>x(n/32);
            for(size_t b=0;b<w.size();++b){w[b].e=127;std::memset(w[b].qs,0x97,16);x[b].d=0x3c00;std::memset(x[b].qs,1,32);}
            if(mode==0)x.back().qs[31]=-128;
            if(mode==1)x.back().d=0x7c00;
            if(mode==2)x.back().d=0xfc00;
            if(mode==3)x.back().d=0x7e01;
            if(mode==4)x.back().d=0xfe01;
            one(n,w,x,false);
        }
    }
    _mm_setcsr(original);
    ggml_tensor w{},x{},ids{};w.type=GGML_TYPE_MXFP4;x.ne[0]=5120;x.ne[1]=1;x.ne[2]=x.ne[3]=1;ids.ne[1]=1;
    need(xeon_mxfp4_rhs_shape(&w,&x,&ids),"valid shape");++shape_checks;
    x.ne[1]=64;need(xeon_mxfp4_rhs_shape(&w,&x,&ids),"64 RHS shape");++shape_checks;
    x.ne[1]=65;need(!xeon_mxfp4_rhs_shape(&w,&x,&ids),"65 RHS reject");++shape_checks;
    x.ne[1]=1;ids.ne[1]=2;need(!xeon_mxfp4_rhs_shape(&w,&x,&ids),"T2 reject");++shape_checks;
    ids.ne[1]=1;x.ne[0]=33;need(!xeon_mxfp4_rhs_shape(&w,&x,&ids),"unaligned K reject");++shape_checks;
    std::printf("{\"scope\":\"Newly compiled current MXFP4 prepare/dot plus pinned CPU reference, synthetic data only; sequential partition coverage, not concurrent backend execution\",\"bit_exact_outputs\":%zu,\"unsafe_rejections\":%zu,\"partition_checks\":%zu,\"shape_checks\":%zu,\"rounding_modes\":4,\"ftz_daz_modes\":2,\"canaries_intact\":true,\"passed\":true}\n",checked,unsafe,partition_checks,shape_checks);
    return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
