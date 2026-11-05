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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct alignas(16) expanded_block {
    int8_t q[256];
    float d;
    uint8_t scale[8];
};
static_assert(sizeof(expanded_block)==272,"expanded stride");

static void expand(const block_iq3_xxs *src,expanded_block *dst,size_t blocks) {
    for(size_t b=0;b<blocks;++b) {
        dst[b].d=ggml_fp16_to_fp32(src[b].d);
        for(int g=0;g<8;++g) {
            uint32_t packed;std::memcpy(&packed,src[b].qs+64+4*g,4);
            dst[b].scale[g]=2*(packed>>28)+1;
            for(int j=0;j<32;++j) {
                const uint32_t grid=iq3xxs_grid[src[b].qs[8*g+j/4]];
                const int magnitude=(grid>>(8*(j%4)))&255;
                const uint8_t signs=ksigns_iq2xs[(packed>>(7*(j/8)))&127];
                dst[b].q[32*g+j]=(signs&(1<<(j%8))) ? -magnitude:magnitude;
            }
        }
    }
}

static inline __m128i partial(const int8_t *w,const int8_t *x,__m128i scale) {
    const __m128i q=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w));
    const __m128i y=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
    return _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q),_mm_sign_epi8(y,q)),scale);
}

__attribute__((noinline)) static float packed_dot(int n,const expanded_block *w,const block_q8_K *x) {
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i s0=_mm_setzero_si128(),s1=s0,s2=s0,s3=s0;
        for(int g=0;g<8;g+=2) {
            const __m128i a=_mm_set1_epi16(w[b].scale[g]);
            const __m128i z=_mm_set1_epi16(w[b].scale[g+1]);
            s0=_mm_add_epi32(s0,partial(w[b].q+32*g,x[b].qs+32*g,a));
            s1=_mm_add_epi32(s1,partial(w[b].q+32*g+16,x[b].qs+32*g+16,a));
            s2=_mm_add_epi32(s2,partial(w[b].q+32*g+32,x[b].qs+32*g+32,z));
            s3=_mm_add_epi32(s3,partial(w[b].q+32*g+48,x[b].qs+32*g+48,z));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_add_epi32(s0,s2)),_mm_add_epi32(s1,s3),1);
        const float d=w[b].d*x[b].d;
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(ints)));
    }
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(acc,1),_mm256_castps256_ps128(acc));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    return .25f*_mm_cvtss_f32(sum);
}

static double now_ms() {return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}

static void correctness() {
    std::mt19937 rng(123);size_t outputs=0;
    for(int k:{256,512,768,2304,5120}) for(int trial=0;trial<512;++trial) {
        const int nb=k/256;std::vector<block_iq3_xxs>w(nb);std::vector<expanded_block>p(nb);std::vector<block_q8_K>x(nb);
        for(int b=0;b<nb;++b) {
            uint16_t d;
            do {d=uint16_t(rng());} while((d&0x7c00)==0x7c00);
            w[b].d=d;for(auto &q:w[b].qs)q=rng();
            x[b].d=float(int(rng()%2001)-1000)/1024;
            for(auto &q:x[b].qs)q=int(rng()%256)-128;
        }
        expand(w.data(),p.data(),nb);float ref;
        ggml_vec_dot_iq3_xxs_q8_K(k,&ref,0,w.data(),0,x.data(),0,1);
        const float got=packed_dot(k,p.data(),x.data());
        if(std::memcmp(&ref,&got,4)!=0)throw std::runtime_error("random bit mismatch");
        ++outputs;
    }
    std::printf("{\"kind\":\"correctness\",\"random_outputs\":%zu,\"bit_mismatches\":0}\n",outputs);
}

static void inventory(int argc,char **argv) {
    std::map<std::string,size_t> bytes;size_t iq_blocks=0;int iq_tensors=0;
    for(int file=2;file<argc;++file) {
        auto g=gguf_init_from_file(argv[file],{true,nullptr});if(!g)throw std::runtime_error("GGUF metadata");
        for(int64_t i=0;i<gguf_get_n_tensors(g);++i) {
            const std::string name=gguf_get_tensor_name(g,i);
            if(name.rfind("blk.",0)!=0 || name.find("_exps.weight")==std::string::npos)continue;
            const int layer=std::stoi(name.substr(4));if(layer>=37)continue;
            const auto type=gguf_get_tensor_type(g,i);const auto ne=gguf_get_tensor_ne(g,i);
            const size_t size=ggml_row_size(type,ne[0])*ne[1]*ne[2]*ne[3];bytes[ggml_type_name(type)]+=size;
            if(type==GGML_TYPE_IQ3_XXS){iq_blocks+=size/sizeof(block_iq3_xxs);++iq_tensors;}
        }
        gguf_free(g);
    }
    std::printf("{\"kind\":\"inventory\",\"IQ3_XXS_CPU_tensors\":%d,\"expanded_bytes\":%zu,\"original_bytes\":%zu,\"types_bytes\":{",iq_tensors,iq_blocks*sizeof(expanded_block),iq_blocks*sizeof(block_iq3_xxs));
    bool first=true;for(const auto &v:bytes){std::printf("%s\"%s\":%zu",first?"":",",v.first.c_str(),v.second);first=false;}std::puts("}}");
}

int main(int argc,char **argv) try {
    auto backend=ggml_backend_cpu_init();if(!backend)return 1;
    if(argc>=3 && std::string(argv[1])=="inventory"){inventory(argc,argv);ggml_backend_free(backend);return 0;}
    correctness();
    if(argc!=3)throw std::runtime_error("usage: shard tensor OR inventory shards...");
    auto g=gguf_init_from_file(argv[1],{true,nullptr});if(!g)throw std::runtime_error("GGUF metadata");
    const int64_t ti=gguf_find_tensor(g,argv[2]);if(ti<0||gguf_get_tensor_type(g,ti)!=GGML_TYPE_IQ3_XXS)throw std::runtime_error("IQ3 tensor missing");
    const auto ne=gguf_get_tensor_ne(g,ti);const int k=ne[0],m=ne[1],nb=k/256;
    if((k!=5120&&k!=2304)||m<1||m>5120||ne[2]<6)throw std::runtime_error("unexpected shape");
    const size_t rows=size_t(m)*6,blocks=rows*nb;
    std::vector<block_iq3_xxs>w(blocks);std::vector<expanded_block>p(blocks);
    std::ifstream file(argv[1],std::ios::binary);file.seekg(gguf_get_data_offset(g)+gguf_get_tensor_offset(g,ti));
    if(!file.read(reinterpret_cast<char *>(w.data()),w.size()*sizeof(w[0])))throw std::runtime_error("slice read");gguf_free(g);
    const double prep=now_ms();expand(w.data(),p.data(),blocks);const double expand_ms=now_ms()-prep;
    std::mt19937 rng(77);std::uniform_real_distribution<float>d(-.5f,.5f);
    std::vector<float>f(k*6);for(auto &v:f)v=d(rng);std::vector<block_q8_K>x(nb*6);
    for(int e=0;e<6;++e)quantize_row_q8_K(f.data()+e*k,x.data()+e*nb,k);
    std::vector<float>a(rows),b(rows);
    for(int threads:{1,6,12}) {
        omp_set_dynamic(0);omp_set_num_threads(threads);
        int pin_errors=0;
        #pragma omp parallel reduction(+:pin_errors)
        {
            const int id=omp_get_thread_num();const int cpu=threads==6 ? (id<3?id:id+3):id;
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);pin_errors+=sched_setaffinity(0,sizeof(mask),&mask)!=0;
        }
        if(pin_errors)throw std::runtime_error("worker affinity");
        auto run=[&](bool expanded) {
            const double start=now_ms();
            #pragma omp parallel for schedule(static)
            for(size_t row=0;row<rows;++row) {
                const auto input=x.data()+(row/m)*nb;
                if(expanded)b[row]=packed_dot(k,p.data()+row*nb,input);
                else ggml_vec_dot_iq3_xxs_q8_K(k,&a[row],0,w.data()+row*nb,0,input,0,1);
            }
            return now_ms()-start;
        };
        run(false);run(true);
        if(std::memcmp(a.data(),b.data(),rows*sizeof(float))!=0)throw std::runtime_error("realweight bit mismatch");
        std::vector<double>plain,expanded;
        for(int repeat=0;repeat<7;++repeat)for(bool choice:{false,true,true,false}) {
            const double elapsed=run(choice);(choice?expanded:plain).push_back(elapsed);
        }
        std::printf("{\"kind\":\"timing\",\"k\":%d,\"m\":%d,\"experts\":6,\"threads\":%d,\"original_bytes\":%zu,\"expanded_bytes\":%zu,\"expand_ms\":%.6f,\"stock_ms\":%.6f,\"expanded_ms\":%.6f,\"speedup\":%.6f,\"real_outputs\":%zu,\"bit_mismatches\":0}\n",
          k,m,threads,w.size()*sizeof(w[0]),p.size()*sizeof(p[0]),expand_ms,median(plain),median(expanded),median(plain)/median(expanded),rows);std::fflush(stdout);
    }
    ggml_backend_free(backend);return 0;
} catch(const std::exception &e){std::fprintf(stderr,"probe failed: %s\n",e.what());return 1;}
