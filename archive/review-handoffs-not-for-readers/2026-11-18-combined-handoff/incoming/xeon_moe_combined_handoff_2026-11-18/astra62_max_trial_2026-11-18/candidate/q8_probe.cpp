#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void quantize_row_q8_K_ref(const float *, block_q8_K *, int64_t);

struct Rng {
    uint32_t state;
    uint32_t get() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return state;
    }
};
static float as_float(uint32_t bits) {
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
static void fill(float * x, int pattern, int n, Rng & rng) {
    float scale = std::ldexp(1.f, int(rng.get()%221)-110);
    for (int i=0; i<n; ++i) {
        int j = i%256;
        float f = (int(rng.get()%32767) - 16383)/16384.f;
        switch (pattern) {
            case 0: f *= scale; break;
            case 1: f = (rng.get()&1) ? 0.f : -0.f; break;
            case 2: f = (j == 15 ? 1.f : j == 16 ? -1.f : f/2)*scale; break;
            case 3: f = (j == 191 ? -1.f : j == 192 ? 1.f : f/2)*scale; break;
            case 4: f = (j < 255 ? (j - 127.5f) : -127.f); if (j == 0) f = 127.f; break;
            case 5: {
                const float mid = float(int(rng.get()%254)-127) + .5f;
                f = j == 0 ? -127.f : std::nextafter(mid, rng.get()&1 ? 1000.f : -1000.f);
                break;
            }
            case 6: f = as_float((rng.get()&0x807fffffu) | ((rng.get()%255) << 23)); break;
            case 7: f = (j==92 ? std::numeric_limits<float>::max() : j == 91 ? -std::numeric_limits<float>::max() : f); break;
            case 8: f = as_float(rng.get()&0x807fffffu); break;
            case 9: f = (j%37==1 ? std::numeric_limits<float>::quiet_NaN() : f); break;
            case 10: f = (j%37==1 ? std::numeric_limits<float>::infinity() : f); break;
            case 11: f = (j%41==2 ? -std::numeric_limits<float>::infinity() : f); break;
            default: throw std::runtime_error("pattern");
        }
        x[i] = f;
    }
}
static void check() {
    const auto from = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    std::puts("round_mode,patterns,calls,blocks,unequal_bytes");
    for (int rm : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        if (std::fesetround(rm)) throw std::runtime_error("rounding mode");
        Rng rng{0x4a18c257u};
        size_t calls=0, blocks=0, different=0;
        for (int p=0; p<12; ++p) for (int rep=0; rep<128; ++rep) {
            const int nb = std::array<int,6>{0,1,2,3,7,28}[rep%6];
            const int n = nb*256;
            std::vector<float> storage(n+4);
            float * x = storage.data() + 1 + rep%3;
            fill(x,p,n,rng);
            std::vector<block_q8_K> ref(nb+2), actual(nb+2);
            std::memset(ref.data(), uint8_t(13+rep), ref.size()*sizeof(block_q8_K));
            std::memcpy(actual.data(), ref.data(), ref.size()*sizeof(block_q8_K));
            quantize_row_q8_K_ref(x,ref.data()+1,n);
            from(x,actual.data()+1,n);
            auto * a = reinterpret_cast<const uint8_t *>(actual.data());
            auto * r = reinterpret_cast<const uint8_t *>(ref.data());
            for (size_t j=0;j<ref.size()*sizeof(block_q8_K);++j) {
                if (a[j]!=r[j]) {
                    if (different++ == 0) std::fprintf(stderr,"mode=%d pattern=%d rep=%d byte=%zu %u != %u\n",rm,p,rep,j,a[j],r[j]);
                }
            }
            ++calls; blocks+=nb;
        }
        std::printf("%d,12,%zu,%zu,%zu\n",rm,calls,blocks,different);
        if(different) throw std::runtime_error("reference mismatch");
    }
    std::fesetround(FE_TONEAREST);
}
static void bench(int k, int rows, int loops, int samples, uint32_t seed) {
    if(k<=0 || k%256 || rows<=0 || loops<=0 || samples<=0) throw std::runtime_error("invalid shape");
    Rng rng{seed};
    std::vector<float> x(k*rows);
    for(int r=0;r<rows;++r) fill(x.data()+r*k,r%5 == 0 ? 2 : 0,k,rng);
    std::vector<block_q8_K> out(k*rows/256);
    auto from = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    auto run=[&]() {
        for(int r=0;r<rows;++r) from(x.data()+r*k,out.data()+r*(k/256),k);
    };
    for(int n=0;n<3;++n) run();
    for(int s=0;s<samples;++s) {
        auto start=std::chrono::steady_clock::now();
        for(int n=0;n<loops;++n) run();
        auto end=std::chrono::steady_clock::now();
        uint64_t h=UINT64_C(14695981039346656037);
        const auto * bytes = reinterpret_cast<const uint8_t *>(out.data());
        for(size_t i=0;i<out.size()*sizeof(block_q8_K);++i) h=(h^bytes[i])*UINT64_C(1099511628211);
        std::printf("%d,%.1f,0,%016llx\n",s,std::chrono::duration<double,std::nano>(end-start).count()/loops,(unsigned long long)h);
    }
}
int main(int argc, char **argv) {
    try {
        ggml_cpu_init();
        if(argc==2 && std::string(argv[1])=="check") check();
        else if(argc==7 && std::string(argv[1])=="bench") bench(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoul(argv[6]));
        else throw std::runtime_error("check or bench K ROWS LOOPS SAMPLES SEED");
    } catch(const std::exception & e) {
        std::fprintf(stderr,"q8_probe: %s\n",e.what()); return 1;
    }
}
