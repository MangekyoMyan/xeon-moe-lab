#include "tiled-kernel.h"
#include <immintrin.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static inline void reference(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                             const uint8_t signs[4], uint8_t * dst) {
    const uint64_t g[4] = {g0, g1, g2, g3};
    for (int l = 0; l < 4; ++l) {
        const uint8_t * v = (const uint8_t *) &g[l];
        for (int j = 0; j < 8; ++j) {
            dst[8*l+j] = (signs[l] & (1 << j)) ? uint8_t(128-v[j]) : uint8_t(128+v[j]);
        }
    }
}

struct input { uint64_t g[4]; uint8_t s[4]; };
int main() {
    std::mt19937_64 rng(42);
    std::vector<input> values(262144);
    std::vector<std::array<uint8_t,32>> outputs(values.size());
    for (auto & v : values) {
        for (auto & g : v.g) g = rng();
        for (auto & s : v.s) s = uint8_t(rng());
    }
    for (const auto & v : values) {
        std::array<uint8_t,32> original, actual;
        reference(v.g[0],v.g[1],v.g[2],v.g[3],v.s,original.data());
        tiled_unpk_sign32(v.g[0],v.g[1],v.g[2],v.g[3],v.s,actual.data());
        if (original != actual) return 1;
        for (int l=0;l<4;++l) for (int j=0;j<8;++j) {
            const int magnitude = int((v.g[l] >> (8*j)) & 255);
            const uint8_t expected = uint8_t(128 + ((v.s[l] & (1<<j)) ? -magnitude : magnitude));
            if (actual[8*l+j] != expected) return 2;
        }
    }
    printf("Exact byte agreement: %zu random cases, independent scalar oracle passed\n",values.size());
    for (int trial=0;trial<8;++trial) {
        const auto start=std::chrono::steady_clock::now();
        for (int repeat=0;repeat<16;++repeat) {
            for (size_t i=0;i<values.size();++i) {
                const auto & v=values[i];
                if (trial % 2) tiled_unpk_sign32(v.g[0],v.g[1],v.g[2],v.g[3],v.s,outputs[i].data());
                else reference(v.g[0],v.g[1],v.g[2],v.g[3],v.s,outputs[i].data());
            }
            asm volatile("" : : "g"(outputs.data()) : "memory");
        }
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        printf("implementation=%s seconds=%.6f checksum=%u\n",trial%2?"ssse3":"existing",elapsed,unsigned(outputs[trial][trial]));
    }
}
