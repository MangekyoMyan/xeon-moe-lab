#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <omp.h>
#include "compact-kernel.inc"

struct alignas(16) rhs_correction { int16_t pairs[16][8]; };
static_assert(sizeof(rhs_correction) == 256, "correction size");
static size_t checked = 0;
static void need(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

static bool prepare(int k, const block_q8_K * x, rhs_correction * out) {
    bool safe = true;
    const __m128i bias = _mm_set1_epi8(62);
    const __m128i minimum = _mm_set1_epi8(-128);
    for (int b = 0; b < k/256; ++b) {
        safe = safe && std::isfinite(x[b].d);
        for (int h = 0; h < 16; ++h) {
            const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 16*h));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(values, minimum))) safe = false;
            _mm_store_si128(reinterpret_cast<__m128i *>(out[b].pairs[h]), _mm_maddubs_epi16(bias, values));
        }
    }
    return safe;
}

__attribute__((noinline)) static float prepared_dot(int n, const compact_block * w, const block_q8_K * x, const rhs_correction * correction) {
    const __m128i lut = _mm_setr_epi8(66,74,82,90,98,106,114,124,58,50,42,34,26,18,10,0);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n/256; ++b) {
        __m128i low = _mm_setzero_si128(), high = low;
        for (int g = 0; g < 8; ++g) {
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs + 16*g));
            const __m128i q0 = _mm_shuffle_epi8(lut, _mm_and_si128(bits, mask));
            const __m128i q1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(bits, 4), mask));
            const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g));
            const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g + 16));
            const __m128i c0 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g]));
            const __m128i c1 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g+1]));
            const __m128i p0 = _mm_sub_epi16(_mm_maddubs_epi16(q0, x0), c0);
            const __m128i p1 = _mm_sub_epi16(_mm_maddubs_epi16(q1, x1), c1);
            const __m128i scale = _mm_set1_epi16(w[b].scale[g]);
            low = _mm_add_epi32(low, _mm_madd_epi16(p0, scale));
            high = _mm_add_epi32(high, _mm_madd_epi16(p1, scale));
        }
        const __m256i ints = _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d), _mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}

#include "variants.inc"

static double now() { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

static void run(int k, int rows_per_expert, int experts, int threads, bool edge, bool timing) {
    const size_t blocks = k/256, rows = size_t(rows_per_expert)*experts;
    std::mt19937 rng(6511 + k + rows_per_expert);
    std::vector<compact_block> weights(rows*blocks);
    std::vector<block_q8_K> input(experts*blocks);
    std::vector<rhs_correction> correction(input.size());
    std::vector<uint8_t> safe(experts);
    std::vector<float> reference(rows), output(rows);
    for (auto & w : weights) {
        for (auto & q : w.qs) q = rng();
        for (auto & s : w.scale) s = 2*(rng()%16) + 1;
        w.d = float(int(rng()%1024)-512)/4096;
    }
    for (auto & x : input) {
        x.d = float(int(rng()%1024)-512)/4096;
        for (auto & q : x.qs) q = int(rng()%255)-127;
    }
    if (edge) for (int e = 0; e < experts; e += 2) input[e*blocks].qs[e%256] = -128;
    auto compute = [&](int mode, std::vector<float> & out) {
        if (mode) for (int e = 0; e < experts; ++e) safe[e] = prepare(k, input.data()+e*blocks, correction.data()+e*blocks);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t row = 0; row < rows; ++row) {
            const size_t e = row/rows_per_expert;
            const auto * w = weights.data()+row*blocks;
            const auto * x = input.data()+e*blocks;
            const auto * c = correction.data()+e*blocks;
            if (!mode || !safe[e]) out[row] = compact_dot(k, w, x);
            else if (mode == 1) out[row] = prepared_dot(k, w, x, c);
            else if (mode == 2) out[row] = prepared_dot_u1(k, w, x, c);
            else if (mode == 3) out[row] = prepared_dot_u2(k, w, x, c);
            else out[row] = prepared_dot_u4(k, w, x, c);
        }
    };
    compute(0, reference);
    for (int mode = 1; mode <= 4; ++mode) {
        compute(mode, output);
        need(std::memcmp(reference.data(), output.data(), rows*4)==0, "prepared bit mismatch");
        for (int e = 0; e < experts; ++e) need(bool(safe[e]) == !(edge && e%2==0), "fallback detection");
        checked += rows;
    }
    if (!timing) return;
    for (int mode = 1; mode <= 4; ++mode) {
        std::vector<double> a, b;
        for (int rep = 0; rep < 6; ++rep) for (int arm : {0, mode, mode, 0}) {
            const double start = now(); compute(arm, output); const double elapsed = now()-start;
            need(std::memcmp(reference.data(), output.data(), rows*4)==0, "timed prepared mismatch");
            checked += rows; (arm ? b : a).push_back(elapsed);
        }
        std::printf("{\"k\":%d,\"rows_per_expert\":%d,\"experts\":%d,\"threads\":%d,\"mode\":%d,\"base_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.6f,\"scratch_bytes\":%zu}\n",
            k,rows_per_expert,experts,threads,mode,median(a),median(b),median(a)/median(b),correction.size()*sizeof(rhs_correction));
        std::fflush(stdout);
    }
}

int main(int argc,char **) try {
    for (int k : {256,512,2304,5120}) for (int rows : {1,7,129}) for (bool edge : {false,true}) run(k,rows,3,1,edge,false);
    if (argc > 1) for (int threads : {1,12}) for (int experts : {6,24}) for (int k : {2304,5120}) run(k,k==2304?5120:2304,experts,threads,false,true);
    std::printf("{\"pass\":true,\"outputs_checked\":%zu}\n",checked);
} catch(const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
