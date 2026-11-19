#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "iq3-cache.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Shape {
    ggml_type type;
    int k, rows, experts, slots, tokens, activations, threads;
    bool cache, pad, raw, ref;
    int pattern; // 0 mixed, 1 same expert, 2 all skipped, 3 round-robin, 4 zeros
};
struct Rng {
    uint32_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static char * ptr(ggml_tensor * t, int row = 0, int plane = 0) {
    return static_cast<char *>(t->data) + row*t->nb[1] + plane*t->nb[2];
}
static int & route(ggml_tensor * ids, int slot, int token) {
    return *reinterpret_cast<int *>(static_cast<char *>(ids->data) + slot*ids->nb[0] + token*ids->nb[1]);
}
static ggml_type get_type(const std::string & name) {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_MXFP4, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        if (name == ggml_type_name(type)) return type;
    }
    throw std::runtime_error("unknown weight type");
}
static uint64_t hash_bytes(const void * data, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i])*UINT64_C(1099511628211);
    return hash;
}
static void fill_weights(ggml_tensor * w, Rng & rng) {
    if (w->type == GGML_TYPE_F32) {
        auto * values = static_cast<float *>(w->data);
        for (int64_t i = 0; i < ggml_nelements(w); ++i) values[i] = (int(rng.next()%511) - 255)/256.f;
        return;
    }
    auto * bytes = static_cast<uint8_t *>(w->data);
    size_t bs = ggml_type_size(w->type);
    for (size_t offset = 0; offset < ggml_nbytes(w); offset += bs) {
        for (size_t i = 0; i < bs; ++i) bytes[offset + i] = uint8_t(rng.next());
        if (w->type == GGML_TYPE_MXFP4) {
            bytes[offset] = 119 + rng.next()%7;
        } else {
            ggml_fp16_t d = ggml_fp32_to_fp16((int(rng.next()%31) - 15)/512.f);
            std::memcpy(bytes + offset, &d, sizeof(d));
        }
    }
}
static void dequant(ggml_type t, const void * x, float * y, int n) {
    if (t == GGML_TYPE_F32) std::memcpy(y, x, n*sizeof(float));
    else if (t == GGML_TYPE_Q8_K) {
        const auto * blocks = static_cast<const block_q8_K *>(x);
        for (int j = 0; j < n; ++j) y[j] = blocks[j/256].d * blocks[j/256].qs[j%256];
    } else {
        const auto convert = ggml_get_type_traits(t)->to_float;
        require(convert != nullptr, "dequantization unavailable");
        convert(x, y, n);
    }
}
struct Data {
    const Shape s;
    ggml_context * wc = nullptr, * ac = nullptr;
    ggml_backend_buffer_t wb = nullptr, ab = nullptr;
    ggml_tensor * w = nullptr, * x = nullptr, * ids = nullptr, * y = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_threadpool * tp = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> scratch;
    void * owner = nullptr;
    uint64_t epoch = 0;

    Data(Shape shape, uint32_t seed) : s(shape) {
        require(s.k > 0 && s.rows > 0 && s.experts > 0 && s.slots > 0 && s.tokens > 0 &&
                s.activations > 0 && s.slots%s.activations == 0 && s.threads > 0, "invalid shape");
        Rng rng{seed};
        wc = ggml_init({1024*1024, nullptr, true});
        ac = ggml_init({1024*1024, nullptr, true});
        const auto bt = ggml_backend_cpu_buffer_type();
        w = ggml_new_tensor_3d(wc, s.type, s.k, s.rows, s.experts);
        wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, bt);
        require(wb, "weight buffer");
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        fill_weights(w, rng);
        const ggml_type xt = s.raw ? ggml_get_type_traits_cpu(s.type)->vec_dot_type : GGML_TYPE_F32;
        if (s.pad) {
            ggml_tensor * xb = ggml_new_tensor_3d(ac, xt, s.k + 2*ggml_blck_size(xt), s.activations + 1, s.tokens);
            x = ggml_view_3d(ac, xb, s.k, s.activations, s.tokens, xb->nb[1], xb->nb[2], ggml_type_size(xt));
            auto * ib = ggml_new_tensor_2d(ac, GGML_TYPE_I32, s.tokens+2, s.slots);
            ids = ggml_transpose(ac, ggml_view_2d(ac, ib, s.tokens, s.slots, ib->nb[1], sizeof(int32_t)));
        } else {
            x = ggml_new_tensor_3d(ac, xt, s.k, s.activations, s.tokens);
            ids = ggml_new_tensor_2d(ac, GGML_TYPE_I32, s.slots, s.tokens);
        }
        y = ggml_mul_mat_id(ac, w, x, ids);
        g = ggml_new_graph_custom(ac, 32, false);
        ggml_build_forward_expand(g, y);
        ab = ggml_backend_alloc_ctx_tensors_from_buft(ac, bt);
        require(ab, "activation buffer");
        ggml_backend_buffer_clear(ab, 0xa5);
        std::vector<float> xf(s.k);
        for (int t = 0; t < s.tokens; ++t) {
            for (int a = 0; a < s.activations; ++a) {
                for (int j = 0; j < s.k; ++j) {
                    xf[j] = (int(rng.next()%2047) - 1023)/1024.f;
                    if (j%31 == 0) xf[j] = 0.f;
                    if (j%256 == 7) xf[j] = 2.f;
                    if (j%256 == 14) xf[j] = -2.f;
                    if (s.pattern == 4 || ((j/256 + a + t)%11 == 5)) xf[j] = 0.f;
                    if (s.pattern == 6) xf[j] = j%256 < 3 ? .1f : (j%256 == 17 ? -.05f : 0.f);
                }
                if (xt == GGML_TYPE_F32) {
                    std::memcpy(ptr(x, a, t), xf.data(), s.k*sizeof(float));
                } else {
                    ggml_get_type_traits_cpu(xt)->from_float(xf.data(), ptr(x, a, t), s.k);
                    if (xt == GGML_TYPE_Q8_K) {
                        auto * blocks = reinterpret_cast<block_q8_K *>(ptr(x,a,t));
                        for (int b = 0; b < s.k/256; ++b) {
                            blocks[b].qs[(a + t + b)%256] = -128;
                            for (int g = 0; g < 16; ++g) {
                                int sum = 0;
                                for (int j = 0; j < 16; ++j) sum += blocks[b].qs[16*g+j];
                                blocks[b].bsums[g] = int16_t(sum);
                            }
                        }
                    }
                }
            }
            for (int a = 0; a < s.slots; ++a) {
                int e = int(rng.next()%s.experts);
                if ((a+3*t)%11 == 0) e = -1;
                if (a > 0 && (a+t)%4 == 1) e = route(ids, a-1, t);
                if (s.pattern == 1) e = s.experts-1;
                if (s.pattern == 2) e = -1;
                if (s.pattern >= 3) e = (t*s.slots + a)%s.experts;
                route(ids, a, t) = e;
            }
        }
        if (s.cache) {
            owner = xeon_iq3_cache_create_v1(size_t(1024) << 20, 0, 1);
            require(owner && xeon_iq3_cache_add_v1(owner, w) == 1, "cache registration");
        }
        auto p = ggml_threadpool_params_default(s.threads);
        tp = ggml_threadpool_new(&p);
        plan = ggml_graph_plan(g, s.threads, tp);
        plan.use_ref = s.ref;
        scratch.assign(plan.work_size + 256, 0x5a);
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(scratch.data() + 64) + 63) & ~uintptr_t(63);
        plan.work_data = reinterpret_cast<uint8_t *>(aligned);
    }
    ~Data() {
        xeon_iq3_cache_free_v1(owner);
        ggml_threadpool_free(tp);
        ggml_backend_buffer_free(ab);
        ggml_backend_buffer_free(wb);
        ggml_free(ac);
        ggml_free(wc);
    }
    void compute() {
        if (s.pattern == 5) {
            for (int t = 0; t < s.tokens; ++t) for (int a = 0; a < s.slots; ++a)
                route(ids, a, t) = int(((epoch*s.tokens + t)*s.slots + a)%s.experts);
            ++epoch;
        }
        require(ggml_graph_compute(g, &plan) == GGML_STATUS_SUCCESS, "compute");
    }
    void guards() {
        for (size_t i = 1; i <= 32; ++i) {
            require(plan.work_data[-int(i)] == 0x5a, "workspace underflow");
            require(plan.work_data[plan.work_size + i - 1] == 0x5a, "workspace overflow");
        }
    }
    double verify() {
        const auto qt = ggml_get_type_traits_cpu(s.type)->vec_dot_type;
        std::vector<float> weights(s.k), input(s.k);
        std::vector<uint8_t> quantized(ggml_row_size(qt, s.k));
        double worst = 0;
        for (int t = 0; t < s.tokens; ++t) for (int a = 0; a < s.slots; ++a) {
            const int expert = route(ids, a, t);
            const int xi = a%s.activations;
            const float * actual = reinterpret_cast<const float *>(ptr(y, a, t));
            if (expert == -1) {
                for (int r = 0; r < s.rows; ++r) require(actual[r] == 0.f && !std::signbit(actual[r]), "skipped output");
                continue;
            }
            const void * data = ptr(x, xi, t);
            if (x->type != qt) {
                ggml_get_type_traits_cpu(qt)->from_float(static_cast<const float *>(data), quantized.data(), s.k);
                data = quantized.data();
            }
            dequant(qt, data, input.data(), s.k);
            for (int r = 0; r < s.rows; ++r) {
                const char * wr = ptr(w,r,expert);
                dequant(s.type, wr, weights.data(), s.k);
                double expected = 0, abs_total = 0;
                for (int j = 0; j < s.k; ++j) {
                    float coeff = weights[j];
                    if (qt == GGML_TYPE_Q8_K && static_cast<const block_q8_K *>(data)[j/256].qs[j%256] == -128) {
                        // With psignb, signed negation of byte -128 stays -128.
                        ggml_fp16_t scale;
                        std::memcpy(&scale, wr + (j/256)*ggml_type_size(s.type), sizeof(scale));
                        coeff = std::copysign(std::fabs(coeff), ggml_fp16_to_fp32(scale));
                    }
                    const double term = double(coeff)*input[j];
                    expected += term;
                    abs_total += std::fabs(term);
                }
                const double rel = std::fabs(double(actual[r]) - expected)/std::max(1e-20, abs_total);
                require(std::isfinite(rel) && rel < 5e-6, "dequantized-double oracle");
                worst = std::max(worst, rel);
            }
        }
        return worst;
    }
};
static void check(const char * dumpfile, bool edges) {
    std::vector<Shape> cases = {
        {GGML_TYPE_F32,37,17,13,4,9,1,1,false,true,false,false,0},
        {GGML_TYPE_F32,17,3,257,8,65,2,3,false,true,false,false,0},
        {GGML_TYPE_F32,7,1,256,8,512,1,3,false,false,false,false,3},
        {GGML_TYPE_F32,11,129,1,3,13,3,3,false,true,false,false,1},
        {GGML_TYPE_F32,5,9,7,4,17,2,2,false,true,false,false,2},
        {GGML_TYPE_MXFP4,256,17,7,4,9,1,3,false,true,false,false,1},
        {GGML_TYPE_MXFP4,288,17,7,4,9,4,1,false,true,false,false,0},
        {GGML_TYPE_MXFP4,320,19,3,4,23,2,3,false,true,true,false,0},
        {GGML_TYPE_MXFP4,256,17,3,2,1,1,1,false,false,false,false,1},
    };
    for (auto wt : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}) {
        for (bool cache : {false, true}) {
            cases.push_back({wt,256,17,7,8,1,1,4,cache,false,false,false,3});
            cases.push_back({wt,768,71,7,8,1,2,3,cache,true,false,false,0});
            cases.push_back({wt,2048,41,3,4,1,4,3,cache,true,false,false,1});
            cases.push_back({wt,7168,9,4,4,1,4,1,cache,false,false,false,4});
            cases.push_back({wt,1280,17,7,4,1,2,3,cache,true,true,false,0});
            cases.push_back({wt,512,65,5,8,9,2,4,cache,true,false,false,0});
            cases.push_back({wt,1024,18,3,4,1,4,3,cache,false,false,true,3});
        }
    }
    if (edges) {
        cases = {
            {GGML_TYPE_F32,19,19,257,15,1,3,3,false,true,false,false,0},
            {GGML_TYPE_F32,19,19,257,16,1,4,3,false,true,false,false,3},
            {GGML_TYPE_F32,19,19,257,17,1,1,3,false,true,false,false,0},
            {GGML_TYPE_F32,3,3,1,16,1,4,3,false,true,false,false,2},
            {GGML_TYPE_MXFP4,288,17,1,16,1,4,3,false,true,false,false,1},
            {GGML_TYPE_MXFP4,256,67,13,15,1,5,3,false,true,true,false,0},
            {GGML_TYPE_IQ3_S,1024,65,4,16,1,4,3,false,true,false,false,1},
            {GGML_TYPE_IQ3_S,1024,65,4,16,1,4,3,true,true,false,false,1},
            {GGML_TYPE_IQ2_S,768,35,13,16,1,4,4,true,true,true,false,0},
            {GGML_TYPE_IQ3_XXS,512,33,257,17,1,1,2,true,false,false,false,3},
            {GGML_TYPE_IQ2_S,768,65,7,8,13,4,3,false,true,false,false,0},
        };
    }
    std::ofstream dump(dumpfile, std::ios::binary);
    require(dump.good(), "output dump");
    int no = 0;
    std::puts("case,seed,type,k,rows,experts,slots,tokens,activations,threads,cache,pad,raw,ref,pattern,work_bytes,cache_worker_hits,hash64,max_abs_error_over_sum_abs");
    for (uint32_t seed : {0x17295e3u, 0x2ed715bu, 0xe7ea3f91u}) {
        for (const auto & s : cases) {
            Data d(s, seed);
            const uint64_t hits = xeon_iq3_cache_hits_v1();
            d.compute();
            d.guards();
            const uint64_t delta = xeon_iq3_cache_hits_v1() - hits;
            require((delta > 0) == (s.cache && s.tokens == 1 && !s.ref), "cache path");
            const double worst = d.verify();
            dump.write(static_cast<const char *>(d.y->data), ggml_nbytes(d.y));
            require(dump.good(), "dump write");
            std::printf("%d,%u,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%llu,%016llx,%.9g\n",
                no++,seed,ggml_type_name(s.type),s.k,s.rows,s.experts,s.slots,s.tokens,s.activations,s.threads,
                s.cache,s.pad,s.raw,s.ref,s.pattern,d.plan.work_size,(unsigned long long)delta,
                (unsigned long long)hash_bytes(d.y->data,ggml_nbytes(d.y)),worst);
        }
    }
}
static void bench(int argc, char ** argv) {
    if (argc != 15) throw std::runtime_error("bench TYPE K M E S T A THREADS CACHE PATTERN LOOPS SAMPLES SEED");
    int a = 2;
    Shape s{};
    s.type=get_type(argv[a++]); s.k=std::stoi(argv[a++]); s.rows=std::stoi(argv[a++]); s.experts=std::stoi(argv[a++]);
    s.slots=std::stoi(argv[a++]); s.tokens=std::stoi(argv[a++]); s.activations=std::stoi(argv[a++]); s.threads=std::stoi(argv[a++]);
    s.cache=std::stoi(argv[a++]); s.pattern=std::stoi(argv[a++]);
    const int loops=std::stoi(argv[a++]), samples=std::stoi(argv[a++]);
    const uint32_t seed=std::stoul(argv[a++]);
    require(loops > 0 && samples > 0, "invalid repeat count");
    Data d(s,seed);
    for (int i = 0; i < 3; ++i) d.compute();
    for (int j = 0; j < samples; ++j) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < loops; ++i) d.compute();
        auto end = std::chrono::steady_clock::now();
        d.guards();
        double ns = std::chrono::duration<double,std::nano>(end-start).count()/loops;
        std::printf("%d,%.1f,%zu,%016llx\n",j,ns,d.plan.work_size,
            (unsigned long long)hash_bytes(d.y->data,ggml_nbytes(d.y)));
    }
}
static void rounded_graphs(const char * file, int mode) {
    const int rounding[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    require(mode >= 0 && mode < 4 && std::fesetround(rounding[mode]) == 0, "fesetround");
    std::ofstream dump(file,std::ios::binary);
    require(dump.good(), "rounding dump");
    std::puts("mode,type,k,threads,raw,pattern,hash");
    for (ggml_type type : {GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S}) {
        for (int threads : {1,3}) for(bool raw : {false,true}) for (int pattern : {3,6}) {
            for (int k : {256,1280}) {
                Shape s{type,k,49,7,6,1,3,threads,true,true,raw,false,pattern};
                Data d(s,0x871defbu);
                d.compute(); d.guards();
                dump.write(static_cast<const char *>(d.y->data),ggml_nbytes(d.y));
                require(dump.good(),"write rounding dump");
                std::printf("%d,%s,%d,%d,%d,%d,%016llx\n",mode,ggml_type_name(type),k,threads,raw,pattern,
                            (unsigned long long)hash_bytes(d.y->data,ggml_nbytes(d.y)));
            }
        }
    }
}
static void shape_plans() {
    std::puts("projection,type,k,rows,experts,slots,tokens,activations,threads,work_bytes");
    for (auto wt : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_MXFP4}) {
        for (bool down : {false,true}) {
            for (int tokens : {1,32,128,512,1024}) {
                const int k = down ? 2304 : 5120, m = down ? 5120 : 2304;
                const int a = down ? 6 : 1, threads = tokens == 1 ? 12 : 24;
                ggml_context * ctx = ggml_init({1024*1024,nullptr,true});
                auto * w = ggml_new_tensor_3d(ctx,wt,k,m,384);
                auto * x = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,a,tokens);
                auto * ids = ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,tokens);
                auto * y = ggml_mul_mat_id(ctx,w,x,ids);
                auto * graph = ggml_new_graph_custom(ctx,32,false);
                ggml_build_forward_expand(graph,y);
                auto plan = ggml_graph_plan(graph,threads,nullptr);
                std::printf("%s,%s,%d,%d,384,6,%d,%d,%d,%zu\n",down ? "down" : "gate/up",ggml_type_name(wt),k,m,tokens,a,threads,plan.work_size);
                ggml_free(ctx);
            }
        }
    }
}
int main(int argc, char ** argv) {
    try {
        ggml_cpu_init();
        if (argc == 4 && std::string(argv[1]) == "rounding") rounded_graphs(argv[3], std::stoi(argv[2]));
        else if (argc == 2 && std::string(argv[1]) == "plans") shape_plans();
        else if (argc == 3 && (std::string(argv[1]) == "check" || std::string(argv[1]) == "edges")) check(argv[2], std::string(argv[1])=="edges");
        else if (argc > 1 && std::string(argv[1]) == "bench") bench(argc, argv);
        else throw std::runtime_error("expected check DUMP or bench ...");
    } catch (const std::exception & e) {
        std::fprintf(stderr,"graph_probe: %s\n",e.what());
        return 1;
    }
}
