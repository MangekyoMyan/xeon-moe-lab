#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "iq3-cache.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
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
            const char * setting = std::getenv("TEST_EXPERT_MASK");
            require(setting && s.experts <= 63, "missing test mask or invalid dimensions");
            const uint64_t mask = std::stoull(setting);
            std::vector<uint32_t> experts;
            for (int e = s.experts - 1; e >= 0; --e) {
                if ((mask >> e) & 1) experts.push_back(e);
            }
            if (!experts.empty()) {
                owner = xeon_iq3_cache_create_v1(size_t(1024) << 20, 0, 1);
                require(owner && xeon_iq3_cache_add_selected_v2(owner, w, experts.data(), experts.size()) == 1,
                        "sparse cache registration");
            }
            auto table = xeon_iq3_cache_lookup_experts_v2(w);
            require((table != nullptr) == !experts.empty(), "cache table presence");
            for (int e = 0; table && e < s.experts; ++e) {
                require((table[e] != nullptr) == bool((mask >> e) & 1), "sparse pointer identity");
            }
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
        require(ggml_graph_compute(g, &plan) == GGML_STATUS_SUCCESS, "compute");
    }
    void guards() {
        for (size_t i = 1; i <= 32; ++i) {
            require(plan.work_data[-int(i)] == 0x5a, "workspace underflow");
            require(plan.work_data[plan.work_size + i - 1] == 0x5a, "workspace overflow");
        }
    }
};

static const ggml_type iq_types[] = {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S};
using Counters = std::array<uint64_t, 8>;
using ReadCounters = void (*)(uint64_t *);

static const int loop_routes[][6] = {
    {0,1,2,3,4,5},
    {6,6,6,6,6,6},
    {-1,-1,-1,-1,-1,-1},
    {-1,5,4,4,1,0},
    {3,1,4,0,5,2},
};
static void set_routes(Data & d, int phase) {
    for (int t=0; t<d.s.tokens; ++t) {
        for (int i=0; i<6; ++i) {
            route(d.ids,i,t) = loop_routes[(phase+t)%5][i];
        }
    }
}
static void verify_skips(Data & d) {
    for (int t=0; t<d.s.tokens; ++t) {
        for (int i=0; i<6; ++i) {
            if (route(d.ids,i,t) >= 0) continue;
            const float * y = reinterpret_cast<const float *>(ptr(d.y,i,t));
            require(std::all_of(y,y+d.s.rows,[](float value){return value==0;}), "nonzero skipped slot");
        }
    }
}
static void numeric(int mode, const char * output) {
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    require(mode>=0 && mode<4 && std::fesetround(modes[mode]) == 0, "rounding mode");
    std::ofstream dump(output,std::ios::binary);
    require(dump.good(), "open output dump");
    const auto read = reinterpret_cast<ReadCounters>(dlsym(RTLD_DEFAULT,"test_reset_counts"));
    require(read, "numeric tests need the path counter DSO");
    std::puts("case,phase,type,k,rows,tokens,inputs,threads,padded,raw_q8,ref,offset,bytes,hash,native_dots,legacy_dots,fold_select,prepare_blocks,prepare_calls,mx_prepare,mx_dot,prepare_round_mask");
    uint64_t offset=0;
    auto run = [&](const char * name, Shape shape, bool phases) {
        Data d(shape,153311);
        for (int i=0; i<(phases?5:1); ++i) {
            set_routes(d,i);
            Counters counts{};
            read(counts.data());
            d.compute();
            read(counts.data());
            d.guards();
            verify_skips(d);
            const size_t n = ggml_nbytes(d.y);
            dump.write(static_cast<const char *>(d.y->data),n);
            require(dump.good(), "write output dump");
            std::printf("%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%zu,%016llx", name,i,ggml_type_name(shape.type),
                    shape.k,shape.rows,shape.tokens,shape.activations,shape.threads,shape.pad,shape.raw,shape.ref,
                    (unsigned long long)offset,n,(unsigned long long)hash_bytes(d.y->data,n));
            for (uint64_t count : counts) std::printf(",%llu",(unsigned long long)count);
            std::putchar('\n');
            offset+=n;
        }
    };
    for (ggml_type type:iq_types) {
        // The same graph and scratch see route changes, repeated experts, skipped slots and unselected experts.
        run("strided_down", {type,1280,17,7,6,1,6,3,true,true,false,false,6}, true);
        run("shared_gate", {type,512,17,7,6,1,1,1,true,false,false,false,3}, true);
        run("strided_broadcast", {type,256,17,7,6,1,3,3,true,true,false,false,6}, true);
        run("prequantized", {type,512,17,7,6,1,6,3,true,true,true,false,3}, true);
        run("reference", {type,256,17,7,6,1,2,3,true,true,false,true,3}, false);
        run("two_tokens", {type,512,17,7,6,2,2,3,true,true,false,false,6}, false);
    }
    run("mxfp4_prepack", {GGML_TYPE_MXFP4,512,17,7,6,1,6,3,false,false,false,false,3}, false);
}
static void bench(int k, int rows, int inputs, int threads, int loops, int samples, uint32_t seed) {
    require(loops>0 && samples>0,"timing repeat count");
    Shape s{GGML_TYPE_IQ3_XXS,k,rows,7,6,1,inputs,threads,true,false,false,false,3};
    Data d(s,seed);
    set_routes(d,0);
    for (int i=0;i<3;++i) d.compute();
    for (int sample=0;sample<samples;++sample) {
        auto begin=std::chrono::steady_clock::now();
        for (int i=0;i<loops;++i) d.compute();
        auto end=std::chrono::steady_clock::now();
        d.guards();
        double ns=std::chrono::duration<double,std::nano>(end-begin).count()/loops;
        std::printf("%d,%.1f,%zu,%016llx\n",sample,ns,d.plan.work_size,
                    (unsigned long long)hash_bytes(d.y->data,ggml_nbytes(d.y)));
    }
}
static void report_library(const char * name, const void * fn) {
    Dl_info info{};
    require(dladdr(fn,&info) != 0 && info.dli_fname, "loaded function provider");
    std::fprintf(stderr,"provider,%s,%s\n",name,info.dli_fname);
}
int main(int argc, char ** argv) {
    try {
        report_library("cpu",reinterpret_cast<const void *>(&ggml_graph_compute));
        report_library("base",reinterpret_cast<const void *>(&ggml_mul_mat_id));
        std::ifstream maps("/proc/self/maps");
        std::string line;
        std::set<std::string> libraries;
        while (std::getline(maps,line)) {
            const auto path = line.find('/');
            if (path != line.npos && line.find("/libggml",path) != line.npos) libraries.insert(line.substr(path));
        }
        for (const auto & path : libraries) std::fprintf(stderr,"map,%s\n",path.c_str());
        ggml_cpu_init();
        if (argc==4 && std::string(argv[1])=="numeric") numeric(std::stoi(argv[2]),argv[3]);
        else if (argc==9 && std::string(argv[1])=="bench") {
            bench(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),
                  std::stoi(argv[6]),std::stoi(argv[7]),std::stoul(argv[8]));
        } else throw std::runtime_error("numeric MODE DUMP | bench K ROWS INPUTS THREADS LOOPS SAMPLES SEED");
    } catch(const std::exception & e) {
        std::fprintf(stderr,"graph_probe: %s\n",e.what());
        return 1;
    }
}
