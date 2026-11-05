#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

struct shape {
    int k, m, n, w2, w3, x2, x3;
    bool row_view;
};

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: kernel-graph-probe OUTPUT.bin\n");
        return 1;
    }
    std::ofstream out(argv[1], std::ios::binary);
    if (!out) return 1;
    ggml_cpu_init();
    int cases = 0;
    for (const auto type : {GGML_TYPE_Q8_0}) {
        const int unit = type == GGML_TYPE_Q8_0 ? 32 : 256;
        const shape shapes[] = {
            {4096, 65, 1, 1, 1, 1, 1, false},
            {1408, 35, 1, 2, 2, 4, 2, true},
            {unit,   65, 1, 1, 1, 1, 1, false},
            {3*unit, 33, 1, 1, 1, 1, 1, false},
            {2*unit, 67, 1, 2, 2, 4, 2, false},
            {4096, 17, 3, 1, 1, 2, 2, false},
            {1408, 35, 1, 2, 2, 4, 2, true},
        };
        ggml_quantize_init(type);
        for (const auto & s : shapes) {
            ggml_init_params ip{16u * 1024u * 1024u, nullptr, false};
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) return 2;
            const int full_m = s.m * (s.row_view ? 2 : 1);
            auto * base = ggml_new_tensor_4d(ctx, type, s.k, full_m, s.w2, s.w3);
            auto * weight = s.row_view ? ggml_view_4d(ctx, base, s.k, s.m, s.w2, s.w3,
                2 * base->nb[1], base->nb[2], base->nb[3], 0) : base;
            auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, s.n, s.x2, s.x3);
            std::vector<float> values(s.k), importance(s.k, 1.0f);
            for (int i = 0; i < full_m * s.w2 * s.w3; ++i) {
                for (int j = 0; j < s.k; ++j) {
                    values[j] = float((i * 37 + j * 17 + i * j) % 263 - 131) / 32.0f;
                }
                const auto bytes = ggml_quantize_chunk(type, values.data(),
                    static_cast<char *>(base->data) + i * base->nb[1], 0, 1, s.k, importance.data());
                if (bytes != base->nb[1]) return 3;
            }
            for (int64_t i = 0; i < ggml_nelements(x); ++i) {
                static_cast<float *>(x->data)[i] = float((i * 23 + 11) % 257 - 128) / 64.0f;
            }
            auto * product = ggml_mul_mat(ctx, weight, x);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, product);
            std::vector<char> reference;
            for (const int threads : {1, 4, 12, 24}) {
                ggml_set_f32(product, NAN);
                if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) return 4;
                for (int64_t i = 0; i < ggml_nelements(product); ++i) {
                    if (!std::isfinite(static_cast<float *>(product->data)[i])) return 5;
                }
                const size_t bytes = ggml_nbytes(product);
                if (threads == 1) {
                    reference.resize(bytes);
                    std::memcpy(reference.data(), product->data, bytes);
                } else if (std::memcmp(reference.data(), product->data, bytes) != 0) {
                    std::fprintf(stderr, "Thread mismatch type=%s K=%d M=%d N=%d view=%d\n",
                        ggml_type_name(type), s.k, s.m, s.n, s.row_view);
                    return 6;
                }
                out.write(static_cast<const char *>(product->data), bytes);
                if (!out) return 7;
                std::printf("PASS type=%s K=%d M=%d N=%d weights=%dx%d activation=%dx%d row_view=%d threads=%d\n",
                    ggml_type_name(type), s.k, s.m, s.n, s.w2, s.w3, s.x2, s.x3, s.row_view, threads);
                ++cases;
            }
            ggml_free(ctx);
        }
    }
    ggml_quantize_free();
    std::printf("PASS %d cases\n", cases);
    return 0;
}
