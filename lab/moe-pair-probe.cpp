#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

struct shape {
    int k, m, experts, selected, tokens;
    bool row_view, separate_input, duplicate_ids;
};

static unsigned rng = 0x7310042;
static unsigned next() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void fill_weight(ggml_tensor * weight) {
    auto * bytes = static_cast<unsigned char *>(weight->data);
    const size_t block_bytes = ggml_type_size(weight->type);
    for (size_t i = 0; i < ggml_nbytes(weight); i += block_bytes) {
        const ggml_fp16_t scale = ggml_fp32_to_fp16(0.01f * (1 + next() % 7));
        std::memcpy(bytes + i, &scale, sizeof(scale));
        for (size_t j = sizeof(scale); j < block_bytes; ++j) bytes[i + j] = next();
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) return 1;
    ggml_cpu_init();
    const shape shapes[] = {
        {256, 1, 3, 1, 1, false, false, false},
        {768, 33, 9, 6, 1, false, false, false},
        {4096, 65, 8, 6, 1, false, false, false},
        {4096, 2048, 8, 6, 1, false, false, false},
        {512, 67, 7, 4, 1, false, false, true},
        {256, 35, 5, 2, 2, false, false, false},
        {512, 35, 5, 2, 1, true, false, false},
        {256, 17, 3, 2, 1, false, true, false},
    };
    int cases = 0;
    for (const auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_0}) {
        for (const auto & s : shapes) {
            ggml_init_params ip{192u * 1024u * 1024u, nullptr, false};
            auto * ctx = ggml_init(ip);
            if (!ctx) return 2;
            const int full_m = s.m * (s.row_view ? 2 : 1);
            auto * gate_base = ggml_new_tensor_3d(ctx, type, s.k, full_m, s.experts);
            auto * up_base = ggml_new_tensor_3d(ctx, type, s.k, full_m, s.experts);
            fill_weight(gate_base);
            fill_weight(up_base);
            std::vector<ggml_backend_buffer_t> owners, mapped;
            if (s.m == 2048) {
                for (auto * t : {gate_base, up_base}) {
                    auto * owner = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(t));
                    if (!owner) return 7;
                    void * data = ggml_backend_buffer_get_base(owner);
                    std::memcpy(data, t->data, ggml_nbytes(t));
                    auto * buffer = ggml_backend_cpu_buffer_from_ptr(data, ggml_nbytes(t));
                    t->data = data;
                    t->buffer = buffer;
                    owners.push_back(owner);
                    mapped.push_back(buffer);
                }
            }
            auto * gate_weight = s.row_view ? ggml_view_3d(ctx, gate_base, s.k, s.m, s.experts,
                2 * gate_base->nb[1], gate_base->nb[2], 0) : gate_base;
            auto * up_weight = s.row_view ? ggml_view_3d(ctx, up_base, s.k, s.m, s.experts,
                2 * up_base->nb[1], up_base->nb[2], 0) : up_base;
            auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, 1, s.tokens);
            for (int64_t i = 0; i < ggml_nelements(x); ++i)
                static_cast<float *>(x->data)[i] = float(int(next() % 257) - 128) / 64.0f;
            auto * x2 = s.separate_input ? ggml_dup_tensor(ctx, x) : x;
            if (s.separate_input) std::memcpy(x2->data, x->data, ggml_nbytes(x));
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.selected, s.tokens);
            auto * gate = ggml_mul_mat_id(ctx, gate_weight, x, ids);
            auto * up = ggml_mul_mat_id(ctx, up_weight, x2, ids);
            ggml_set_name(gate, "pair_probe_gate");
            ggml_set_name(up, "pair_probe_up");
            ggml_set_output(gate);
            ggml_set_output(up);
            auto * result = ggml_swiglu_clamp(ctx, gate, up, 10.0f);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            for (int round = 0; round < 2; ++round) {
                for (int i = 0; i < s.selected * s.tokens; ++i)
                    static_cast<int32_t *>(ids->data)[i] = s.duplicate_ids ? 2 : (i + round + 1) % s.experts;
                std::vector<char> reference;
                for (int threads : {1, 4, 12}) {
                    for (auto * t : {gate, up, result}) ggml_set_f32(t, NAN);
                    if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS) return 3;
                    std::vector<char> values;
                    for (auto * t : {gate, up, result}) {
                        for (int64_t i = 0; i < ggml_nelements(t); ++i)
                            if (!std::isfinite(static_cast<float *>(t->data)[i])) return 4;
                        auto * begin = static_cast<const char *>(t->data);
                        values.insert(values.end(), begin, begin + ggml_nbytes(t));
                    }
                    if (threads == 1) reference = values;
                    else if (values != reference) {
                        std::fprintf(stderr, "Thread mismatch: %s K=%d M=%d threads=%d\n", ggml_type_name(type), s.k, s.m, threads);
                        return 5;
                    }
                    output.write(values.data(), values.size());
                    if (!output) return 6;
                    ++cases;
                }
            }
            std::printf("PASS %s K=%d M=%d experts=%d selected=%d tokens=%d view=%d separate=%d duplicate=%d\n",
                ggml_type_name(type), s.k, s.m, s.experts, s.selected, s.tokens,
                s.row_view, s.separate_input, s.duplicate_ids);
            ggml_free(ctx);
            for (auto * buffer : mapped) ggml_backend_buffer_free(buffer);
            for (auto * buffer : owners) ggml_backend_buffer_free(buffer);
        }
    }
    std::printf("PASS %d cases\n", cases);
    return 0;
}
