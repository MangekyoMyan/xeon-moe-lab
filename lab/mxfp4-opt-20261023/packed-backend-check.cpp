#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float random_float(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (int(state & 65535) - 32768) / 32768.0f;
}

int main() {
    assert(std::getenv("GGML_CPU_MXFP4_PACKED") && std::strcmp(std::getenv("GGML_CPU_MXFP4_PACKED"), "1") == 0);
    ggml_cpu_init();
    auto * backend = ggml_backend_cpu_init();
    assert(backend);
    auto buft = ggml_backend_cpu_repack_buffer_type();
    assert(buft && std::strcmp(ggml_backend_buft_name(buft), "CPU_REPACK") == 0);
    uint64_t cases = 0, compared = 0, bytes_checked = 0;
    for (int k : {32, 64, 96, 2304, 2880, 5120}) {
        for (int m : {4, 12, 64}) {
            const int experts = 4;
            auto * raw_ctx = ggml_init({1 << 20, nullptr, true});
            auto * packed_ctx = ggml_init({1 << 20, nullptr, true});
            assert(raw_ctx && packed_ctx);
            auto * raw = ggml_new_tensor_3d(raw_ctx, GGML_TYPE_MXFP4, k, m, experts);
            auto * packed = ggml_new_tensor_3d(packed_ctx, GGML_TYPE_MXFP4, k, m, experts);
            auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(packed_ctx, buft);
            assert(buffer && packed->extra);
            std::vector<unsigned char> data(ggml_nbytes(raw));
            raw->data = data.data();
            std::vector<float> row(k);
            uint32_t seed = 197 + k + m;
            const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, k);
            for (int r = 0; r < m * experts; ++r) {
                for (float & value : row) value = random_float(seed);
                ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->from_float(row.data(), data.data() + r * stride, k);
            }
            ggml_backend_tensor_set(packed, data.data(), 0, data.size());
            const auto * packed_bytes = (const unsigned char *) packed->data;
            for (int r = 0; r < m * experts; ++r) {
                for (int b = 0; b < k / 32; ++b) {
                    const auto * original = data.data() + r * stride + b * 17;
                    const auto * group = packed_bytes + size_t(r / 4) * 4 * stride + size_t(b) * 68;
                    assert(group[r % 4] == original[0]);
                    assert(!std::memcmp(group + 4 + (r % 4) * 16, original + 1, 16));
                    bytes_checked += 17;
                }
            }
            for (int selected : {1, 3, 4}) {
                for (int tokens : {1, 2, 17}) {
                    for (bool broadcast : {false, true}) {
                        for (bool duplicate : {false, true}) {
                            auto * ctx = ggml_init({16 << 20, nullptr, false});
                            assert(ctx);
                            auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, broadcast ? 1 : selected, tokens);
                            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, tokens);
                            for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = random_float(seed);
                            for (int t = 0; t < tokens; ++t) {
                                for (int i = 0; i < selected; ++i) ((int32_t *) ids->data)[t * selected + i] = duplicate ? experts - 1 : (t + i) % experts;
                            }
                            auto * reference = ggml_mul_mat_id(ctx, raw, x, ids);
                            auto * candidate = ggml_mul_mat_id(ctx, packed, x, ids);
                            auto * ref_graph = ggml_new_graph(ctx);
                            auto * graph = ggml_new_graph(ctx);
                            ggml_build_forward_expand(ref_graph, reference);
                            ggml_build_forward_expand(graph, candidate);
                            for (int threads : {1, 24}) {
                                auto params = ggml_threadpool_params_default(threads);
                                params.strict_cpu = true;
                                for (int i = 0; i < threads; ++i) params.cpumask[i] = true;
                                auto * pool = ggml_threadpool_new(&params);
                                assert(pool);
                                auto ref_plan = ggml_graph_plan(ref_graph, threads, pool);
                                auto plan = ggml_graph_plan(graph, threads, pool);
                                std::vector<unsigned char> ref_work(ref_plan.work_size), work(plan.work_size);
                                ref_plan.work_data = ref_work.data();
                                plan.work_data = work.data();
                                assert(ggml_graph_compute(ref_graph, &ref_plan) == GGML_STATUS_SUCCESS);
                                assert(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
                                if (std::memcmp(reference->data, candidate->data, ggml_nbytes(reference))) {
                                    std::fprintf(stderr, "Mismatch K%d M%d S%d T%d broadcast%d duplicate%d threads%d\n", k, m, selected, tokens, broadcast, duplicate, threads);
                                    return 3;
                                }
                                for (int64_t i = 0; i < ggml_nelements(reference); ++i) assert(std::isfinite(((float *) reference->data)[i]));
                                ++cases;
                                compared += ggml_nelements(reference);
                                ggml_threadpool_free(pool);
                            }
                            ggml_free(ctx);
                        }
                    }
                }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(packed_ctx);
            ggml_free(raw_ctx);
            std::printf("{\"k\":%d,\"m\":%d,\"cases_so_far\":%llu}\n", k, m, (unsigned long long) cases);
            std::fflush(stdout);
        }
    }
    std::printf("{\"passed\":true,\"cases\":%llu,\"float_values_compared\":%llu,\"weight_bytes_roundtrip_checked\":%llu,\"duplicate_ids_checked\":true,\"bit_equal\":true}\n", (unsigned long long) cases, (unsigned long long) compared, (unsigned long long) bytes_checked);
    ggml_backend_free(backend);
}
