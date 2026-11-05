#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
template<class F> static double median_ms(F fn, int count = 3) {
    std::vector<double> samples;
    for (int i = 0; i < count; ++i) {
        const auto start = clock_type::now();
        fn();
        samples.push_back(milliseconds(start));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size()/2];
}

struct graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_t backend;
    ggml_tensor * bank;
    ggml_tensor * weights;
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * output;
    ggml_cgraph * gf;
    graph(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int experts, int tokens, bool broadcast)
        : backend(backend) {
        ctx = ggml_init({4*1024*1024, nullptr, true});
        if (!ctx) throw std::runtime_error("metadata allocation failed");
        // Match the cache's extra expert padding for CUDA MMQ overreads.
        bank = ggml_new_tensor_3d(ctx, type, k, m, experts + 1);
        weights = ggml_view_3d(ctx, bank, k, m, experts, bank->nb[1], bank->nb[2], 0);
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, broadcast ? 1 : 6, tokens);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, tokens);
        output = ggml_mul_mat_id(ctx, weights, input, ids);
        ggml_set_name(bank, "real_expert_weights");
        ggml_set_name(input, "synthetic_activations");
        ggml_set_name(ids, "controlled_routing");
        ggml_set_name(output, "expert_result");
        if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("backend rejects operator");
        gf = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(gf, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("graph allocation failed");
        ggml_backend_buffer_clear(buffer, 0);
    }
    void compute() {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
        ggml_backend_synchronize(backend);
    }
    ~graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

int main(int argc, char ** argv) try {
    if (argc != 7) throw std::runtime_error("usage: shard tensor tokens hot|spread threads seed");
    const int tokens = std::stoi(argv[3]);
    const bool spread = std::string(argv[4]) == "spread";
    const int threads = std::stoi(argv[5]);
    const int seed = std::stoi(argv[6]);
    if ((tokens != 1 && tokens != 16 && tokens != 128) || threads < 1 || threads > 24) throw std::runtime_error("out of bounded test range");
    const int experts = spread ? tokens*6 : 6;
    const bool broadcast = std::string(argv[2]).find("down_exps") == std::string::npos;

    gguf_context * meta = gguf_init_from_file(argv[1], {true, nullptr});
    if (!meta) throw std::runtime_error("GGUF metadata read failed");
    const int64_t ti = gguf_find_tensor(meta, argv[2]);
    if (ti < 0) throw std::runtime_error("tensor missing");
    const auto type = gguf_get_tensor_type(meta, ti);
    const auto ne = gguf_get_tensor_ne(meta, ti);
    const int64_t k = ne[0], m = ne[1];
    if (ne[2] < experts || ne[3] != 1 || k > 5120 || m > 5120) throw std::runtime_error("unexpected tensor dimensions");
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t expert_bytes = row_bytes*m;
    const size_t bytes = expert_bytes*experts;
    const size_t offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, ti);
    if (bytes > 700*1024*1024) throw std::runtime_error("bounded weight slice exceeded");
    std::vector<uint8_t> raw(bytes);
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(offset);
    if (!file.read(reinterpret_cast<char *>(raw.data()), bytes)) throw std::runtime_error("weight slice read failed");
    gguf_free(meta);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> input(k*(broadcast ? 1 : 6)*tokens);
    for (float & f : input) f = dist(rng);
    std::vector<int32_t> ids(6*tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < 6; ++j) ids[t*6 + j] = spread ? t*6+j : (j+t)%6;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) throw std::runtime_error("backend init failed");
    ggml_backend_cpu_set_n_threads(cpu, threads);
    auto pool_params = ggml_threadpool_params_default(threads);
    pool_params.poll = 0;
    pool_params.strict_cpu = true;
    for (int i = 0; i < 24; ++i) pool_params.cpumask[i] = true;
    auto pool = ggml_threadpool_new(&pool_params);
    if (!pool) throw std::runtime_error("CPU threadpool allocation failed");
    ggml_backend_cpu_set_threadpool(cpu, pool);
    bool pass;
    {
        graph gc(cpu, type, k, m, experts, tokens, broadcast);
        graph gg(gpu, type, k, m, experts, tokens, broadcast);
        for (graph * g : {&gc, &gg}) {
            ggml_backend_tensor_set(g->weights, raw.data(), 0, bytes);
            ggml_backend_tensor_set(g->input, input.data(), 0, input.size()*sizeof(float));
            ggml_backend_tensor_set(g->ids, ids.data(), 0, ids.size()*sizeof(int32_t));
            g->compute();
        }
        std::vector<float> yc(ggml_nelements(gc.output)), yg(yc.size());
        ggml_backend_tensor_get(gc.output, yc.data(), 0, yc.size()*sizeof(float));
        ggml_backend_tensor_get(gg.output, yg.data(), 0, yg.size()*sizeof(float));
        double error = 0, norm = 0, max_error = 0;
        int nonfinite = 0;
        for (size_t i = 0; i < yc.size(); ++i) {
            if (!std::isfinite(yc[i]) || !std::isfinite(yg[i])) { ++nonfinite; continue; }
            const double d = double(yc[i]) - yg[i];
            error += d*d;
            norm += double(yc[i])*yc[i];
            max_error = std::max(max_error, std::abs(d));
        }
        // Independent sampled FP64 dots over dequantized weights and original F32 inputs.
        std::vector<float> row(k);
        const auto to_float = ggml_get_type_traits(type)->to_float;
        if (!to_float) throw std::runtime_error("no dequantizer for reference");
        double ref_norm = 0, cpu_ref_error = 0, gpu_ref_error = 0;
        int ref_nonfinite = 0;
        for (int t = 0; t < tokens; ++t) for (int j = 0; j < 6; ++j) for (int s = 0; s < 8; ++s) {
            const int64_t ri = (m-1)*s/7;
            to_float(raw.data() + ids[t*6+j]*expert_bytes + ri*row_bytes, row.data(), k);
            const float * x = input.data() + (t*(broadcast ? 1 : 6) + (broadcast ? 0 : j))*k;
            double ref = 0;
            for (int64_t col = 0; col < k; ++col) ref += double(row[col])*x[col];
            if (!std::isfinite(ref)) { ++ref_nonfinite; continue; }
            const size_t oi = (t*6+j)*m+ri;
            ref_norm += ref*ref;
            cpu_ref_error += std::pow(ref-yc[oi], 2);
            gpu_ref_error += std::pow(ref-yg[oi], 2);
        }
        const double nmse = error/std::max(norm, 1e-30);
        const double cpu_ref_nmse = cpu_ref_error/std::max(ref_norm, 1e-30);
        const double gpu_ref_nmse = gpu_ref_error/std::max(ref_norm, 1e-30);
        pass = norm > 0 && ref_norm > 0 && nonfinite == 0 && ref_nonfinite == 0 && nmse < 5e-4 && cpu_ref_nmse < 5e-4 && gpu_ref_nmse < 5e-4;
        const double cpu_ms = median_ms([&]{gc.compute();});
        const double gpu_ms = median_ms([&]{gg.compute();}, 7);
        auto upload = [&](const void * src) {
            ggml_backend_tensor_set_async(gpu, gg.weights, src, 0, bytes);
            ggml_backend_synchronize(gpu);
        };
        const double pageable_ms = median_ms([&]{upload(raw.data());});
        const double transfer_compute_ms = median_ms([&]{upload(raw.data()); gg.compute();});
        const int transfer_cpu = sched_getcpu();
        auto pinned = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), bytes);
        if (!pinned) throw std::runtime_error("pinned staging allocation failed");
        const std::string pinned_kind = ggml_backend_buft_name(ggml_backend_buffer_get_type(pinned));
        void * pinned_data = ggml_backend_buffer_get_base(pinned);
        const double staging_ms = median_ms([&]{std::memcpy(pinned_data, raw.data(), bytes);});
        const double pinned_ms = median_ms([&]{upload(pinned_data);});
        const double staged_total_ms = median_ms([&]{std::memcpy(pinned_data, raw.data(), bytes); upload(pinned_data); gg.compute();});
        ggml_backend_buffer_free(pinned);
        std::printf("{\"schema\":2,\"persistent_pool\":true,\"poll\":0,\"transfer_cpu\":%d,\"pinned_buffer_type\":\"%s\",\"type\":\"%s\",\"k\":%lld,\"m\":%lld,\"tokens\":%d,\"experts\":%d,\"pattern\":\"%s\",\"broadcast\":%s,\"threads\":%d,\"weight_bytes\":%zu,\"cpu_ms\":%.6f,\"gpu_resident_ms\":%.6f,\"pageable_upload_ms\":%.6f,\"pageable_upload_compute_ms\":%.6f,\"pinned_upload_ms\":%.6f,\"staging_ms\":%.6f,\"staged_upload_compute_ms\":%.6f,\"cpu_gpu_nmse\":%.12g,\"cpu_ref_nmse\":%.12g,\"gpu_ref_nmse\":%.12g,\"max_abs_error\":%.9g,\"nonfinite\":%d,\"reference_nonfinite\":%d,\"pass\":%s}\n",
                    transfer_cpu, pinned_kind.c_str(),
                    ggml_type_name(type), (long long)k, (long long)m, tokens, experts, spread ? "spread" : "hot", broadcast ? "true" : "false", threads, bytes,
                    cpu_ms, gpu_ms, pageable_ms, transfer_compute_ms, pinned_ms, staging_ms, staged_total_ms,
                    nmse, cpu_ref_nmse, gpu_ref_nmse, max_error, nonfinite, ref_nonfinite, pass ? "true" : "false");
    }
    ggml_backend_cpu_set_threadpool(cpu, nullptr);
    ggml_threadpool_free(pool);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return pass ? 0 : 2;
} catch (const std::exception & e) {
    std::fprintf(stderr, "probe failed: %s\n", e.what());
    return 1;
}
