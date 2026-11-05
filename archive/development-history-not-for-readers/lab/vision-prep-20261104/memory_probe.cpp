#include "mtmd.h"
#include "ggml-backend.h"
#include "llama.h"
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 3) return 1;
    const bool gpu = std::strcmp(argv[2], "gpu") == 0;
    ggml_backend_load_all();
    auto params = mtmd_context_params_default();
    params.use_gpu = gpu;
    params.warmup = true;
    params.n_threads = 1;
    params.image_max_tokens = 256;
    params.batch_max_tokens = 256;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    size_t free_bytes = 0, total_bytes = 0;
    if (gpu) {
        params.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!params.device) return 2;
        ggml_backend_dev_memory(params.device, &free_bytes, &total_bytes);
    }
    // The mtmd sizing API skips weight-file reads and reserves graph buffers.
    // No encoder inference and no text-model loading are performed.
    auto usage = mtmd_get_memory_usage(argv[1], params);
    if (usage.backend_mem_usage.empty()) return 3;
    size_t gpu_bytes = 0, cpu_bytes = 0;
    for (const auto & item : usage.backend_mem_usage) {
        if (ggml_backend_dev_type(item.first) == GGML_BACKEND_DEVICE_TYPE_GPU) gpu_bytes += item.second;
        else cpu_bytes += item.second;
    }
    std::printf("{\"gpu_bytes\":%zu,\"cpu_bytes\":%zu,\"image_max_tokens\":%d,\"weight_data_read\":false,\"gpu_total_bytes\":%zu,\"gpu_free_before_bytes\":%zu}\n",
                gpu_bytes, cpu_bytes, usage.image_max_tokens, total_bytes, free_bytes);
    return gpu && !gpu_bytes ? 4 : 0;
}
