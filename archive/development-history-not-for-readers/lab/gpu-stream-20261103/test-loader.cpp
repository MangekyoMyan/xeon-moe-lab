#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    llama_backend_init();
    auto gpu=ggml_backend_cuda_init(0);
    if (!gpu) return 2;
    ggml_backend_dev_t devices[]={ggml_backend_get_device(gpu),nullptr};
    llama_model_tensor_buft_override overrides[]={
        {"blk\\.0\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_cpu_buffer_type()},
        {nullptr,nullptr}
    };
    for (int i=0;i<3;++i) {
        auto params=llama_model_default_params();
        params.devices=devices;
        params.n_gpu_layers=99;
        params.tensor_buft_overrides=overrides;
        params.load_mode=LLAMA_LOAD_MODE_MMAP;
        params.lazy_mode=LLAMA_LAZY_MODE_OFF;
        params.use_extra_bufts=false;
        auto model=llama_model_load_from_file(argv[1],params);
        if (!model) return 3;
        llama_model_free(model);
    }
    ggml_backend_free(gpu);
    llama_backend_free();
    std::puts("synthetic MoE model load/register/free repeated 3 times: PASS");
}
