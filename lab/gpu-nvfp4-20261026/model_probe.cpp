#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::atoi(argv[3]);
    mp.load_mtp = std::atoi(argv[2]) != 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    const std::string prompt = "The capital of Japan is Tokyo. Explain why computer memory bandwidth matters for language model inference in two sentences.";
    int nt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (nt <= 0 || nt > 256) return 4;
    std::vector<llama_token> tokens(nt);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), nt, true, true) != nt) return 5;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 256;
    cp.n_ubatch = 128;
    cp.n_threads = std::atoi(argv[4]);
    cp.n_threads_batch = cp.n_threads;
    cp.op_offload = mp.n_gpu_layers > 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 6;
    auto batch = llama_batch_get_one(tokens.data(), nt);
    const int nv = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < 32; ++step) {
        if (llama_decode(ctx, batch) != 0) return 7;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) return 8;
        for (int i = 0; i < nv; ++i) {
            if (!std::isfinite(logits[i])) {
                std::fprintf(stderr, "Non-finite logit: step=%d id=%d value=%g\n", step, i, logits[i]);
                return 9;
            }
        }
        if (step == 0) {
            std::ofstream out(argv[5], std::ios::binary);
            out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
            if (!out) return 10;
        }
        llama_token next = std::max_element(logits, logits + nv) - logits;
        std::printf("%d %d %.9g\n", step, next, logits[next]);
        std::fflush(stdout);
        if (llama_vocab_is_eog(vocab, next)) break;
        tokens.assign(1, next);
        batch = llama_batch_get_one(tokens.data(), 1);
    }
    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
