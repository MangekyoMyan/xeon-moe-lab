#include "llama.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static void set_batch(llama_batch_ext * batch, const llama_token * tokens, int count, llama_pos position) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = position + i;
        llama_batch_ext_set_pos(batch, index, &pos);
    }
    llama_batch_ext_set_output_logits(batch, count - 1, true);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: deepseek-logits-probe MODEL.gguf OUTPUT.bin\n");
        return 1;
    }
    std::ofstream output(argv[2], std::ios::binary);
    if (!output) return 1;
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    model_params.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    std::string prompt;
    for (int i = 0; i < 30; ++i) {
        prompt += "The verification code is BLUE-731. Tokyo is the capital of Japan. Seventeen plus twenty-five equals forty-two. ";
    }
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (count < 160) return 3;
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) != count) return 3;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 4096;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_threads = 24;
    context_params.n_threads_batch = 24;
    auto * context = llama_init_from_model(model, context_params);
    if (!context) return 4;
    auto generation_params = ggml_threadpool_params_default(24);
    auto prompt_params = ggml_threadpool_params_default(24);
    generation_params.strict_cpu = prompt_params.strict_cpu = true;
    generation_params.paused = prompt_params.paused = true;
    for (int i = 0; i < 24; ++i) generation_params.cpumask[i] = prompt_params.cpumask[i] = true;
    auto * generation_pool = ggml_threadpool_new(&generation_params);
    auto * prompt_pool = ggml_threadpool_new(&prompt_params);
    llama_attach_threadpool(context, generation_pool, prompt_pool);
    auto * batch = llama_batch_ext_init(context);
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    const int32_t records = 34;
    const int32_t prefix = 127;
    output.write(reinterpret_cast<const char *>(&vocabulary_size), sizeof(vocabulary_size));
    output.write(reinterpret_cast<const char *>(&records), sizeof(records));
    output.write(reinterpret_cast<const char *>(&prefix), sizeof(prefix));
    output.write(reinterpret_cast<const char *>(tokens.data()), 160 * sizeof(llama_token));
    set_batch(batch, tokens.data(), prefix, 0);
    for (int record = 0; record < records; ++record) {
        if (llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch)) return 5;
        const float * logits = llama_get_logits_ith(context, -1);
        if (!logits) return 6;
        llama_token best = 0;
        for (int i = 0; i < vocabulary_size; ++i) {
            if (!std::isfinite(logits[i])) return 7;
            if (logits[i] > logits[best]) best = i;
        }
        const int32_t position = prefix - 1 + record;
        output.write(reinterpret_cast<const char *>(&position), sizeof(position));
        output.write(reinterpret_cast<const char *>(&best), sizeof(best));
        output.write(reinterpret_cast<const char *>(logits), vocabulary_size * sizeof(float));
        if (!output) return 8;
        std::printf("position=%d argmax=%d logits=%d\n", position, best, vocabulary_size);
        std::fflush(stdout);
        if (record + 1 < records) set_batch(batch, &tokens[prefix + record], 1, prefix + record);
    }
    llama_batch_ext_free(batch);
    llama_detach_threadpool(context);
    llama_free(context);
    ggml_threadpool_free(generation_pool);
    ggml_threadpool_free(prompt_pool);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
