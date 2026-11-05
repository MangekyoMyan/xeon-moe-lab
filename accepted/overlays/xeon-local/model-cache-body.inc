    const char * iq3_env = std::getenv("GGML_CPU_IQ3_COMPACT_MIB");
    if (iq3_env && *iq3_env && std::strcmp(iq3_env, "0") != 0) {
        // Strict bounded decimal input. This is an extra RAM budget, not VRAM or KV cache.
        size_t mib = 0;
        for (const char * c = iq3_env; *c; ++c) {
            if (*c < '0' || *c > '9' || mib > 98304) throw std::runtime_error("IQ cache: invalid MiB budget");
            mib = mib*10+size_t(*c-'0');
        }
        if (mib < 1 || mib > 98304) throw std::runtime_error("IQ cache: budget must be 1..98304 MiB");
        pimpl->iq3_compact.init(mib*1024*1024);
        size_t count = 0, skipped = 0;
        LLAMA_LOG_WARN("IQ cache: preparing CPU single-token layout, budget %zu MiB; original weights retained\n", mib);
        bool low_memory = false;
        std::vector<const ggml_tensor *> candidates;
        for (const auto & layer : layers) {
            for (const auto * tensor : {layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
                if (!tensor || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || ml.lazy.has(tensor)) continue;
                if (tensor->type == GGML_TYPE_IQ3_XXS || tensor->type == GGML_TYPE_IQ2_S || tensor->type == GGML_TYPE_IQ3_S) candidates.push_back(tensor);
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const ggml_tensor * a, const ggml_tensor * b) {
            return xeon_iq_cache_priority(a->type, a->name) < xeon_iq_cache_priority(b->type, b->name);
        });
        for (const auto * tensor : candidates) {
            if (low_memory) { ++skipped; continue; }
            LLAMA_LOG_INFO("IQ cache: checking %s (%s)\n", tensor->name, ggml_type_name(tensor->type));
            const int status = pimpl->iq3_compact.add(pimpl->iq3_compact.owner, tensor);
            if (status == 1) {
                ++count;
                LLAMA_LOG_WARN("IQ cache: ready %s (%s); total %.3f GiB\n", tensor->name, ggml_type_name(tensor->type), pimpl->iq3_compact.used(pimpl->iq3_compact.owner)/1073741824.0);
            } else {
                ++skipped;
                if (status < 0) {
                    low_memory = true;
                    LLAMA_LOG_WARN("IQ cache: allocation/reserve limit; remaining tensors use original CPU path\n");
                }
            }
        }
        LLAMA_LOG_WARN("IQ cache: complete %zu tensors, %.3f GiB additional RAM, %zu fallback; decode only\n", count, pimpl->iq3_compact.used(pimpl->iq3_compact.owner)/1073741824.0, skipped);
    }
