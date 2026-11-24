#pragma once
#include "ggml.h"
#include "ggml-backend.h"
struct llama_model;
// Experimental, context-owned backend. No public llama class layout changes.
ggml_backend_t xeon_hot_backend_create(const llama_model & model, ggml_backend_t outer_cpu,
                                     ggml_backend_t outer_gpu, int threads, int batch_threads);
void xeon_hot_backend_finish(ggml_backend_t outer_cpu);
ggml_tensor * xeon_hot_build(ggml_backend_t outer_cpu, ggml_backend_sched_t sched,
                            ggml_context * ctx, ggml_tensor * input, ggml_tensor * ids,
                            ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
                            float clamp, int layer);
void xeon_hot_place_tail(ggml_backend_t outer_cpu, ggml_backend_sched_t sched, ggml_tensor * tail);

#include <cstdint>
struct xeon_hot_trial_stats {uint64_t builds=0,payload=0,allocation=0,calls[40]{},hist[40][7]{};char mode=0;bool ready=false;};
int xeon_hot_trial_set_mode(ggml_backend_t cpu,char mode);
int xeon_hot_trial_snapshot(ggml_backend_t cpu,xeon_hot_trial_stats *out);
