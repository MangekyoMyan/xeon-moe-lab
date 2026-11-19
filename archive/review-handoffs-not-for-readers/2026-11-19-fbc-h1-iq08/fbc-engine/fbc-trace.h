#pragma once
#include "llama.h"
#include "ggml-backend.h"
LLAMA_API void xeon_fbc_trace_arm(const char * path);
LLAMA_API bool xeon_fbc_trace_pending();
void xeon_fbc_trace_graph(ggml_cgraph *g,ggml_backend_sched_t sched,bool after);
