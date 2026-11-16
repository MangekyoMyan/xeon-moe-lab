#pragma once
#include "ggml.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#ifdef __cplusplus
extern "C" {
#endif
// Test-only mode selection. Change only with no graph execution in flight.
void xeon_iq_abs_set_mode(int mode);
int xeon_iq_abs_get_mode(void);
void xeon_iq_abs_set_diagnostics(int enabled);
int xeon_iq_abs_diagnostics(void);
void xeon_iq_abs_counts(uint64_t out[3]);
void xeon_iq_abs_report(uint64_t prepared_blocks, uint64_t rows, uint64_t unsafe_ranges);
int xeon_iq_abs_prepare(int n, const void * input, int first, int end, void * output);
void xeon_iq2_abs_dot(int n, float * out, const void * weights, const void * input, const void * prepared);
void xeon_iq3s_abs_dot(int n, float * out, const void * weights, const void * input, const void * prepared);
#ifdef __cplusplus
}
#endif
struct xeon_iq_abs_state {
    int mode;
    int diagnostic;
    int64_t last_input;
    int safe;
    char * data;
    size_t stride;
    uint64_t safe_mask;
    uint64_t prepared_blocks, rows, unsafe_ranges;
};
static inline bool xeon_iq_abs_shape(const struct ggml_tensor * w, const struct ggml_tensor * x, const struct ggml_tensor * ids) {
    return (w->type == GGML_TYPE_IQ2_S || w->type == GGML_TYPE_IQ3_S) && ids->ne[1] == 1 &&
        x->ne[2] == 1 && x->ne[3] == 1 && x->ne[1] > 0 && x->ne[1] <= 64 &&
        x->ne[0] > 0 && x->ne[0] <= INT_MAX && x->ne[0] % 256 == 0 &&
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_Q8_K) && x->nb[0] == ggml_type_size(x->type);
}
static inline size_t xeon_iq_abs_bytes(int64_t k, int64_t inputs, int nth) {
    GGML_ASSERT(k > 0 && k % 256 == 0 && inputs > 0 && inputs <= 64 && nth > 0);
    size_t count = (size_t) inputs > (size_t) nth ? (size_t) inputs : (size_t) nth;
    GGML_ASSERT((size_t) nth <= SIZE_MAX / sizeof(uint64_t));
    size_t masks = (size_t) nth * sizeof(uint64_t);
    GGML_ASSERT((size_t) k <= (SIZE_MAX - masks - 63) / count);
    return (size_t) k * count + masks;
}
