#pragma once
#include "ggml.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// Private ABI v1. Owner must outlive all graphs that read its model weights.
void * xeon_iq3_cache_create_v1(size_t budget, size_t reserve, int threads);
int xeon_iq3_cache_add_v1(void * owner, const struct ggml_tensor * tensor);
void xeon_iq3_cache_free_v1(void * owner);
size_t xeon_iq3_cache_used_v1(const void * owner);
const void * xeon_iq3_cache_lookup(const struct ggml_tensor * tensor);
uint64_t xeon_iq3_cache_hits_v1(void);
size_t xeon_iq3_cache_total_v1(void);
void xeon_iq3_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
#ifdef __cplusplus
}
#endif
