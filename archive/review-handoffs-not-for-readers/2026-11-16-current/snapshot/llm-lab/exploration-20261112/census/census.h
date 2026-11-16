#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif
void xeon_moe_census_v1(const struct ggml_tensor * weights, const struct ggml_tensor * ids, const int64_t * counts, int64_t experts);
#ifdef __cplusplus
}
#endif
