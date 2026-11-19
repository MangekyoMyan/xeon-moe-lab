#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#ifdef __cplusplus
extern "C" {
#endif
// Each worker prepares disjoint complete Q8_0 blocks; caller publishes with a barrier.
// Returns1 for safe blocks,0 for -128/nonfinite scales. Even empty ranges return1.
int xeon_mxfp4_rhs_prepare_v1(int n,const void * input,int first,int end,void * sidecar);
// Dot requires the same n/input used for preparation, all worker ranges safe,
// and no concurrent mutation of the input or prepared sidecar. No ownership transfer.
void xeon_mxfp4_rhs_dot_v1(int n,float * output,const void * weights,const void * input,const void * sidecar);
#ifdef __cplusplus
}
#endif
static inline int xeon_mxfp4_rhs_shape(const struct ggml_tensor * w,const struct ggml_tensor * x,const struct ggml_tensor * ids) {
    return w->type==GGML_TYPE_MXFP4 && ids->ne[1]==1 && x->ne[2]==1 && x->ne[3]==1 &&
           x->ne[1]>0 && x->ne[1]<=64 && x->ne[0]>0 && x->ne[0]<=INT_MAX && x->ne[0]%32==0;
}
static inline size_t xeon_mxfp4_rhs_stride(int64_t n) {
    // Each32 input values use16bytes of signed correction and4bytes of FP32 scale.
    return (((size_t)n/32*20)+63)&~(size_t)63;
}
