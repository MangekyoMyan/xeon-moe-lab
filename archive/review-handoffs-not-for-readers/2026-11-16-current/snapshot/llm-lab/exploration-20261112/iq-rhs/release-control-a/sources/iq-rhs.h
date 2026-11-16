#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#ifdef __cplusplus
extern "C" {
#endif
// Disjoint whole Q8_K blocks. Empty ranges return1; -128/nonfinite scale returns0.
int xeon_iq3_rhs_prepare_v1(int n,const void * input,int first,int end,void * correction);
// Requires compact IQ3_XXS weights, the same input/n, and all prepared ranges safe.
void xeon_iq3_rhs_dot_v1(int n,float * output,const void * compact_weights,const void * input,const void * correction);
#ifdef __cplusplus
}
#endif
static inline int xeon_iq3_rhs_shape(const struct ggml_tensor * w,const struct ggml_tensor * x,const struct ggml_tensor * ids) {
    return w->type==GGML_TYPE_IQ3_XXS && ids->ne[1]==1 && x->ne[2]==1 && x->ne[3]==1 &&
           x->ne[1]>0 && x->ne[1]<=64 && x->ne[0]>0 && x->ne[0]<=INT_MAX && x->ne[0]%256==0;
}
