#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

/* Only used for an untimed diagnostic process, never for performance runs. */
typedef void (*dot_fn)(int,float *,const void *,const void *,const void *);
static dot_fn next_dot;
static atomic_ulong calls;
__attribute__((constructor)) static void initialize(void) {
    next_dot=(dot_fn)dlsym(RTLD_NEXT,"ggml_vec_dot_mxfp4_q8_0_cols2");
    if (!next_dot) { fputs("paired symbol missing\n",stderr); exit(2); }
}
void ggml_vec_dot_mxfp4_q8_0_cols2(int n,float * s,const void * w,const void * x0,const void * x1) {
    atomic_fetch_add_explicit(&calls,1,memory_order_relaxed);
    next_dot(n,s,w,x0,x1);
}
__attribute__((destructor)) static void report(void) {
    fprintf(stderr,"{\"paired_kernel_calls\":%lu,\"diagnostic_only\":true}\n",atomic_load(&calls));
}
