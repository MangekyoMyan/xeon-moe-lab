#include "ggml.h"
#include "ggml-impl.h"
#include "llama-batch.h"
#include <cstddef>
#include <cstdio>
int main() {
    static_assert(offsetof(ggml_tensor,name)==256);
    static_assert(offsetof(ggml_cgraph,n_nodes)==4);
    static_assert(offsetof(ggml_cgraph,nodes)==16);
    static_assert(offsetof(llama_ubatch,n_tokens)==4);
    std::puts("Validated graph/tensor/ubatch offsets against production headers");
}
