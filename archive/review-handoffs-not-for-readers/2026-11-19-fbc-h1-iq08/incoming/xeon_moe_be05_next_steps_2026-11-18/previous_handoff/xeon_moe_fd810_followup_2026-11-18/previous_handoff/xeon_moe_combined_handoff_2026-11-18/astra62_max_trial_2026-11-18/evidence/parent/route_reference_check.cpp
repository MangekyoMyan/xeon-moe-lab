#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Case {
    int experts, slots, tokens, input_slots, threads;
    bool ids_transposed, input_strided, half_weights;
};

static int weight_value(int e, int m, int k) {
    return (e*11 + m*7 + k*3) % 13 - 6;
}
static int input_value(int k, int s, int t) {
    return (k*3 + s*5 + t*7) % 17 - 8;
}
static int expert_id(const Case & c, int s, int t, int phase) {
    if (phase == 1) return -1;
    if (phase == 2) return (s + 2*t) % 4 == 0 ? -1 : (c.experts - 1 - s + c.experts) % c.experts;
    if (phase == 3) return (t + 1) % c.experts;
    return (t*29 + s*53 + phase*17) % c.experts;
}
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void run_case(const Case & c, int index) {
    const int kdim = 64, mdim = 17;
    ggml_context * ctx = ggml_init({64u << 20, nullptr, false});
    require(ctx != nullptr, "context allocation");
    ggml_tensor * w = ggml_new_tensor_3d(ctx, c.half_weights ? GGML_TYPE_F16 : GGML_TYPE_F32, kdim, mdim, c.experts);
    for (int e=0;e<c.experts;++e) for (int m=0;m<mdim;++m) for (int k=0;k<kdim;++k) {
        char * dest = static_cast<char *>(w->data) + e*w->nb[2] + m*w->nb[1] + k*w->nb[0];
        float val = float(weight_value(e,m,k));
        if (c.half_weights) {
            ggml_fp16_t h = ggml_fp32_to_fp16(val);
            std::memcpy(dest, &h, sizeof(h));
        } else std::memcpy(dest, &val, sizeof(val));
    }
    ggml_tensor * x;
    if (c.input_strided) {
        ggml_tensor * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kdim, 2, c.input_slots, c.tokens);
        x = ggml_view_3d(ctx, storage, kdim, c.input_slots, c.tokens,
                        kdim*2*sizeof(float), kdim*2*c.input_slots*sizeof(float), kdim*sizeof(float));
    } else x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kdim, c.input_slots, c.tokens);
    for (int t=0;t<c.tokens;++t) for (int s=0;s<c.input_slots;++s) for (int k=0;k<kdim;++k) {
        float val = float(input_value(k,s,t));
        char * dest = static_cast<char *>(x->data) + t*x->nb[2] + s*x->nb[1] + k*x->nb[0];
        std::memcpy(dest, &val, sizeof(val));
    }
    ggml_tensor * ids = c.ids_transposed
        ? ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.tokens, c.slots))
        : ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.slots, c.tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_cplan plan = ggml_graph_plan(graph, c.threads, nullptr);
    void * storage = nullptr;
    require(posix_memalign(&storage, 64, plan.work_size + 128) == 0, "workspace allocation");
    auto * bytes = static_cast<uint8_t *>(storage);
    std::memset(bytes, 0xa5, plan.work_size + 128);
    plan.work_data = bytes + 64;
    uint64_t hash = UINT64_C(1469598103934665603);
    int64_t checked = 0;
    for (int phase=0;phase<5;++phase) {
        for (int t=0;t<c.tokens;++t) for (int s=0;s<c.slots;++s) {
            const int32_t id = expert_id(c,s,t,phase);
            std::memcpy(static_cast<char *>(ids->data) + s*ids->nb[0] + t*ids->nb[1], &id, sizeof(id));
        }
        std::fill_n(static_cast<float *>(out->data), ggml_nelements(out), std::numeric_limits<float>::quiet_NaN());
        require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "graph compute status");
        for (int t=0;t<c.tokens;++t) for (int s=0;s<c.slots;++s) for (int m=0;m<mdim;++m) {
            const int e = expert_id(c,s,t,phase);
            int expected = 0;
            if (e >= 0) for (int k=0;k<kdim;++k) expected += weight_value(e,m,k) * input_value(k,s%c.input_slots,t);
            float actual;
            std::memcpy(&actual, static_cast<const char *>(out->data) + m*out->nb[0] + s*out->nb[1] + t*out->nb[2], sizeof(actual));
            if (actual != float(expected)) {
                std::fprintf(stderr,"case=%d phase=%d token=%d slot=%d m=%d actual=%g expected=%d\n",index,phase,t,s,m,actual,expected);
                throw std::runtime_error("independent exact integer reference mismatch");
            }
            uint32_t bits;
            std::memcpy(&bits,&actual,sizeof(bits));
            for (int b=0;b<4;++b) { hash ^= (bits>>(8*b))&255u; hash *= UINT64_C(1099511628211); }
            ++checked;
        }
        for (int i=0;i<64;++i) {
            require(bytes[i] == 0xa5, "workspace prefix guard modified");
            require(bytes[64+plan.work_size+i] == 0xa5, "workspace suffix guard modified");
        }
    }
    std::printf("{\"case\":%d,\"E\":%d,\"slots\":%d,\"tokens\":%d,\"input_slots\":%d,\"threads\":%d,\"ids_transposed\":%s,\"input_strided\":%s,\"half_weights\":%s,\"workspace_bytes\":%zu,\"values_checked\":%lld,\"output_fnv64\":\"%016llx\",\"status\":\"pass\"}\n",
                index,c.experts,c.slots,c.tokens,c.input_slots,c.threads,c.ids_transposed?"true":"false",c.input_strided?"true":"false",c.half_weights?"true":"false",plan.work_size,(long long)checked,(unsigned long long)hash);
    std::free(storage);
    ggml_free(ctx);
}

int main() {
    ggml_cpu_init();
    const std::vector<Case> cases = {
        {1,1,1,1,1,false,false,false},
        {7,6,1,1,7,false,false,false},
        {7,6,1,2,7,false,true,false},
        {7,6,3,6,3,true,true,false},
        {384,6,1,1,7,true,true,false},
        {384,6,33,2,7,true,true,false},
        {384,6,1024,1,1,false,false,false},
        {384,6,1024,6,7,true,true,false},
        {1,1,1,1,1,false,false,true},
        {7,6,1,1,7,false,false,true},
        {7,6,1,2,7,false,true,true},
        {7,6,3,6,3,true,true,true},
        {384,6,1,1,7,true,true,true},
        {384,6,33,2,7,true,true,true},
        {384,6,1024,1,1,false,false,true},
        {384,6,1024,6,7,true,true,true},
    };
    try {
        for (size_t i=0;i<cases.size();++i) run_case(cases[i],int(i));
    } catch (const std::exception & e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
    return 0;
}
