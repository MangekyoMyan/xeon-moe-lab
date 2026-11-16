#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "iq3-cache.h"
#include "ggml-cpu/quants.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#include <cstdlib>

static size_t cases = 0, outputs = 0, page_queries = 0;
static void need(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

static void verify_pages(const void * pointer, int k, int m) {
    const char * mode = std::getenv("GGML_CPU_IQ_ROW_LOCAL");
    if (!mode || std::strcmp(mode,"1") || k!=5120 || m!=2304) return;
    constexpr size_t half=3317760;
    for (int expected=0;expected<2;++expected) for (size_t off : {size_t(0),size_t(4096),half/2,half-4096}) {
        void * address=const_cast<char *>(static_cast<const char *>(pointer))+expected*half+off;
        int node=-1;
        need(syscall(SYS_get_mempolicy,&node,nullptr,0,address,MPOL_F_ADDR|MPOL_F_NODE)==0,"NUMA page query failed");
        need(node==expected,"physical page on wrong node");
        unsigned long mask=0;int policy=-1;
        need(syscall(SYS_get_mempolicy,&policy,&mask,sizeof(mask)*8,address,MPOL_F_ADDR)==0,"NUMA policy query failed");
        need(policy==MPOL_BIND && mask==(1UL<<expected),"wrong bound VMA policy");
        ++page_queries;
    }
}

static void run(ggml_backend_t cpu, ggml_type type, int k, int m, int experts, int tokens,
                bool shared, bool strided, bool fp32, bool duplicate) {
    const int used = 6, slots = shared ? 1 : used;
    auto ctx = ggml_init({8*1024*1024,nullptr,true}); need(ctx,"ctx");
    auto w = ggml_new_tensor_3d(ctx,type,k,m,experts);
    ggml_set_name(w,k==5120?(shared?"blk.0.ffn_up_exps.weight":"blk.0.ffn_gate_exps.weight"):"blk.0.ffn_down_exps.weight");
    const auto xt = fp32 ? GGML_TYPE_F32 : GGML_TYPE_Q8_K;
    const size_t stride = ggml_row_size(xt,k);
    auto backing = ggml_new_tensor_3d(ctx,xt,k,strided?slots*2:slots,strided?tokens*2+1:tokens);
    auto x = strided ? ggml_view_3d(ctx,backing,k,slots,tokens,stride*2,stride*4*slots,stride) : backing;
    auto ids = ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);
    auto y = ggml_mul_mat_id(ctx,w,x,ids);
    auto graph = ggml_new_graph_custom(ctx,32,false); ggml_build_forward_expand(graph,y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx,cpu); need(buffer,"buffer");
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(8153+k+m+experts+tokens);
    std::vector<uint8_t> source(ggml_nbytes(w));
    for (auto & q : source) q = rng();
    for (size_t off = 0; off < source.size(); off += ggml_type_size(type)) {
        uint16_t scale = 0x1800 + 16*((off/ggml_type_size(type))%8);
        std::memcpy(source.data()+off,&scale,2);
    }
    ggml_backend_tensor_set(w,source.data(),0,source.size());
    std::vector<uint8_t> activations(ggml_nbytes(backing)); std::vector<float> floats(k);
    for (size_t off = 0; off < activations.size(); off += stride) {
        for (auto & value : floats) value = float(int(rng()%2001)-1000)/2048;
        if (fp32) std::memcpy(activations.data()+off,floats.data(),stride);
        else quantize_row_q8_K(floats.data(),activations.data()+off,k);
    }
    ggml_backend_tensor_set(backing,activations.data(),0,activations.size());
    std::vector<int32_t> routes(used*tokens);
    const int32_t chosen[6] = {experts-1,1,2,3,4,5};
    for (int t = 0; t < tokens; ++t) for (int i = 0; i < used; ++i) routes[t*used+i] = chosen[(i+t)%used];
    if (duplicate) for (int t = 0; t < tokens; ++t) { routes[t*used+1] = routes[t*used]; routes[t*used+5] = -1; }
    ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
    std::vector<float> baseline(ggml_nelements(y)), result(baseline.size());
    auto compute = [&](std::vector<float> & out) {
        need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"compute");
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
    };
    compute(baseline);
    const size_t per_expert = size_t(k/256)*m*144;
    const size_t full_bytes = (per_expert*experts+4095)/4096*4096;
    void * full = xeon_iq3_cache_create_v1(full_bytes,0,12); need(full,"full owner");
    need(xeon_iq3_cache_add_v1(full,w)==1,"v1 add");
    const auto * legacy = static_cast<const uint8_t *>(xeon_iq3_cache_lookup(w)); need(legacy,"v1 lookup");
    std::vector<uint8_t> packed(legacy,legacy+per_expert*experts);
    compute(result); need(std::memcmp(result.data(),baseline.data(),result.size()*4)==0,"v1 reference");
    xeon_iq3_cache_free_v1(full);
    std::vector<std::vector<uint32_t>> masks;
    for (int hit = 0; hit <= used; ++hit) {
        std::vector<uint32_t> selection;
        for (int i = hit-1; i >= 0; --i) selection.push_back(chosen[i]);
        masks.push_back(selection);
    }
    masks.push_back({0});
    masks.push_back({uint32_t(experts-1),0,uint32_t(experts/2)});
    std::vector<uint32_t> all(experts); std::iota(all.begin(),all.end(),0); masks.push_back(all);
    std::reverse(all.begin(),all.end()); masks.push_back(all);
    for (const auto & selected : masks) {
        const size_t bytes = (per_expert*selected.size()+4095)/4096*4096;
        void * owner = xeon_iq3_cache_create_v1(std::max(size_t(1),bytes),0,12); need(owner,"owner");
        const int added = xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size());
        need(added==(selected.empty()?0:1),"selected add");
        need(xeon_iq3_cache_used_v1(owner)==bytes,"sparse bytes");
        auto table = xeon_iq3_cache_lookup_experts_v2(w);
        need(bool(table)==!selected.empty(),"table presence");
        bool identity = selected.size()==size_t(experts);
        for (size_t i = 0; i < selected.size(); ++i) identity = identity && selected[i]==i;
        need(bool(xeon_iq3_cache_lookup(w))==identity,"legacy identity contract");
        for (int e = 0; e < experts; ++e) {
            const bool hit = std::find(selected.begin(),selected.end(),e)!=selected.end();
            const void * pointer = table ? table[e] : nullptr;
            need(bool(pointer)==hit,"expert table mapping");
            if (hit) { need(std::memcmp(pointer,packed.data()+e*per_expert,per_expert)==0,"wrong expert payload"); verify_pages(pointer,k,m); }
        }
        ggml_tensor wrong = *w; wrong.nb[2] += 16;
        need(!xeon_iq3_cache_lookup_experts_v2(&wrong),"wrong stride lookup");
        const auto hits = xeon_iq3_cache_hits_v1();
        compute(result);
        need(std::memcmp(result.data(),baseline.data(),result.size()*4)==0,"mixed graph bit mismatch");
        need((xeon_iq3_cache_hits_v1()>hits)==(tokens==1 && !selected.empty()),"single-token dispatch");
        const bool routed_hit = std::any_of(routes.begin(),routes.end(),[&](int32_t id) {
            return id >= 0 && std::find(selected.begin(),selected.end(),uint32_t(id)) != selected.end();
        });
        if (tokens == 1 && routed_hit) {
            // Synthetic fault injection proves cached rows execute; restore before release.
            auto altered = source;
            const size_t original_expert_bytes = source.size()/experts;
            for (uint32_t e : selected) for (size_t off = 0; off < original_expert_bytes; off += ggml_type_size(type)) {
                uint16_t zero = 0; std::memcpy(altered.data()+e*original_expert_bytes+off,&zero,2);
            }
            ggml_backend_tensor_set(w,altered.data(),0,altered.size());
            compute(result);
            need(std::memcmp(result.data(),baseline.data(),result.size()*4)==0,"cached dispatch bypassed");
            ggml_backend_tensor_set(w,source.data(),0,source.size());
        }
        if (!selected.empty()) {
            void * other = xeon_iq3_cache_create_v1(bytes,0,1); need(other,"second owner");
            need(xeon_iq3_cache_add_selected_v2(other,w,selected.data(),selected.size())==0,"duplicate source owner");
            auto second = source; ggml_tensor second_tensor = *w; second_tensor.data = second.data();
            need(xeon_iq3_cache_add_selected_v2(other,&second_tensor,selected.data(),selected.size())==1,"independent owner add");
            need(xeon_iq3_cache_total_v1()==2*bytes,"two owner accounting");
            xeon_iq3_cache_free_v1(other);
            need(!xeon_iq3_cache_lookup_experts_v2(&second_tensor)&&xeon_iq3_cache_lookup_experts_v2(w),"independent owner release");
        }
        outputs += result.size(); ++cases;
        xeon_iq3_cache_free_v1(owner);
        need(!xeon_iq3_cache_lookup_experts_v2(w),"stale table");
        need(xeon_iq3_cache_total_v1()==0,"global accounting");
        if (!selected.empty()) {
            owner = xeon_iq3_cache_create_v1(bytes-1,0,1);
            need(xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size())==0,"budget fallback");
            xeon_iq3_cache_free_v1(owner);
            owner = xeon_iq3_cache_create_v1(bytes,std::numeric_limits<size_t>::max(),1);
            need(xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size())==-1,"reserve fallback");
            xeon_iq3_cache_free_v1(owner);
        }
    }
    void * bad = xeon_iq3_cache_create_v1(full_bytes,0,1);
    uint32_t duplicate_ids[2] = {0,0}, outside = experts;
    need(xeon_iq3_cache_add_selected_v2(bad,w,nullptr,1)==0,"null selections");
    need(xeon_iq3_cache_add_selected_v2(bad,w,duplicate_ids,2)==0,"duplicate selections");
    need(xeon_iq3_cache_add_selected_v2(bad,w,&outside,1)==0,"out of range selection");
    need(!xeon_iq3_cache_lookup_experts_v2(w) && xeon_iq3_cache_used_v1(bad)==0,"failed add publication");
    xeon_iq3_cache_free_v1(bad);
    std::vector<uint8_t> unchanged(source.size()); ggml_backend_tensor_get(w,unchanged.data(),0,unchanged.size());
    need(unchanged==source,"original weights changed");
    compute(result); need(std::memcmp(result.data(),baseline.data(),result.size()*4)==0,"free fallback");
    uint16_t new_scale = 0x2400; std::memcpy(source.data(),&new_scale,2);
    ggml_backend_tensor_set(w,source.data(),0,source.size()); compute(baseline);
    uint32_t reload_ids[2] = {uint32_t(experts-1),0};
    const size_t reload_bytes = (2*per_expert+4095)/4096*4096;
    void * reload = xeon_iq3_cache_create_v1(reload_bytes,0,1);
    need(xeon_iq3_cache_add_selected_v2(reload,w,reload_ids,2)==1,"same-address reload");
    compute(result); need(std::memcmp(result.data(),baseline.data(),result.size()*4)==0,"stale same-address data");
    xeon_iq3_cache_free_v1(reload);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main() try {
    auto cpu = ggml_backend_cpu_init(); need(cpu,"CPU");
    for (int threads : {1,12}) {
        ggml_backend_cpu_set_n_threads(cpu,threads);
        auto params = ggml_threadpool_params_default(threads); params.poll=0; params.strict_cpu=true;
        for (int i=0;i<threads;++i) params.cpumask[i]=true;
        auto pool=ggml_threadpool_new(&params); ggml_backend_cpu_set_threadpool(cpu,pool);
        for (auto type : {GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S}) {
            run(cpu,type,5120,2304,7,1,true,false,true,false);
            run(cpu,type,5120,2304,7,1,false,true,false,true);
            run(cpu,type,5120,2304,7,3,true,true,true,true);
            run(cpu,type,2304,5120,7,1,false,false,true,false);
        }
        ggml_backend_cpu_set_threadpool(cpu,nullptr); ggml_threadpool_free(pool);
    }
    need(xeon_iq3_cache_total_v1()==0,"cache leak"); ggml_backend_free(cpu);
    setenv("GGML_CPU_IQ_ROW_LOCAL","bad",1);
    need(!xeon_iq3_cache_create_v1(4096,0,1),"invalid policy accepted");
    std::fprintf(stderr,"Verified %zu physical-page and VMA-policy queries\n",page_queries);
    std::printf("{\"cases\":%zu,\"outputs\":%zu,\"bit_mismatches\":0,\"pass\":true}\n",cases,outputs);
} catch(const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
