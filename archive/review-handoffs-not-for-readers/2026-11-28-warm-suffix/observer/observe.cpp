// Diagnostic periods only. Forward all inference to the installed libraries.
// For a named synthetic request, record the real batches, assigned backend and
// one full vocabulary row when the server requests an output.
#include "llama.h"
#include "llama-batch.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "json.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
template<class T> static T next_symbol(const char *name) {
    auto fn = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
    if (!fn) std::abort();
    return fn;
}
static uint64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Placement {
    int tokens = 0;
    std::array<std::array<uint64_t,3>,2> mask{};
};
struct Call {
    std::vector<Placement> ubatches;
};
static thread_local Call *active = nullptr;
static thread_local Placement *current = nullptr;
static unsigned serial = 0;  // one server slot and one inference caller

static int projection(const ggml_tensor *t, int &layer) {
    if (t->op != GGML_OP_MUL_MAT_ID) return -1;
    const char *prefixes[] = {"ffn_moe_gate-", "ffn_moe_up-", "ffn_moe_down-"};
    for (int k = 0; k < 3; ++k) {
        const auto len = std::strlen(prefixes[k]);
        if (std::strncmp(t->name, prefixes[k], len) != 0) continue;
        char *end = nullptr;
        long n = std::strtol(t->name + len, &end, 10);
        if (end == t->name + len || *end || n < 0 || n >= 37) continue;
        layer = int(n);
        return k;
    }
    return -1;
}
extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s, ggml_cgraph *g) {
    static auto real = next_symbol<decltype(&ggml_backend_sched_graph_compute_async)>("ggml_backend_sched_graph_compute_async");
    if (!active) return real(s,g);
    active->ubatches.emplace_back();
    auto *prior = current;
    current = &active->ubatches.back();
    auto rc = real(s,g);
    current = prior;
    return rc;
}
extern "C" enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t b, ggml_cgraph *g) {
    static auto real = next_symbol<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");
    if (current) {
        const char *name = ggml_backend_name(b);
        int dev = !std::strcmp(name,"CPU") ? 0 : !std::strcmp(name,"CUDA0") ? 1 : -1;
        if (dev >= 0) {
            for (int i=0; i<ggml_graph_n_nodes(g); ++i) {
                auto *t = ggml_graph_node(g,i); int l=-1;
                const int k = projection(t,l);
                if (k < 0) continue;
                current->tokens = int(t->ne[2]);
                current->mask[dev][k] |= uint64_t(1) << l;
            }
        }
    }
    return real(b,g);
}
extern "C" int32_t llama_process(llama_context *ctx, enum llama_process_type type, llama_batch_ext *batch) {
    static auto real = next_symbol<decltype(&llama_process)>("llama_process");
    static const char *base = std::getenv("XEON_SUFFIX_RECORD");
    if (!base || !batch || type != LLAMA_PROCESS_TYPE_DECODE || active) return real(ctx,type,batch);
    std::string phase;
    std::ifstream phase_file(fs::path(base)/"phase");
    if (!(phase_file >> phase) || phase=="idle") return real(ctx,type,batch);
    if (phase.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos) std::abort();
    Call call;
    const auto begin=ns(); active=&call;
    const int32_t rc=real(ctx,type,batch);
    active=nullptr; const auto returned=ns();
    json j={{"phase",phase},{"seq",++serial},{"begin_ns",begin},{"submit_return_ns",returned},{"rc",rc}};
    std::vector<int> ids,positions;
    size_t outputs=0;
    for (const auto &t:batch->tokens) {
        ids.push_back(t.id);positions.push_back(t.pos[0]);outputs+=t.output;
        if (t.has_embd || t.seq_ids.size()!=1 || !t.seq_ids.count(0)) std::abort();
    }
    j["input_ids"]=ids; j["pos"]=positions; j["outputs"]=outputs;
    j["ubatches"]=json::array();
    for (const auto &u:call.ubatches) {
        j["ubatches"].push_back({{"tokens",u.tokens},{"CPU",u.mask[0]},{"CUDA0",u.mask[1]}});
    }
    if (rc==0 && outputs==1) {
        // This extra synchronization and file output occur only in B2/A2.
        const float *row=llama_get_logits_ith(ctx,-1);
        if (!row || llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)))!=129280) std::abort();
        j["logits_ready_ns"]=ns();
        std::string name=std::to_string(serial)+".f32";
        std::ofstream f(fs::path(base)/name,std::ios::binary);
        f.write(reinterpret_cast<const char*>(row),129280*4);
        f.close();if(!f)std::abort();
        j["logits_file"]=name;
    }
    std::ofstream f(fs::path(base)/"calls.jsonl",std::ios::app);
    f<<j.dump()<<'\n';f.close();if(!f)std::abort();
    return rc;
}
