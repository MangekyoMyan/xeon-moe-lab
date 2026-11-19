// Quality-only, one-shot metadata observation. No eval callback or graph mutation.
#include "fbc-trace.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
using json=nlohmann::ordered_json;
namespace {
struct Pending {
    std::string path;
    std::unordered_map<const ggml_tensor*,std::string> ids;
    std::vector<ggml_tensor*> tensors;
    std::vector<ggml_tensor*> original_nodes;
    bool before=false;
};
thread_local Pending pending;
std::string identify(ggml_tensor *t){
    if(!t)return "null";
    auto it=pending.ids.find(t);if(it!=pending.ids.end())return it->second;
    std::string id="T"+std::to_string(pending.tensors.size());pending.ids[t]=id;pending.tensors.push_back(t);return id;
}
}
void xeon_fbc_trace_arm(const char *path){
    if(!path||!*path||!pending.path.empty())throw std::runtime_error("invalid pending FBC trace");
    pending=Pending{};pending.path=path;
}
bool xeon_fbc_trace_pending(){return !pending.path.empty();}
void xeon_fbc_trace_graph(ggml_cgraph *g,ggml_backend_sched_t sched,bool after){
    if(pending.path.empty())return;
    if(after!=pending.before)throw std::runtime_error("FBC trace phase mismatch");
    json j={{"phase",after?"post_allocation_optimization":"original_before_allocation"},{"nodes",json::array()},{"tensors",json::array()}};
    const int n=ggml_graph_n_nodes(g);
    if(!after)for(int i=0;i<n;++i){auto t=ggml_graph_node(g,i);if(pending.ids.count(t))throw std::runtime_error("duplicate original graph node");identify(t);pending.original_nodes.push_back(t);}
    for(int i=0;i<n;++i)j["nodes"].push_back(identify(ggml_graph_node(g,i)));
    for(size_t i=0;i<pending.tensors.size();++i){
        auto t=pending.tensors[i];json src=json::array();for(auto s:t->src)src.push_back(identify(s));
        const std::string view=identify(t->view_src);
        const std::string opname=ggml_op_name(t->op);
        json a={{"id",pending.ids.at(t)},{"name",t->name},{"op",opname},{"op_description",ggml_op_desc(t)},{"type",ggml_type_name(t->type)},
            {"ne",std::vector<int64_t>(t->ne,t->ne+4)},{"nb",std::vector<size_t>(t->nb,t->nb+4)},{"flags",t->flags},{"src",src},{"view_src",view},{"view_offset",t->view_offs}};
        if(opname.find("CUSTOM")==std::string::npos&&opname.find("MAP")==std::string::npos){
            const auto p=reinterpret_cast<const uint8_t*>(t->op_params);
            a["op_params_bytes"]=std::vector<uint8_t>(p,p+sizeof(t->op_params));
        }else a["op_params_bytes"]="pointer-bearing parameters omitted";
        if(after){
            // Query only original main nodes, already registered by allocation.
            // Copied source tensors can be unregistered: do not insert them by getter.
            bool original=i<pending.original_nodes.size();auto backend=original?ggml_backend_sched_get_tensor_backend(sched,t):nullptr;
            auto buffer=t->buffer?t->buffer:(t->view_src?t->view_src->buffer:nullptr);
            a["assigned_backend"]=backend?json(ggml_backend_name(backend)):json(nullptr);
            a["buffer_type"]=buffer?json(ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer))):json(nullptr);
            a["backend_query"]=original?"original_node":"not_queried";
        }
        j["tensors"].push_back(a);
    }
    if(after){
        j["scheduler_splits"]=ggml_backend_sched_get_n_splits(sched);
        j["scheduler_pipeline_copies"]=ggml_backend_sched_get_n_copies(sched);
        j["backend_buffers"]=json::array();for(int i=0;i<ggml_backend_sched_get_n_backends(sched);++i){auto b=ggml_backend_sched_get_backend(sched,i);j["backend_buffers"].push_back({{"name",ggml_backend_name(b)},{"bytes",ggml_backend_sched_get_buffer_size(sched,b)}});}
    }
    const std::string path=pending.path+(after?"-post.json":"-pre.json");
    if(std::filesystem::exists(path))throw std::runtime_error("FBC trace overwrite");
    std::ofstream f(path);f<<j.dump(2)<<'\n';f.close();if(!f)throw std::runtime_error("FBC trace write");
    if(after)pending=Pending{};else pending.before=true;
}
