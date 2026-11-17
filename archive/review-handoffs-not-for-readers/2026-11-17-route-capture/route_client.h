#pragma once
#include "route37.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "json.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
using json=nlohmann::ordered_json;
inline void need(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
struct API {
    decltype(&xeon_route37_open_v1) open; decltype(&xeon_route37_label_v1) label;
    decltype(&xeon_route37_begin_eval_v1) begin; decltype(&xeon_route37_end_eval_v1) end;
    decltype(&xeon_route37_complete_call_v1) finish; decltype(&xeon_route37_close_v1) close;
    decltype(&xeon_route37_record_v1) record; decltype(&xeon_route37_stats_v1) stats;
    decltype(&xeon_route37_read_record_v1) read; decltype(&xeon_route37_read_attempt_v1) attempt;
    decltype(&xeon_route37_read_call_v1) call;
    API(){
#define XR_LOAD(m,n) m=reinterpret_cast<decltype(m)>(dlsym(RTLD_DEFAULT,n));need(m,n)
        XR_LOAD(open,"xeon_route37_open_v1");XR_LOAD(label,"xeon_route37_label_v1");XR_LOAD(begin,"xeon_route37_begin_eval_v1");XR_LOAD(end,"xeon_route37_end_eval_v1");
        XR_LOAD(finish,"xeon_route37_complete_call_v1");XR_LOAD(close,"xeon_route37_close_v1");XR_LOAD(record,"xeon_route37_record_v1");XR_LOAD(stats,"xeon_route37_stats_v1");
        XR_LOAD(read,"xeon_route37_read_record_v1");XR_LOAD(attempt,"xeon_route37_read_attempt_v1");XR_LOAD(call,"xeon_route37_read_call_v1");
#undef XR_LOAD
    }
};
inline json providers(){
    json result=json::array();
    for(const char *name:{"llama_decode","llama_get_logits_ith","ggml_backend_cpu_init","ggml_backend_sched_graph_compute_async",
        "xeon_route37_open_v1","xeon_route37_matches_v1","xeon_route37_label_v1","xeon_route37_begin_eval_v1","xeon_route37_end_eval_v1","xeon_route37_complete_call_v1","xeon_route37_close_v1","xeon_route37_record_v1","xeon_route37_stats_v1","xeon_route37_read_record_v1","xeon_route37_read_attempt_v1","xeon_route37_read_call_v1"}){
        void *fn=dlsym(RTLD_DEFAULT,name);Dl_info info{};need(fn&&dladdr(fn,&info)&&info.dli_fname,"provider lookup failed");
        auto p=std::filesystem::canonical(info.dli_fname);struct stat st{};need(!stat(p.c_str(),&st),"provider stat");
        result.push_back({{"symbol",name},{"path",p.string()},{"device",uint64_t(st.st_dev)},{"inode",uint64_t(st.st_ino)},{"bytes",uint64_t(st.st_size)}});
    }
    return result;
}
inline json label_json(const xeon_route37_label &x){return {{"request",x.request},{"sequence",x.sequence},{"decode_call",x.decode_call},{"phase",x.phase},{"logical_input_token",x.logical_token}};}
inline json snapshot(API &api){
    xeon_route37_stats s{};need(api.stats(&s),"collector not readable; an eval/call remains open");
    json out={{"stats",{{"records",s.records},{"attempts",s.attempts},{"calls",s.calls},{"dropped_events",s.dropped},{"complete_decode_attempts",s.complete_decode_attempts},{"flags",s.flags},{"enabled",bool(s.enabled)},{"observing",bool(s.observing)}}},{"records",json::array()},{"attempts",json::array()},{"calls",json::array()}};
    for(size_t i=0;i<s.records;++i){xeon_route37_record x{};need(api.read(i,&x),"record read");out["records"].push_back({{"ordinal",x.ordinal},{"attempt_id",x.attempt},{"request",x.request},{"sequence",x.sequence},{"decode_call",x.decode_call},{"ubatch",x.ubatch},{"logical_input_token",x.logical_token},{"position",x.position},{"generation",x.graph_generation},{"context",x.context},{"layer",x.layer},{"phase",x.phase},{"projection",x.projection},{"n_tokens",x.n_tokens},{"ids",std::vector<int32_t>(x.ids,x.ids+6)},{"flags",x.flags}});}
    for(size_t i=0;i<s.attempts;++i){xeon_route37_attempt x{};need(api.attempt(i,&x),"attempt read");out["attempts"].push_back({{"label",label_json(x.label)},{"id",x.id},{"generation",x.generation},{"first_record",x.first_record},{"records",x.records},{"layers",x.layers},{"position",x.position},{"context",x.context},{"ubatch",x.ubatch},{"n_tokens",x.n_tokens},{"flags",x.flags},{"graph_status",x.graph_status},{"observing",bool(x.observing)},{"scope_t1",bool(x.scope_t1)},{"graph_returned",bool(x.graph_returned)},{"caller_completed",bool(x.caller_completed)}});}
    for(size_t i=0;i<s.calls;++i){xeon_route37_call x{};need(api.call(i,&x),"call read");out["calls"].push_back({{"label",label_json(x.label)},{"first_attempt",x.first_attempt},{"attempts",x.attempts},{"flags",x.flags},{"public_or_driver_status",x.status},{"synchronized",bool(x.synchronized)},{"completed",bool(x.completed)}});}
    return out;
}
inline void save_json(const std::filesystem::path &path,const json &data){
    auto temp=path;temp+=".tmp";std::ofstream f(temp);need(bool(f),"output open");f<<data.dump(2)<<'\n';f.close();need(!f.fail(),"output write");std::filesystem::rename(temp,path);
}
