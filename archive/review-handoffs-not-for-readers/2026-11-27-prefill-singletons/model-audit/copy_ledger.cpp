// Diagnostic only. Pass native callbacks and every upload through unchanged.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

struct Binding {ggml_backend_sched_copy_callback fn;void *data;};
static std::mutex lock;
static std::unordered_map<ggml_backend_sched_t,std::unique_ptr<Binding>> bindings;
static thread_local std::vector<std::pair<size_t,size_t>> *ranges=nullptr;
static thread_local std::string phase="startup";
extern "C" void pp512_ledger_label(const char *label){phase=label;}
static long long ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
template<class T>static T sym(const char *name){auto f=(T)dlsym(RTLD_NEXT,name);if(!f)std::abort();return f;}
static bool wrap(ggml_backend_t b,const ggml_tensor *src,ggml_tensor *dst,ggml_cgraph*g,void*p){
    const auto cb=*static_cast<Binding*>(p);int layer=-1;
    if(std::sscanf(src->name,"blk.%d.",&layer)!=1 || (layer!=4&&layer!=24) || !std::strstr(src->name,"_exps.weight"))
        return cb.fn(b,src,dst,g,cb.data);
    std::vector<std::pair<size_t,size_t>> calls;ranges=&calls;
    auto begin=ns();const bool handled=cb.fn(b,src,dst,g,cb.data);auto end=ns();ranges=nullptr;
    std::lock_guard<std::mutex> guard(lock);
    static FILE *f=std::fopen(std::getenv("PP512_COPY_LEDGER"),"wx");if(!f)std::abort();
    int n=ggml_graph_n_nodes(g);auto *first=n?ggml_graph_node(g,0):nullptr;
    std::fprintf(f,"%s\t%lld\t%lld\t%s\t%s\t%d\t%d\t%s\t%s\t",phase.c_str(),begin,end,src->name,ggml_backend_name(b),(int)handled,n,
        first?first->name:"",first?ggml_op_name(first->op):"");
    for(int i=0;i<n;++i){auto*t=ggml_graph_node(g,i);std::fprintf(f,"%s%s/%s",i?",":"",t->name,ggml_op_name(t->op));}
    std::fputc('\t',f);
    for(auto &c:calls)std::fprintf(f,"%zu+%zu,",c.first,c.second);
    std::fputc('\n',f);std::fflush(f);
    return handled;
}
extern "C" void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t s,ggml_backend_sched_copy_callback cb,void *data){
    static auto original=sym<decltype(&ggml_backend_sched_set_copy_callback)>("ggml_backend_sched_set_copy_callback");
    if(!cb){original(s,nullptr,data);return;}
    std::lock_guard<std::mutex> guard(lock);
    auto& p=bindings[s];if(!p)p=std::make_unique<Binding>();*p={cb,data};original(s,wrap,p.get());
}
extern "C" void ggml_backend_tensor_set_async(ggml_backend_t b,ggml_tensor*t,const void *data,size_t off,size_t bytes){
    static auto original=sym<decltype(&ggml_backend_tensor_set_async)>("ggml_backend_tensor_set_async");
    if(ranges)ranges->emplace_back(off,bytes);
    original(b,t,data,off,bytes);
}
