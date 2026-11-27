// Test-only interposer for the exact DeepSeek41 two-layer, 512-text-token path.
// Native weights are streamed into the scheduler's original destination buffer.
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-model.h"
#include "llama-context.h"
#include "json.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/syscall.h>
#include <sched.h>
#include <unistd.h>
namespace fs=std::filesystem;
using json=nlohmann::ordered_json;
static constexpr int T=1024,K=6,D=5120,F=2304,E=384;
static void need(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
extern "C" void split_gather(const float*,const int32_t*,float*,int,int,int,cudaStream_t);
extern "C" void split_scatter(const float*,const int32_t*,float*,int,int,cudaStream_t);
#include "storage.h"
template<class V>static V original(const char *s){auto f=reinterpret_cast<V>(dlsym(RTLD_NEXT,s));if(!f)std::abort();return f;}
using Compute=decltype(&ggml_backend_graph_compute_async);
static Compute native_submit(){static auto f=original<Compute>("ggml_backend_graph_compute_async");return f;}
// Same non-owning slice as the fork's internal (non-exported) ggml_graph_view.
static ggml_cgraph slice(ggml_cgraph *g,int first,int last){
    need(first>=0 && first<last && last<=g->n_nodes,"valid graph slice");
    ggml_cgraph v{};v.n_nodes=last-first;v.nodes=g->nodes+first;v.use_counts=g->use_counts;
    v.visited_hash_set=g->visited_hash_set;v.order=g->order;return v;
}
static size_t gpu_free(){size_t avail,total;ck(cudaMemGetInfo(&avail,&total));return avail;}
static uint64_t hash_bytes(const void*p,size_t n){uint64_t v=14695981039346656037ULL;auto*b=(const uint8_t*)p;for(size_t i=0;i<n;++i)v=(v^b[i])*1099511628211ULL;return v;}
struct Packed{
    Context ctx;Buffer buf;int n;Layer weights;
    ggml_tensor *x,*ids,*out;ggml_cgraph *g;std::array<ggml_cgraph,3> phases;
    Packed(Layer &source,int count,ggml_backend_t gpu):n(count){
        weights.limit=source.limit;
        for(int j=0;j<3;++j){
            auto *w=source.w[j];auto *v=ggml_new_tensor_3d(ctx.p,w->type,w->ne[0],w->ne[1],w->ne[2]);
            // Prebound, unowned weight metadata. Each native split binds the real
            // scheduler staging destination before the descriptor can be consumed.
            v->buffer=w->buffer;v->data=w->data;weights.w[j]=v;
        }
        x=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,1,n);ids=ggml_new_tensor_2d(ctx.p,GGML_TYPE_I32,1,n);
        out=ffn(ctx,weights,x,ids);g=ggml_new_graph_custom(ctx.p,32,false);ggml_build_forward_expand(g,out);
        need(ggml_graph_n_nodes(g)==4 && ggml_graph_node(g,0)->src[0]==weights.w[0] &&
             ggml_graph_node(g,1)->src[0]==weights.w[1] && ggml_graph_node(g,2)->op==GGML_OP_GLU,"packed graph topology");
        buf.allocate(ctx,gpu);
        phases={slice(g,0,1),slice(g,1,3),slice(g,3,4)};
    }
    void bind(int j,ggml_tensor *dst){
        auto *v=weights.w[j];
        need(v->type==dst->type && ggml_are_same_shape(v,dst) && std::equal(v->nb,v->nb+4,dst->nb) &&
             dst->data && dst->buffer && !ggml_backend_buffer_is_host(dst->buffer) && !dst->view_src,"native staged weight layout");
        v->data=dst->data;v->buffer=dst->buffer;v->extra=dst->extra;
    }
};
struct Scratch{
    Layer w;std::unique_ptr<Packed> gpu;std::unique_ptr<Cpu> cpu;
};
struct Match{ggml_tensor *gate=nullptr,*up=nullptr,*act=nullptr,*down=nullptr;bool ok=false;};
struct Binding{ggml_backend_sched_copy_callback fn;void *context;};
static std::unordered_map<ggml_backend_sched_t,std::unique_ptr<Binding>>bindings;
struct Experiment {
    llama_context *ctx;ggml_backend_t cpu,gpu;ggml_threadpool_t pool;Worker worker;Stream stream;
    DeviceArray<int32_t> cs{E},gs{K*T};DeviceArray<float> relay{size_t(D)*E};
    std::array<int32_t,K*T> ids{};std::map<int,Scratch> scratch;std::map<int,Match> matches;
    std::unique_ptr<Partition> plan;int layer=-1,step=-1,pending=-1;int mode=0,mask=0;
    bool validate=false;json records=json::array(),team;
    explicit Experiment(llama_context*c,const llama_model*m):ctx(c){
        gpu=nullptr;auto sch=ctx->get_sched();for(int i=0;i<ggml_backend_sched_get_n_backends(sch);++i){
            auto b=ggml_backend_sched_get_backend(sch,i);if(std::string(ggml_backend_name(b))=="CUDA0")gpu=b;}
        need(gpu,"CUDA backend");cpu=ggml_backend_cpu_init();auto pp=ggml_threadpool_params_default(24);pp.poll=0;pp.strict_cpu=true;
        for(int i=0;i<24;++i)pp.cpumask[i]=true;
        pool=ggml_threadpool_new(&pp);need(cpu&&pool,"private CPU pool");
        ggml_backend_cpu_set_n_threads(cpu,24);ggml_backend_cpu_set_threadpool(cpu,pool);
        worker.submit([&]{team=verify_team(cpu);});worker.wait();
        for(int l:{4,24}){auto &src=m->layers.at(l);auto&w=scratch[l].w;
            w.w={src.ffn_gate_exps,src.ffn_up_exps,src.ffn_down_exps};w.limit=m->hparams.swiglu_clamp_exp[l];
            for(auto t:w.w)need(t&&ggml_backend_buffer_is_host(t->buffer),"original host weights");}
    }
    ~Experiment(){worker.wait();ggml_backend_synchronize(gpu);scratch.clear();ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);}
    static bool same_name(ggml_tensor *t,const std::string &name){return t&&name==t->name;}
    void inspect(ggml_cgraph *g){
        need(layer==-1,"prior layer drained");matches.clear();
        std::map<std::string,ggml_tensor*> named;std::map<const ggml_tensor*,unsigned> uses;
        for(int j=0;j<ggml_graph_n_nodes(g);++j){auto*t=ggml_graph_node(g,j);
            named[t->name]=t;for(auto*s:t->src)if(s)++uses[s];}
        for(int l:{4,24}){
            const auto n="-"+std::to_string(l);
            auto&a=matches[l];a.gate=named["ffn_moe_gate"+n];a.up=named["ffn_moe_up"+n];a.act=named["ffn_moe_swiglu_limited"+n];a.down=named["ffn_moe_down"+n];
            if(!a.gate||!a.up||!a.down||!a.act||a.gate->ne[2]!=T)continue;
            a.ok=a.gate->op==GGML_OP_MUL_MAT_ID&&a.up->op==GGML_OP_MUL_MAT_ID&&a.down->op==GGML_OP_MUL_MAT_ID&&
                a.act->op==GGML_OP_GLU&&ggml_get_glu_op(a.act)==GGML_GLU_OP_SWIGLU_CLAMP&&
                a.gate->ne[0]==F&&a.gate->ne[1]==K&&a.down->ne[0]==D&&a.down->ne[1]==K&&a.down->ne[2]==T&&
                a.gate->src[1]==a.up->src[1]&&a.gate->src[2]==a.up->src[2]&&a.gate->src[2]==a.down->src[2]&&
                a.act->src[0]==a.gate&&a.act->src[1]==a.up&&a.down->src[1]==a.act&&
                uses[a.gate]==1&&uses[a.up]==1&&uses[a.act]==1 &&
                !(a.gate->flags&GGML_TENSOR_FLAG_OUTPUT)&&!(a.up->flags&GGML_TENSOR_FLAG_OUTPUT)&&!(a.act->flags&GGML_TENSOR_FLAG_OUTPUT);
        }
    }
    bool copy(Binding&b,ggml_backend_t dev,const ggml_tensor *w,ggml_tensor *dst,ggml_cgraph *g){
        if(!mode||gpu!=dev)return false;
        auto *t=ggml_graph_n_nodes(g)?ggml_graph_node(g,0):nullptr;
        if(!t||t->op!=GGML_OP_MUL_MAT_ID||t->ne[2]!=T||t->src[0]!=dst)return false;
        int l=-1,j=-1;
        for(int k:{4,24}){
            if(!(mask&(k==4?1:2)))continue;
            auto &a=matches[k];if(!a.ok)continue;
            if(t==a.gate){l=k;j=0;}else if(t==a.up){l=k;j=1;}else if(t==a.down){l=k;j=2;}
        }
        if(j<0)return false;
        auto &s=scratch.at(l);auto &a=matches.at(l);
        need(w==s.w.w[j],"original tensor identity");
        if(j==0){
            need(layer==-1&&ggml_graph_n_nodes(g)==1,"gate split starts layer");
            auto *id=t->src[2],*x=t->src[1];
            need(id->ne[0]==K&&id->ne[1]==T&&id->nb[0]==4&&id->type==GGML_TYPE_I32&&
                 x->type==GGML_TYPE_F32&&ggml_is_contiguous(x)&&ggml_nelements(x)==D*T,"ready router inputs");
            ggml_backend_synchronize(gpu);
            ck(cudaMemcpy2DAsync(ids.data(),K*4,id->data,id->nb[1],K*4,T,cudaMemcpyDeviceToHost,stream.s));ck(cudaStreamSynchronize(stream.s));
            plan=std::make_unique<Partition>(ids.data(),mode!=1);
            if(plan->gs.empty()|| (mode!=1&&plan->cs.empty())){records.push_back({{"layer",l},{"cpu_pairs",0},{"gpu_pairs",T*K},{"native_fallback",true}});plan.reset();return false;}
            const int ng=plan->gs.size(),nc=plan->cs.size();need(nc<=E,"singleton bound");
            bool resize=!s.gpu||s.gpu->n!=ng;
            if(resize){s.gpu.reset();s.cpu.reset();s.gpu=std::make_unique<Packed>(s.w,ng,gpu);
                if(nc)s.cpu=std::make_unique<Cpu>(cpu,s.w,nc,1);}
            layer=l;step=0;records.push_back({{"layer",l},{"cpu_pairs",nc},{"gpu_pairs",ng},{"new_shape",resize},
                {"scratch_GPU_B",ggml_backend_buffer_get_size(s.gpu->buf.p)},{"free_gpu_B",gpu_free()}});
            if(validate){std::vector<float> xv(D*T);ggml_backend_tensor_get(x,xv.data(),0,xv.size()*4);
                records.back()["x_fnv"]=hash_bytes(xv.data(),xv.size()*4);records.back()["ids_fnv"]=hash_bytes(ids.data(),ids.size()*4);}
            if(nc){
                ck(cudaMemcpyAsync(cs.p,plan->cs.data(),nc*4,cudaMemcpyHostToDevice,stream.s));
                split_gather((float*)x->data,cs.p,relay.p,D,nc,K,stream.s);ck(cudaGetLastError());
                ck(cudaMemcpyAsync(s.cpu->x->data,relay.p,size_t(D)*nc*4,cudaMemcpyDeviceToHost,stream.s));ck(cudaStreamSynchronize(stream.s));
                std::memcpy(s.cpu->ids->data,plan->ci.data(),nc*4);
                auto *cpu_graph=s.cpu->g;worker.submit([this,cpu_graph]{need(ggml_backend_graph_compute(cpu,cpu_graph)==GGML_STATUS_SUCCESS,"CPU FFN");});
                if(mode==2)worker.wait();
            }
            ck(cudaMemcpyAsync(gs.p,plan->gs.data(),ng*4,cudaMemcpyHostToDevice,stream.s));
            ck(cudaMemcpyAsync(s.gpu->ids->data,plan->gi.data(),ng*4,cudaMemcpyHostToDevice,stream.s));
            split_gather((float*)x->data,gs.p,(float*)s.gpu->x->data,D,ng,K,stream.s);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream.s));
        }else if(layer==-1){return false;}
        need(layer==l&&step==j&&pending==-1,"ordered native split copies");
        need(j!=1||(ggml_graph_n_nodes(g)==2&&ggml_graph_node(g,1)==a.act),"up and clamp split");
        need(j!=2||(ggml_graph_n_nodes(g)>=2&&ggml_graph_node(g,1)->op==GGML_OP_MUL&&
                    ggml_graph_node(g,1)->src[0]==a.down),"down followed by original weighting");
        s.gpu->bind(j,dst);
        need(b.fn(dev,w,s.gpu->weights.w[j],&s.gpu->phases[j],b.context),"native registered-weight range callback");
        pending=j;return true;
    }
    bool compute(ggml_backend_t dev,ggml_cgraph *g,ggml_status&status){
        if(dev!=gpu||pending<0||!mode)return false;
        auto &a=matches.at(layer);auto &s=scratch.at(layer);
        auto first=ggml_graph_node(g,0);ggml_tensor *expected=step==0?a.gate:step==1?a.up:a.down;
        need(first==expected&&pending==step,"dispatch matches staged tensor");pending=-1;
        status=native_submit()(gpu,&s.gpu->phases[step]);if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("packed GPU projection");
        if(step<2){++step;return true;}
        ggml_backend_synchronize(gpu);
        split_scatter((float*)s.gpu->out->data,gs.p,(float*)a.down->data,D,plan->gs.size(),stream.s);ck(cudaGetLastError());
        if(!plan->cs.empty()){
            worker.wait();const int nc=plan->cs.size();
            ck(cudaMemcpyAsync(relay.p,s.cpu->down->data,size_t(D)*nc*4,cudaMemcpyHostToDevice,stream.s));
            split_scatter(relay.p,cs.p,(float*)a.down->data,D,nc,stream.s);ck(cudaGetLastError());
        }
        ck(cudaStreamSynchronize(stream.s));
        if(validate){std::vector<float> raw(size_t(D)*K*T);ggml_backend_tensor_get(a.down,raw.data(),0,raw.size()*4);
            records.back()["down_fnv"]=hash_bytes(raw.data(),raw.size()*4);}
        auto rest=slice(g,1,ggml_graph_n_nodes(g));
        status=native_submit()(gpu,&rest);layer=step=-1;plan.reset();return true;
    }
};
static Experiment *test=nullptr;
static thread_local bool reentrant=false;
struct InHook{InHook(){reentrant=true;}~InHook(){reentrant=false;}};
static bool dispatch_copy(ggml_backend_t dev,const ggml_tensor *src,ggml_tensor *dst,ggml_cgraph*g,void *user){
    auto &b=*static_cast<Binding*>(user);
    if(test&&!reentrant&&test->mode&&dev==test->gpu){InHook in;if(test->copy(b,dev,src,dst,g))return true;}
    return b.fn(dev,src,dst,g,b.context);
}
extern "C" void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t s,ggml_backend_sched_copy_callback cb,void*u){
    static auto fn=original<decltype(&ggml_backend_sched_set_copy_callback)>("ggml_backend_sched_set_copy_callback");
    if(!cb){fn(s,nullptr,u);return;}
    auto &b=bindings[s];if(!b)b=std::make_unique<Binding>();*b={cb,u};fn(s,dispatch_copy,b.get());
}
extern "C" bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t s,ggml_cgraph*g){
    static auto fn=original<decltype(&ggml_backend_sched_alloc_graph)>("ggml_backend_sched_alloc_graph");
    if(test&&!reentrant&&s==test->ctx->get_sched())test->inspect(g);
    return fn(s,g);
}
extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t b,ggml_cgraph*g){
    if(test&&!reentrant&&test->mode&&b==test->gpu&&test->pending>=0){
        InHook in;ggml_status ret; if(test->compute(b,g,ret))return ret;
    }
    return native_submit()(b,g);
}
extern "C" void pp512_live_begin(llama_context*c,const llama_model*m){need(!test,"one experiment");test=new Experiment(c,m);}
extern "C" void pp512_live_mode(int mode,int mask,bool validate){need(test&&test->layer==-1,"switch after drain");test->mode=mode;test->mask=mask;test->validate=validate;test->records=json::array();}
extern "C" const char *pp512_live_receipt(){static std::string out;need(test&&test->layer==-1&&test->pending==-1,"complete submitted layers");
    out=json({{"layers",test->records},{"CPU_team",test->team},{"free_gpu_B",gpu_free()},{"matched4",test->matches[4].ok},{"matched24",test->matches[24].ok}}).dump();return out.c_str();}
extern "C" void pp512_live_end(){auto *p=test;test=nullptr;delete p;}
