// Experimental fixed expert residency. Original weights remain CPU-owned.
// The special host backend prevents a nested OpenMP team around the private CPU pool.
#include "hot-backend.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace {
constexpr int D=5120,F=2304,E=384,TOP=6,LAYERS=40;
constexpr size_t GiB=size_t(1024)*1024*1024;
void require(bool ok,const char * msg){if(!ok)throw std::runtime_error(std::string("xeon-hot: ")+msg);}
void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("xeon-hot CUDA: ")+cudaGetErrorString(e));}
struct SampleStamp {
    int64_t begin=0, partition_end=0, cpu_set_end=0, submit_end=0;
    int64_t cpu_begin=0, cpu_end=0, join_begin=0, join_end=0, end=0;
    int matched=0,executed=0,gpu_ops=0;
};
int64_t hot_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct State;
struct Graph{
    ggml_context * ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_backend_t backend=nullptr;
    ggml_tensor * x=nullptr,* ids=nullptr,* y=nullptr;ggml_cgraph * graph=nullptr;
    Graph(ggml_backend_t b,const std::array<ggml_tensor*,3>&w,int slots,float clamp):backend(b){try{
        ctx=ggml_init({512*1024,nullptr,true});require(ctx,"private graph metadata");
        x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,1);ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,slots,1);
        auto up=ggml_mul_mat_id(ctx,w[1],x,ids);auto gate=ggml_mul_mat_id(ctx,w[0],x,ids);
        auto act=ggml_swiglu_clamp(ctx,gate,up,clamp);y=ggml_mul_mat_id(ctx,w[2],act,ids);
        graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"private unsupported op");
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"private activation allocation");ggml_backend_buffer_clear(buffer,0);
    }catch(...){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);throw;}}
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
struct Layer{
    State * owner=nullptr;int index=0;float clamp=0;bool eligible=false;
    std::array<ggml_tensor*,3> original{},gpu_weights{};
    std::vector<uint32_t> selected;std::array<int,E>local{};
    ggml_context * weight_ctx=nullptr;ggml_backend_buffer_t weight_buffer=nullptr;
    std::unique_ptr<Graph>cpu;std::array<std::unique_ptr<Graph>,TOP+1>gpu;
    uint64_t calls=0;std::array<uint64_t,TOP+1>hist{};
    Layer(){local.fill(-1);}
    ~Layer(){for(auto & g:gpu)g.reset();cpu.reset();if(weight_buffer)ggml_backend_buffer_free(weight_buffer);if(weight_ctx)ggml_free(weight_ctx);}
};
std::mutex registry_mutex;
std::map<ggml_backend_t,State*> registry;
void marker(ggml_tensor *,int,int,void *){GGML_ABORT("xeon-hot custom operation was scheduled on the ordinary CPU backend");}
bool marked(const ggml_tensor * op){
    if(op->op!=GGML_OP_CUSTOM)return false;
    ggml_custom_op_params p{};std::memcpy(&p,op->op_params,sizeof(p));return p.fun==marker&&p.n_tasks==1&&p.userdata;
}
bool cpu_weight(const ggml_tensor * w){return w&&w->buffer&&ggml_backend_buffer_is_host(w->buffer)&&ggml_is_contiguous(w);}
bool supported(ggml_type t){return t==GGML_TYPE_IQ2_S||t==GGML_TYPE_IQ3_XXS||t==GGML_TYPE_IQ3_S||t==GGML_TYPE_MXFP4;}
struct State{
    const llama_model * model=nullptr;ggml_backend_t outer_cpu=nullptr,outer_gpu=nullptr,self=nullptr;
    ggml_backend_t private_cpu=nullptr,private_gpu=nullptr;ggml_threadpool_t pool=nullptr;
    std::array<std::unique_ptr<Layer>,LAYERS> layers;
    float * pinned_input=nullptr,*pinned_output=nullptr;int32_t * pinned_ids=nullptr;
    std::mutex execution_mutex;bool ready=false;size_t payload=0,allocation=0,planned_count=0;
    explicit State(const llama_model * m):model(m){}
    ~State(){
        if(private_gpu)ggml_backend_synchronize(private_gpu);
        for(auto & layer:layers)if(layer){
            if(layer->calls)LLAMA_LOG_INFO("xeon-hot-stats layer=%d calls=%llu H=[%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",layer->index,(unsigned long long)layer->calls,
                (unsigned long long)layer->hist[0],(unsigned long long)layer->hist[1],(unsigned long long)layer->hist[2],(unsigned long long)layer->hist[3],(unsigned long long)layer->hist[4],(unsigned long long)layer->hist[5],(unsigned long long)layer->hist[6]);
            layer.reset();
        }
        if(pinned_ids)cudaFreeHost(pinned_ids);
        if(pinned_output)cudaFreeHost(pinned_output);
        if(pinned_input)cudaFreeHost(pinned_input);
        if(private_cpu)ggml_backend_cpu_set_threadpool(private_cpu,nullptr);
        if(pool)ggml_threadpool_free(pool);
        if(private_gpu)ggml_backend_free(private_gpu);
        if(private_cpu)ggml_backend_free(private_cpu);
        std::lock_guard<std::mutex>lock(registry_mutex);auto it=registry.find(outer_cpu);if(it!=registry.end()&&it->second==this)registry.erase(it);
    }
    void read_plan(const char * path){
        require(model,"model required for plan validation");
        std::ifstream f(path);require(bool(f),"open hot plan");std::string magic;int nl,ne,d,ff;size_t count,bytes;
        require(bool(f>>magic>>nl>>ne>>d>>ff>>count>>bytes),"plan header");
        require(magic=="XEON_HOT_FFN_PLAN_V1"&&nl==LAYERS&&ne==E&&d==D&&ff==F&&count<=size_t(LAYERS*E)&&bytes<=16*GiB,"plan dimensions/budget");
        size_t actual_count=0,actual_bytes=0;
        for(int i=0;i<LAYERS;++i){
            int index,n;std::array<std::string,3>types;
            require(bool(f>>index>>types[0]>>types[1]>>types[2]>>n)&&index==i&&n>=0&&n<=E,"plan layer");
            auto layer=std::make_unique<Layer>();layer->owner=this;layer->index=i;layer->clamp=model->hparams.swiglu_clamp_exp[i];
            const auto & m=model->layers.at(i);layer->original={m.ffn_gate_exps,m.ffn_up_exps,m.ffn_down_exps};
            size_t expert_bytes=0;bool eligible=true;
            for(int j=0;j<3;++j){auto w=layer->original[j];require(w&&supported(w->type),"plan weight type");
                std::string name=ggml_type_name(w->type);std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::toupper(c));});
                require(types[j]==name&&w->ne[0]==(j==2?F:D)&&w->ne[1]==(j==2?D:F)&&w->ne[2]==E&&w->ne[3]==1,"plan tensor mismatch");
                eligible=eligible&&cpu_weight(w);expert_bytes+=ggml_row_size(w->type,w->ne[0])*w->ne[1];
            }
            require(std::isfinite(layer->clamp)&&layer->clamp>1e-6f,"activation clamp");
            layer->eligible=eligible;require(eligible||n==0,"selected expert weights must be CPU resident");
            int previous=-1;for(int j=0;j<n;++j){int e;require(bool(f>>e)&&e>previous&&e<E,"sorted unique expert IDs");previous=e;layer->selected.push_back(e);layer->local[e]=j;}
            actual_count+=n;actual_bytes+=n*expert_bytes;layers[i]=std::move(layer);
        }
        std::string extra;require(!(f>>extra),"trailing plan data");require(actual_count==count&&actual_bytes==bytes,"plan payload mismatch");
        payload=bytes;planned_count=count;
    }
    void initialize(){
        require(!ready,"duplicate initialization");
        private_cpu=ggml_backend_cpu_init();require(private_cpu,"private CPU backend");
        auto p=ggml_threadpool_params_default(12);p.poll=0;p.strict_cpu=true;for(int i=0;i<12;++i)p.cpumask[i]=true;
        pool=ggml_threadpool_new(&p);require(pool,"private CPU pool");ggml_backend_cpu_set_n_threads(private_cpu,12);ggml_backend_cpu_set_threadpool(private_cpu,pool);
        if(payload){
            cuda_check(cudaSetDevice(0));cudaDeviceProp prop{};cuda_check(cudaGetDeviceProperties(&prop,0));require(prop.major==7&&prop.minor==0,"Volta device required");
            size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
            LLAMA_LOG_INFO("capacity-bank-Gpre free=%zu total=%zu payload=%zu margin=%zu\n",free,total,payload,size_t(3072)*1024*1024);
            // Include room for new execution allocations and subsequent vision/pool growth.
            require(free>payload+size_t(3072)*1024*1024,"insufficient GPU headroom for plan and later vision/pool growth");
            private_gpu=ggml_backend_cuda_init(0);require(private_gpu,"private CUDA backend");
            cuda_check(cudaHostAlloc(&pinned_input,D*sizeof(float),cudaHostAllocPortable));
            cuda_check(cudaHostAlloc(&pinned_output,D*TOP*sizeof(float),cudaHostAllocPortable));
            cuda_check(cudaHostAlloc(&pinned_ids,TOP*sizeof(int32_t),cudaHostAllocPortable));
        }
        for(auto & ptr:layers){if(!ptr)continue;auto & l=*ptr;if(!l.eligible||l.selected.empty())continue;
            l.cpu=std::make_unique<Graph>(private_cpu,l.original,TOP,l.clamp);
            std::array<float,D> zero{};std::array<int32_t,TOP> disabled;disabled.fill(-1);
            ggml_backend_tensor_set(l.cpu->x,zero.data(),0,sizeof(zero));ggml_backend_tensor_set(l.cpu->ids,disabled.data(),0,sizeof(disabled));
            require(ggml_backend_graph_compute(private_cpu,l.cpu->graph)==GGML_STATUS_SUCCESS,"private CPU warmup");
            l.weight_ctx=ggml_init({512*1024,nullptr,true});require(l.weight_ctx,"bank metadata");
            for(int j=0;j<3;++j){auto w=l.original[j];l.gpu_weights[j]=ggml_new_tensor_3d(l.weight_ctx,w->type,w->ne[0],w->ne[1],l.selected.size());}
            l.weight_buffer=ggml_backend_alloc_ctx_tensors(l.weight_ctx,private_gpu);require(l.weight_buffer,"bank allocation");
            allocation+=ggml_backend_buffer_get_size(l.weight_buffer);ggml_backend_buffer_set_usage(l.weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_buffer_clear(l.weight_buffer,0);
            for(int j=0;j<3;++j){auto w=l.original[j];size_t n=ggml_row_size(w->type,w->ne[0])*w->ne[1];
                for(size_t local=0;local<l.selected.size();++local)ggml_backend_tensor_set(l.gpu_weights[j],static_cast<const char*>(w->data)+size_t(l.selected[local])*w->nb[2],local*n,n);
            }
            // Reserve and exercise each reachable H, including kernel/pool/graph setup.
            for(int h=1;h<=std::min(TOP,int(l.selected.size()));++h){
                l.gpu[h]=std::make_unique<Graph>(private_gpu,l.gpu_weights,h,l.clamp);
                std::array<int32_t,TOP> local_ids{};for(int j=0;j<h;++j)local_ids[j]=j;
                ggml_backend_tensor_set(l.gpu[h]->x,zero.data(),0,sizeof(zero));ggml_backend_tensor_set(l.gpu[h]->ids,local_ids.data(),0,h*sizeof(int32_t));
                for(int repeat=0;repeat<3;++repeat)require(ggml_backend_graph_compute(private_gpu,l.gpu[h]->graph)==GGML_STATUS_SUCCESS,"private GPU warmup");
                std::vector<float> checked(size_t(D)*h);ggml_backend_tensor_get(l.gpu[h]->y,checked.data(),0,checked.size()*sizeof(float));
                for(float value:checked)require(std::isfinite(value)&&value==0.0f,"zero-input private graph output");
                LLAMA_LOG_INFO("capacity-bank-warm layer=%d H=%d repeats=3 zero_output=1\n",l.index,h);
            }
        }
        if(private_gpu){ggml_backend_synchronize(private_gpu);size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));LLAMA_LOG_INFO("capacity-bank-Gpost free=%zu total=%zu margin=%zu\n",free,total,size_t(2560)*1024*1024);require(free>size_t(2560)*1024*1024,"post-allocation reserve for vision/pool growth");}
        size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));
        ready=true;
        LLAMA_LOG_INFO("xeon-hot-ready experts=%zu native_payload=%zu GPU_bank_allocation=%zu GPU_free_before_vision=%zu CPUthreads=12 original-weights-retained=1\n",planned_count,payload,allocation,free);
    }
    void compute(Layer & l,ggml_tensor * dst,char mode='P',SampleStamp * stamp=nullptr){
        require(mode=='C'||mode=='S'||mode=='P',"replay mode");
        require(ready&&l.eligible&&l.cpu,"uninitialized layer");
        const auto x=dst->src[0],ids=dst->src[1];
        require(x&&ids&&x->type==GGML_TYPE_F32&&ids->type==GGML_TYPE_I32&&x->ne[0]==D&&x->ne[1]==1&&x->ne[2]==1&&x->ne[3]==1&&ggml_is_contiguous(x)&&ids->ne[0]==TOP&&ids->ne[1]==1&&ids->ne[2]==1&&ids->ne[3]==1&&dst->type==GGML_TYPE_F32&&dst->ne[0]==D&&dst->ne[1]==TOP&&dst->ne[2]==1&&dst->ne[3]==1&&ggml_is_contiguous(dst),"runtime tensor geometry");
        auto host=[](const ggml_tensor * t){auto b=t->buffer?t->buffer:(t->view_src?t->view_src->buffer:nullptr);return b&&ggml_backend_buffer_is_host(b);};
        require(host(x)&&host(ids)&&host(dst),"dedicated backend requires host input/output buffers");
        require(x->data&&ids->data&&dst->data,"runtime tensor data");
        std::array<int32_t,TOP>cpu_ids{};std::array<int,TOP>hot_slots{};std::array<bool,E>seen{};int hot=0;
        for(int slot=0;slot<TOP;++slot){int32_t e;std::memcpy(&e,static_cast<const char*>(ids->data)+slot*ids->nb[0],sizeof(e));
            require(e>=0&&e<E&&!seen[e],"invalid routing ID");seen[e]=true;
            if(l.local[e]>=0){hot_slots[hot]=slot;pinned_ids[hot]=l.local[e];++hot;cpu_ids[slot]=-1;}else cpu_ids[slot]=e;
        }
        if(stamp){stamp->matched=hot;stamp->partition_end=hot_ns();}
        if(mode=='C'){hot=0;std::memcpy(cpu_ids.data(),ids->data,TOP*sizeof(int32_t));}
        if(stamp)stamp->executed=hot;
        if(hot<TOP){ggml_backend_tensor_set(l.cpu->x,x->data,0,D*sizeof(float));ggml_backend_tensor_set(l.cpu->ids,cpu_ids.data(),0,TOP*sizeof(int32_t));}
        if(stamp)stamp->cpu_set_end=hot_ns();
        if(hot){auto & g=*l.gpu[hot];std::memcpy(pinned_input,x->data,D*sizeof(float));
            ggml_backend_tensor_set_async(private_gpu,g.x,pinned_input,0,D*sizeof(float));ggml_backend_tensor_set_async(private_gpu,g.ids,pinned_ids,0,hot*sizeof(int32_t));
            require(ggml_backend_graph_compute_async(private_gpu,g.graph)==GGML_STATUS_SUCCESS,"GPU graph failed");
            ggml_backend_tensor_get_async(private_gpu,g.y,pinned_output,0,size_t(hot)*D*sizeof(float));
        }
        if(stamp){stamp->submit_end=hot_ns();stamp->gpu_ops=hot?4:0;}
        auto join=[&](){if(stamp)stamp->join_begin=hot_ns();ggml_backend_synchronize(private_gpu);if(stamp){stamp->join_end=hot_ns();++stamp->gpu_ops;}};
        if(hot&&mode=='S')join();
        if(hot<TOP){if(stamp)stamp->cpu_begin=hot_ns();require(ggml_backend_graph_compute(private_cpu,l.cpu->graph)==GGML_STATUS_SUCCESS,"CPU graph failed");ggml_backend_tensor_get(l.cpu->y,dst->data,0,size_t(D)*TOP*sizeof(float));if(stamp)stamp->cpu_end=hot_ns();}
        if(hot&&mode=='P')join();
        if(hot)for(int j=0;j<hot;++j)std::memcpy(static_cast<float*>(dst->data)+hot_slots[j]*D,pinned_output+j*D,D*sizeof(float));
        if(!stamp&&l.calls==0)LLAMA_LOG_INFO("xeon-hot-first layer=%d H=%d resident=%zu\n",l.index,hot,l.selected.size());
        ++l.calls;++l.hist[hot];
    }
};
State * lookup(ggml_backend_t outer){std::lock_guard<std::mutex>lock(registry_mutex);auto it=registry.find(outer);return it==registry.end()?nullptr:it->second;}
const char * backend_name(ggml_backend_t){return "XEON_HOT_FFN";}
void backend_free(ggml_backend_t b){delete static_cast<State*>(b->context);delete b;}
ggml_status backend_compute(ggml_backend_t b,ggml_cgraph * graph){
    auto & s=*static_cast<State*>(b->context);std::lock_guard<std::mutex>lock(s.execution_mutex);
    try{for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto node=ggml_graph_node(graph,i);if(ggml_op_is_empty(node->op))continue;
        require(marked(node),"unexpected node in dedicated backend");ggml_custom_op_params p{};std::memcpy(&p,node->op_params,sizeof(p));auto layer=static_cast<Layer*>(p.userdata);
        require(layer&&layer->owner==&s,"wrong context owner");s.compute(*layer,node);
    }return GGML_STATUS_SUCCESS;}catch(const std::exception & e){if(s.private_gpu)ggml_backend_synchronize(s.private_gpu);LLAMA_LOG_ERROR("%s\n",e.what());return GGML_STATUS_FAILED;}
}
const char * device_name(ggml_backend_dev_t){return "XEON_HOT_FFN";}
const char * device_description(ggml_backend_dev_t){return "Experimental host dispatcher for resident expert FFN";}
void device_memory(ggml_backend_dev_t,size_t * free,size_t * total){*free=0;*total=0;}
enum ggml_backend_dev_type device_type(ggml_backend_dev_t){return GGML_BACKEND_DEVICE_TYPE_ACCEL;}
void device_props(ggml_backend_dev_t dev,ggml_backend_dev_props * props){
    *props={};props->name=device_name(dev);props->description=device_description(dev);props->type=device_type(dev);props->memory_free=0;props->memory_total=0;
}
ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t){return ggml_backend_cuda_host_buffer_type();}
bool device_op(ggml_backend_dev_t,const ggml_tensor * op){return marked(op);}
bool device_buffer(ggml_backend_dev_t,ggml_backend_buffer_type_t buft){return ggml_backend_buft_is_host(buft);}
ggml_backend_device & device(){
    static ggml_backend_device d=[](){ggml_backend_device x{};x.iface.get_name=device_name;x.iface.get_description=device_description;x.iface.get_memory=device_memory;x.iface.get_type=device_type;x.iface.get_props=device_props;x.iface.get_buffer_type=device_buft;x.iface.supports_op=device_op;x.iface.supports_buft=device_buffer;return x;}();return d;
}
ggml_backend_t attach(std::unique_ptr<State> state){
    static ggml_guid guid={0x93,0x41,0x51,0x8d,0x72,0x46,0x11,0x26,0xae,0x38,0x15,0x12,0x09,0x07,0x6b,0xa2};
    auto backend=std::make_unique<ggml_backend>();*backend={};backend->guid=&guid;backend->device=&device();backend->context=state.get();backend->iface.get_name=backend_name;backend->iface.free=backend_free;backend->iface.graph_compute=backend_compute;
    state->self=backend.get();{std::lock_guard<std::mutex>lock(registry_mutex);require(registry.empty(),"context registration race");registry.emplace(state->outer_cpu,state.get());}
    state.release();return backend.release();
}
}

ggml_backend_t xeon_hot_backend_create(const llama_model & model,ggml_backend_t outer_cpu,ggml_backend_t outer_gpu,int threads,int batch_threads){
    const char * enabled=std::getenv("GGML_XEON_HOT_FFN");if(!enabled||!std::strcmp(enabled,"0"))return nullptr;
    require(!std::strcmp(enabled,"1"),"GGML_XEON_HOT_FFN must be0or1");
    require(model.arch==LLM_ARCH_DEEPSEEK41&&model.hparams.n_embd==D&&model.hparams.n_layer_all==LAYERS&&model.layers.size()==LAYERS&&model.hparams.n_expert==E,"supported model required");
    require(threads==12&&batch_threads==24&&model.devices.size()==1&&outer_gpu,"12/24 and singleGPU required");
    if(model.no_alloc()){LLAMA_LOG_INFO("xeon-hot: disabled for no_alloc estimates; hot bank is separately budgeted\n");return nullptr;}
    const char * plan=std::getenv("GGML_XEON_HOT_PLAN");require(plan&&*plan,"explicit plan required");
    {std::lock_guard<std::mutex>lock(registry_mutex);require(registry.empty(),"pilot supports one context only");}
    auto state=std::make_unique<State>(&model);state->outer_cpu=outer_cpu;state->outer_gpu=outer_gpu;state->read_plan(plan);
    return attach(std::move(state));
}
void xeon_hot_backend_finish(ggml_backend_t outer_cpu){if(auto s=lookup(outer_cpu))s->initialize();}
ggml_tensor * xeon_hot_build(ggml_backend_t outer_cpu,ggml_backend_sched_t sched,ggml_context * ctx,ggml_tensor * input,ggml_tensor * ids,ggml_tensor * gate,ggml_tensor * up,ggml_tensor * down,float clamp,int layer){
    auto s=lookup(outer_cpu);if(!s)return nullptr;
    require(layer>=0&&layer<LAYERS,"graph layer index");if(!s->layers[layer])return nullptr;auto & l=*s->layers[layer];
    if(!l.eligible||l.selected.empty())return nullptr;
    require(l.original==std::array<ggml_tensor*,3>{gate,up,down}&&clamp==l.clamp,"graph weight/clamp mismatch");
    if(input->type!=GGML_TYPE_F32||input->ne[0]!=D||input->ne[1]!=1||input->ne[2]!=1||input->ne[3]!=1||ids->type!=GGML_TYPE_I32||ids->ne[0]!=TOP||ids->ne[1]!=1)return nullptr;
    ggml_tensor * args[]={input,ids};auto node=ggml_custom_4d(ctx,GGML_TYPE_F32,D,TOP,1,1,args,2,marker,1,&l);
    ggml_set_name(node,("xeon_hot_ffn_"+std::to_string(layer)).c_str());ggml_backend_sched_set_tensor_backend(sched,node,s->self);return node;
}
void xeon_hot_place_tail(ggml_backend_t outer_cpu,ggml_backend_sched_t sched,ggml_tensor * tail){if(auto s=lookup(outer_cpu))ggml_backend_sched_set_tensor_backend(sched,tail,s->outer_gpu);}
