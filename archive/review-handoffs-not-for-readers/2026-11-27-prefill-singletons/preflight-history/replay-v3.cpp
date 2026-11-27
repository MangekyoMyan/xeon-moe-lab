#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "llama-context.h"
#include "iq3-cache.h"
#include "json.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <unistd.h>

using Clock=std::chrono::steady_clock;
using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
static constexpr int D=5120,F=2304,K=6,T=512,E=384;
static void need(bool ok,const std::string &why){if(!ok)throw std::runtime_error(why);}
static void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
extern "C" void split_gather(const float*,const int32_t*,float*,int,int,int,cudaStream_t);
extern "C" void split_scatter(const float*,const int32_t*,float*,int,int,cudaStream_t);
#include "weights.h"
#include "pool24.h"
#include "activity.h"

static json readj(const fs::path &p){std::ifstream f(p);need(bool(f),"read "+p.string());return json::parse(f);}
static void save(const fs::path &p,const json &v){
    auto tmp=p;tmp+=".tmp";std::ofstream f(tmp);f<<v.dump(2)<<'\n';f.close();need(!f.fail(),"save");fs::rename(tmp,p);
}
template<class V>static std::vector<V> readv(const fs::path &p,size_t n){
    need(fs::file_size(p)==n*sizeof(V),"extent "+p.string());std::vector<V> v(n);
    std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*sizeof(V));need(bool(f),"read array");return v;
}
static void writev(const fs::path &p,const std::vector<float>&v){
    std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*4);f.close();need(!f.fail(),"write array");
}
static std::vector<float> get(ggml_tensor *t){
    need(ggml_is_contiguous(t) && t->type==GGML_TYPE_F32,"read flat f32");std::vector<float> y(ggml_nelements(t));
    ggml_backend_tensor_get(t,y.data(),0,y.size()*4);return y;
}
static json compare(const std::vector<float>&a,const std::vector<float>&b){
    need(a.size()==b.size()&&!a.empty(),"comparison extent");double s=0,maxerr=0;size_t diff=0;
    for(size_t i=0;i<a.size();++i){need(std::isfinite(a[i])&&std::isfinite(b[i]),"finite");
        double e=double(a[i])-b[i];s+=e*e;maxerr=std::max(maxerr,std::abs(e));diff+=std::memcmp(&a[i],&b[i],4)!=0;}
    return {{"n",a.size()},{"bit_different",diff},{"rmse",std::sqrt(s/a.size())},{"max_abs",maxerr}};
}
static size_t free_gpu(){size_t f,t;ck(cudaMemGetInfo(&f,&t));return f;}
static std::string provider(void *fn){Dl_info d{};need(fn&&dladdr(fn,&d)&&d.dli_fname,"symbol");return fs::canonical(d.dli_fname);}
template<class V>struct DeviceArray{
    V *p=nullptr;explicit DeviceArray(size_t n){ck(cudaMalloc(&p,n*sizeof(V)));}~DeviceArray(){cudaFree(p);}
    DeviceArray(const DeviceArray&)=delete;
};
struct Stream{cudaStream_t s=nullptr;Stream(){ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(s);}};
struct Context{ggml_context *p;explicit Context(size_t n=2u<<20):p(ggml_init({n,nullptr,true})){need(p,"graph metadata");}~Context(){ggml_free(p);}};
struct Buffer{ggml_backend_buffer_t p=nullptr;~Buffer(){if(p)ggml_backend_buffer_free(p);}void allocate(Context &ctx,ggml_backend_t b){p=ggml_backend_alloc_ctx_tensors(ctx.p,b);need(p,"backend buffer");}};
struct Pinned{
    void *p=nullptr;ggml_backend_buffer_t buf=nullptr;
    void bind(ggml_tensor *t){auto n=ggml_nbytes(t);ck(cudaHostAlloc(&p,n,cudaHostAllocPortable));buf=ggml_backend_cpu_buffer_from_ptr(p,n);
        need(buf&&ggml_backend_tensor_alloc(buf,t,p)==GGML_STATUS_SUCCESS,"pinned tensor");}
    ~Pinned(){if(buf)ggml_backend_buffer_free(buf);if(p)cudaFreeHost(p);}
};
struct Layer {
    Context ctx;std::array<std::unique_ptr<Mapping>,3>maps;std::array<ggml_tensor*,3>w{};void *cache=nullptr;float limit=0;json record=json::array();
    Layer(const std::string&dir,int l,const json&capture){
        auto m=metadata(dir,l);limit=m.clamp;cache=xeon_iq3_cache_create_v1(size_t(8)<<30,size_t(64)<<30,12);need(cache,"CPU compact owner");
        for(int j=0;j<3;++j){auto &a=m.w[j];maps[j]=std::make_unique<Mapping>(a);
            w[j]=ggml_new_tensor_3d(ctx.p,a.q,a.K,a.M,a.E);ggml_set_name(w[j],a.name.c_str());
            need(ggml_backend_tensor_alloc(maps[j]->buffer,w[j],maps[j]->raw)==GGML_STATUS_SUCCESS,"mapped original tensor");
            auto &wm=capture["weight_meta"][std::to_string(l)][j];need(wm["name"]==a.name && wm["type"]==ggml_type_name(a.q)
                && wm["nb"]==std::vector<size_t>(w[j]->nb,w[j]->nb+4),"actual weight metadata");
            std::vector<uint32_t> packed;
            if(capture["registration"]["experts"].contains(a.name)){
                packed=capture["registration"]["experts"][a.name].get<std::vector<uint32_t>>();
                if(!packed.empty())need(xeon_iq3_cache_add_selected_v2(cache,w[j],packed.data(),packed.size())==1,"subset layer registration");
            }
            auto table=xeon_iq3_cache_lookup_experts_v2(w[j]);
            for(int e=0;e<E;++e)need(bool(table&&table[e])==(std::find(packed.begin(),packed.end(),e)!=packed.end()),"registry per projection");
            record.push_back({{"name",a.name},{"type",ggml_type_name(a.q)},{"bytes_per_expert",a.expertbytes},{"compact_ids",packed},{"host_registered",maps[j]->registered}});
        }
    }
    ~Layer(){if(cache)xeon_iq3_cache_free_v1(cache);}
};
static ggml_tensor *ffn(Context &ctx,Layer &l,ggml_tensor *x,ggml_tensor *ids){
    auto gate=ggml_mul_mat_id(ctx.p,l.w[0],x,ids);auto up=ggml_mul_mat_id(ctx.p,l.w[1],x,ids);
    auto act=ggml_swiglu_clamp(ctx.p,gate,up,l.limit);auto down=ggml_mul_mat_id(ctx.p,l.w[2],act,ids);
    ggml_set_name(down,"local_down");return down;
}
static ggml_tensor *sum_slots(Context &ctx,ggml_cgraph *g,ggml_tensor *pairs,ggml_tensor *weights){
    auto weighted=ggml_mul(ctx.p,pairs,weights);ggml_build_forward_expand(g,weighted);
    std::array<ggml_tensor*,K>v;
    for(int k=0;k<K;++k){v[k]=ggml_view_2d(ctx.p,weighted,D,T,weighted->nb[2],k*weighted->nb[1]);ggml_build_forward_expand(g,v[k]);}
    auto y=v[0];for(int k=1;k<K;++k){y=ggml_add(ctx.p,y,v[k]);ggml_build_forward_expand(g,y);}
    ggml_set_name(y,"local_weighted_sum");return y;
}
struct Inputs{
    Context ctx;Buffer buf;ggml_tensor *x,*backing,*ids,*mix,*canonical,*sum;ggml_cgraph *graph;
    Inputs(ggml_backend_t gpu,const fs::path&dir,const std::string &suffix,size_t stride){
        need(stride==384*4,"native topk stride");x=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,1,T);
        backing=ggml_new_tensor_2d(ctx.p,GGML_TYPE_I32,stride/4,T);
        ids=ggml_view_2d(ctx.p,backing,K,T,stride,0);
        mix=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,1,K,T);
        canonical=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,K,T);
        graph=ggml_new_graph_custom(ctx.p,64,false);sum=sum_slots(ctx,graph,canonical,mix);
        buf.allocate(ctx,gpu);
        auto xv=readv<float>(dir/("x"+suffix+".bin"),D*T);auto iv=readv<int32_t>(dir/("ids"+suffix+".bin"),K*T);
        auto mv=readv<float>(dir/("ffn_moe_weights_scaled"+suffix+".bin"),K*T);
        std::vector<int32_t> padded(stride/4*T);for(int t=0;t<T;++t)std::copy_n(iv.begin()+t*K,K,padded.begin()+t*(stride/4));
        ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);ggml_backend_tensor_set(backing,padded.data(),0,padded.size()*4);
        ggml_backend_tensor_set(mix,mv.data(),0,mv.size()*4);ggml_backend_synchronize(gpu);
    }
};
struct Native{
    // The tiny model supplies the installed llama_context scheduler and its real
    // sched_copy_experts callback; none of its weights enter this FFN.
    llama_context *lc=nullptr;ggml_backend_sched_t sch=nullptr;ggml_backend_t gpu=nullptr;
    Context input_ctx,ctx;Buffer input_buf;ggml_tensor *x,*ids,*down,*result;ggml_cgraph *g;
    bool packed;json meta;
    Native(llama_model *shell,Layer&l,Inputs &io,int n,bool use_pack):packed(use_pack){
        auto p=llama_context_default_params();p.n_ctx=128;p.n_batch=1024;p.n_ubatch=1024;p.n_threads=p.n_threads_batch=24;
        p.op_offload=true;p.no_perf=true;
        lc=llama_init_from_model(shell,p);need(lc,"native scheduler context");sch=lc->get_sched();
        for(int i=0;i<ggml_backend_sched_get_n_backends(sch);++i){auto b=ggml_backend_sched_get_backend(sch,i);if(std::string(ggml_backend_name(b))=="CUDA0")gpu=b;}
        need(gpu,"native CUDA scheduler backend");
        if(packed){x=ggml_new_tensor_3d(input_ctx.p,GGML_TYPE_F32,D,1,n);ids=ggml_new_tensor_2d(input_ctx.p,GGML_TYPE_I32,1,n);input_buf.allocate(input_ctx,gpu);}
        else{x=io.x;ids=io.ids;}
        g=ggml_new_graph_custom(ctx.p,96,false);down=ffn(ctx,l,x,ids);
        if(packed){result=down;ggml_build_forward_expand(g,down);}else result=sum_slots(ctx,g,down,io.mix);
        ggml_backend_sched_reset(sch);
        for(int i=0;i<ggml_graph_n_nodes(g);++i)ggml_backend_sched_set_tensor_backend(sch,ggml_graph_node(g,i),gpu);
        need(ggml_backend_sched_alloc_graph(sch,g),"native scheduler allocation");
        need(ggml_backend_sched_get_tensor_backend(sch,down)==gpu && ggml_backend_sched_get_tensor_backend(sch,result)==gpu,"CUDA placement");
        meta={{"pairs",packed?n:K*T},{"top",packed?1:K},{"tokens",n},{"splits",ggml_backend_sched_get_n_splits(sch)},
              {"staging_and_work_B",ggml_backend_sched_get_buffer_size(sch,gpu)},{"gpu_free_after_B",free_gpu()},
              {"gpu_provider",provider(reinterpret_cast<void*>(gpu->iface.graph_compute))}};
    }
    void run(){need(lc->graph_compute(g,true)==GGML_STATUS_SUCCESS,"installed graph_compute");ggml_backend_synchronize(gpu);}
    static bool get_down(ggml_tensor *t,bool ask,void *out){
        bool match=t->op==GGML_OP_MUL_MAT_ID && std::string(t->name)=="local_down";
        if(match&&!ask)*static_cast<std::vector<float>*>(out)=get(t);
        return match;
    }
    std::vector<float> validate_down(){
        std::vector<float> out;ggml_backend_sched_set_eval_callback(sch,get_down,&out);run();
        ggml_backend_sched_set_eval_callback(sch,nullptr,nullptr);return out;
    }
    ~Native(){if(lc)llama_free(lc);}
};
struct Cpu{
    Context ctx;Pinned input,indices,output;Buffer buf;ggml_cgraph *g;ggml_tensor *x,*ids,*down;
    Cpu(ggml_backend_t cpu,Layer &l,int n,int k){
        x=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,1,n);ids=ggml_new_tensor_2d(ctx.p,GGML_TYPE_I32,k,n);
        down=ffn(ctx,l,x,ids);g=ggml_new_graph_custom(ctx.p,64,false);ggml_build_forward_expand(g,down);
        input.bind(x);indices.bind(ids);output.bind(down);buf.allocate(ctx,cpu);
    }
};
class Worker{
    std::mutex m;std::condition_variable cv;bool ready=false,done=true,quit=false;std::function<void()>job;std::exception_ptr err;
    std::thread thread;
public:
    Worker():thread([this]{for(;;){std::unique_lock<std::mutex> lock(m);cv.wait(lock,[&]{return ready||quit;});if(quit)return;
        auto fn=std::move(job);ready=false;lock.unlock();try{fn();}catch(...){err=std::current_exception();}
        lock.lock();done=true;cv.notify_all();}}){}
    void submit(std::function<void()> f){std::lock_guard<std::mutex> lock(m);need(done&&!ready,"worker busy");err=nullptr;done=false;ready=true;job=std::move(f);cv.notify_all();}
    void wait(){std::unique_lock<std::mutex> lock(m);cv.wait(lock,[&]{return done;});if(err)std::rethrow_exception(err);}
    ~Worker(){{std::unique_lock<std::mutex> lock(m);cv.wait(lock,[&]{return done;});quit=true;cv.notify_all();}thread.join();}
};
struct Partition{
    std::array<int,E>count{};std::vector<int32_t>cs,ci,gs,gi;
    Partition(const int32_t *ids,bool split){
        for(int j=0;j<T*K;++j){need(ids[j]>=0&&ids[j]<E,"route ID");++count[ids[j]];}
        for(int j=0;j<T*K;++j){bool c=split&&count[ids[j]]==1;(c?cs:gs).push_back(j);(c?ci:gi).push_back(ids[j]);}
    }
};
struct Pressure{
    static constexpr size_t n=size_t(96)*1024*1024/sizeof(uint64_t);
    std::vector<uint64_t> data;Pressure():data(2*n){for(size_t i=0;i<data.size();i+=8)data[i]=i+1;}
    void apply(){
        std::array<uint64_t,2> sums{};
        auto read=[&](int s){cpu_set_t set;CPU_ZERO(&set);CPU_SET(12*s,&set);
            if(sched_setaffinity(0,sizeof(set),&set)!=0)std::abort();
            for(size_t i=0;i<n;i+=8)sums[s]+=data[n*s+i];};
        std::thread a(read,0),b(read,1);a.join();b.join();need(sums[0]&&sums[1],"LLC working set");
    }
};
static uint64_t now(){return activity::clock_ns();}
struct Bench{
    llama_model *shell;ggml_backend_t cpu,gpu;Layer &l;Inputs &io;Worker &worker;
    std::unique_ptr<Native> native,allpack,hybrid;
    std::unique_ptr<Cpu> full,rare;
    DeviceArray<int32_t> gpu_slots{T*K},cpu_slots{T*K};DeviceArray<float> tmp{size_t(D)*K*T};
    Stream stream;int32_t *host_ids=nullptr;
    Bench(llama_model*m,ggml_backend_t c,ggml_backend_t g,Layer&ly,Inputs&i,Worker&w):shell(m),cpu(c),gpu(g),l(ly),io(i),worker(w){
        ck(cudaHostAlloc(&host_ids,T*K*4,cudaHostAllocPortable));
    }
    ~Bench(){worker.wait();if(host_ids)cudaFreeHost(host_ids);}
    json run(const std::string &arm){
        uint64_t begin=now(),cpu_begin=0,cpu_end=0,gpu_begin=0,gpu_end=0,route_end=0,join=0;
        struct Drain{Worker &w;~Drain(){try{w.wait();}catch(...){}}} drain{worker};
        size_t nsing=0;
        if(arm=="G"){
            if(!native)native=std::make_unique<Native>(shell,l,io,T,false);
            gpu_begin=now();native->run();gpu_end=now();
        }else{
            ck(cudaMemcpy2DAsync(host_ids,K*4,io.ids->data,io.ids->nb[1],K*4,T,cudaMemcpyDeviceToHost,stream.s));
            ck(cudaStreamSynchronize(stream.s));
            if(arm=="C"){
                if(!full)full=std::make_unique<Cpu>(cpu,l,T,K);
                std::memcpy(full->ids->data,host_ids,T*K*4);route_end=now();
                ck(cudaMemcpyAsync(full->x->data,io.x->data,D*T*4,cudaMemcpyDeviceToHost,stream.s));ck(cudaStreamSynchronize(stream.s));
                cpu_begin=now();need(ggml_backend_graph_compute(cpu,full->g)==GGML_STATUS_SUCCESS,"full CPU");cpu_end=now();
                ck(cudaMemcpyAsync(io.canonical->data,full->down->data,size_t(D)*K*T*4,cudaMemcpyHostToDevice,stream.s));
            }else{
                Partition plan(host_ids,arm!="Gprobe");
                nsing=plan.cs.size();route_end=now();
                auto &gp=arm=="Gprobe"?allpack:hybrid;
                if(!gp)gp=std::make_unique<Native>(shell,l,io,plan.gs.size(),true);
                need(gp->x->ne[2]==int64_t(plan.gs.size()),"new GPU cardinality needs a new graph");
                if(nsing){
                    if(!rare)rare=std::make_unique<Cpu>(cpu,l,nsing,1);
                    need(rare->x->ne[2]==int64_t(nsing),"new CPU cardinality needs a new graph");
                    std::memcpy(rare->ids->data,plan.ci.data(),nsing*4);
                    ck(cudaMemcpyAsync(cpu_slots.p,plan.cs.data(),nsing*4,cudaMemcpyHostToDevice,stream.s));
                    split_gather((float*)io.x->data,cpu_slots.p,tmp.p,D,nsing,K,stream.s);ck(cudaGetLastError());
                    ck(cudaMemcpyAsync(rare->x->data,tmp.p,size_t(D)*nsing*4,cudaMemcpyDeviceToHost,stream.s));ck(cudaStreamSynchronize(stream.s));
                    worker.submit([&]{cpu_begin=now();need(ggml_backend_graph_compute(cpu,rare->g)==GGML_STATUS_SUCCESS,"singleton CPU");cpu_end=now();});
                    if(arm=="S")worker.wait();
                }
                // No packed GPU weights or negative IDs. Original tensor and global
                // IDs go through the installed selective-copy callback each call.
                gpu_begin=now();ck(cudaMemcpyAsync(gpu_slots.p,plan.gs.data(),plan.gs.size()*4,cudaMemcpyHostToDevice,stream.s));
                ck(cudaMemcpyAsync(gp->ids->data,plan.gi.data(),plan.gi.size()*4,cudaMemcpyHostToDevice,stream.s));
                split_gather((float*)io.x->data,gpu_slots.p,(float*)gp->x->data,D,plan.gs.size(),K,stream.s);
                ck(cudaGetLastError());ck(cudaStreamSynchronize(stream.s));gp->run();
                split_scatter((float*)gp->down->data,gpu_slots.p,(float*)io.canonical->data,D,plan.gs.size(),stream.s);
                ck(cudaGetLastError());ck(cudaStreamSynchronize(stream.s));gpu_end=now();
                if(nsing){
                    worker.wait();join=now();
                    ck(cudaMemcpyAsync(tmp.p,rare->down->data,size_t(D)*nsing*4,cudaMemcpyHostToDevice,stream.s));
                    split_scatter(tmp.p,cpu_slots.p,(float*)io.canonical->data,D,nsing,stream.s);ck(cudaGetLastError());
                }
            }
            ck(cudaStreamSynchronize(stream.s));
            need(ggml_backend_graph_compute(gpu,io.graph)==GGML_STATUS_SUCCESS,"canonical CUDA weighting");
            ggml_backend_synchronize(gpu);
        }
        const auto end=now();
        return {{"arm",arm},{"start",begin},{"end",end},{"route_end",route_end},{"cpu_begin",cpu_begin},{"cpu_end",cpu_end},
                {"gpu_branch_begin",gpu_begin},{"gpu_branch_end",gpu_end},{"join",join},{"cpu_selected_slots",arm=="C"?size_t(K*T):nsing}};
    }
    std::vector<float> sum(const std::string &arm){return get(arm=="G"?native->result:io.sum);}
    std::vector<float> slots(const std::string &arm){return arm=="G"?native->validate_down():get(io.canonical);}
};

struct CoreReceipt{int n=0,cpu=-1,masks=0;long tid=0;};
static void core_node(ggml_tensor *out,int ith,int nth,void *data){
    cpu_set_t set;CPU_ZERO(&set);auto *cores=static_cast<CoreReceipt*>(data);
    cores[ith]={nth,sched_getcpu(),sched_getaffinity(0,sizeof(set),&set)?-1:CPU_COUNT(&set),syscall(SYS_gettid)};
    ((float*)out->data)[ith]=ith;
}
static json verify_team(ggml_backend_t cpu){
    Context c(65536);Buffer buf;std::array<CoreReceipt,24> workers{};
    auto *x=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,24);ggml_tensor *args[]={x};
    auto *y=ggml_custom_4d(c.p,GGML_TYPE_F32,24,1,1,1,args,1,core_node,24,workers.data());
    auto *g=ggml_new_graph_custom(c.p,16,false);ggml_build_forward_expand(g,y);buf.allocate(c,cpu);
    need(ggml_backend_graph_compute(cpu,g)==GGML_STATUS_SUCCESS,"team check");json out=json::array();std::set<long>tids;
    for(int i=0;i<24;++i){auto v=workers[i];need(v.n==24&&v.cpu==i&&v.masks==1&&tids.insert(v.tid).second,"24 fixed physical cores");
        out.push_back({{"ith",i},{"cpu",v.cpu},{"tid",v.tid}});}
    return out;
}
static json providers(){
    return {{"llama",provider(reinterpret_cast<void*>(llama_decode))},{"cpu",provider(reinterpret_cast<void*>(ggml_backend_cpu_init))},
        {"base",provider(reinterpret_cast<void*>(ggml_backend_sched_graph_compute_async))},{"cuda",provider(reinterpret_cast<void*>(ggml_backend_cuda_init))}};
}
int main(int argc,char **argv)try{
    if(argc==2&&std::string(argv[1])=="--providers"){std::cout<<providers().dump()<<'\n';return 0;}
    need(argc==3,"SPEC OUT");json spec=readj(argv[1]);fs::path out=argv[2],cap=spec["capture"].get<std::string>();
    bool trace=spec["trace"];int layer=spec["layer"];const auto suffix="-"+std::to_string(layer);
    const json capture=readj(cap/"RUN.json");need(capture["passed"]&&capture["capture"]["native_logits_equal"],"validated real capture");
    const auto provided=providers();
    need(provided.size()==4 && spec.at("providers").size()==4,"four provider entries");
    for(auto it=provided.begin();it!=provided.end();++it)need(it.value()==spec.at("providers").at(it.key()),"provider mismatch "+it.key());
    json report={{"passed",false},{"entry",read_entry_topology()},{"providers",providers()},{"layer",layer},{"samples",json::array()},{"trace",trace}};
    activity::init(trace);ck(cudaSetDevice(0));report["initial_free_gpu_B"]=free_gpu();
    llama_backend_init();
    auto cpu=ggml_backend_cpu_init(),gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backends");
    auto pp=private_pool_params();auto pool=ggml_threadpool_new(&pp);need(pool,"pool");
    ggml_backend_cpu_set_n_threads(cpu,24);ggml_backend_cpu_set_threadpool(cpu,pool);
    ggml_backend_dev_t devices[]={ggml_backend_get_device(gpu),nullptr};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=99;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_OFF;auto *shell=llama_model_load_from_file(spec["shell"].get<std::string>().c_str(),mp);need(shell,"scheduler shell");
    {
        Worker worker;Pressure pressure;
        Layer l(spec["model_dir"],layer,capture);report["weights"]=l.record;report["compact_B"]=xeon_iq3_cache_used_v1(l.cache);
        Inputs io(gpu,cap,suffix,capture["capture"]["tensors"]["ids"+suffix]["nb"][1]);
        auto ids=readv<int32_t>(cap/("ids"+suffix+".bin"),K*T);
        auto refslots=readv<float>(cap/("ffn_moe_down"+suffix+".bin"),size_t(D)*K*T);
        auto refsum=readv<float>(cap/("ffn_moe_out"+suffix+".bin"),D*T);
        Partition p(ids.data(),true);
        for(int t=0;t<T;++t)need(std::set<int32_t>(ids.begin()+K*t,ids.begin()+K*(t+1)).size()==K,"six distinct slots");
        need(p.cs.size()>1&&p.cs.size()+p.gs.size()==size_t(T*K),"actual singleton subsets");
        report["cpu_singleton_slots"]=p.cs;report["gpu_slots"]=p.gs;report["histogram"]=p.count;
        report["team_main"]=verify_team(cpu);worker.submit([&]{report["team_worker"]=verify_team(cpu);});worker.wait();
        Bench b(shell,cpu,gpu,l,io,worker);std::vector<std::string> arms={"G","C","Gprobe","S","P"};
        std::map<std::string,std::vector<float>> gold;std::vector<float> s_slots;
        for(const auto &arm:arms){
            auto stamp=b.run(arm);stamp["phase"]="first";report["samples"].push_back(stamp);
            gold[arm]=b.sum(arm);auto slots=b.slots(arm);
            auto qa=compare(slots,refslots);auto qs=compare(gold[arm],refsum);
            report["quality"][arm]={{"slots_vs_native",qa},{"sum_vs_native",qs}};
            if(arm=="G"){report["quality"][arm]["sum_callback_vs_plain"]=compare(b.sum(arm),gold[arm]);
                report["native_matches_model"]=qa["bit_different"]==0&&qs["bit_different"]==0&&report["quality"][arm]["sum_callback_vs_plain"]["bit_different"]==0;}
            if(arm=="S")s_slots=slots;
            if(arm=="P"){
                report["S_P_slots"]=compare(slots,s_slots);report["S_P_sum"]=compare(gold["P"],gold["S"]);
                need(report["S_P_slots"]["bit_different"]==0 && report["S_P_sum"]["bit_different"]==0,"serial/parallel numerical mismatch");
            }
            if(!trace){writev(out/(arm+"-sum.f32"),gold[arm]);}
            save(out/"RUN.json",report);
        }
        report["G_geometry"]=b.native->meta;report["Gprobe_geometry"]=b.allpack->meta;report["SP_geometry"]=b.hybrid->meta;
        for(const auto &arm:arms){auto s=b.run(arm);s["phase"]="warmup";report["samples"].push_back(s);}
        if(trace){
            for(const auto &arm:arms){
                activity::begin();auto s=b.run(arm);activity::end(out/(arm+"-activity.tsv"));
                s["phase"]="CUPTI";report["samples"].push_back(s);
                need(compare(b.sum(arm),gold[arm])["bit_different"]==0,"observed run reproducibility");
                save(out/"RUN.json",report);
            }
            report["cupti_dropped"]=activity::dropped;report["cupti_errors"]=activity::errors;
        }else{
            for(int block=0;block<2;++block){
                auto ord=arms;if(block)std::reverse(ord.begin(),ord.end());
                for(int ic=0;ic<2;++ic){
                    bool pressure_mode=(block?1-ic:ic)!=0;
                    for(const auto &arm:ord){
                        for(int rep=0;rep<spec["reps"].get<int>();++rep){
                            if(pressure_mode)pressure.apply();
                            auto s=b.run(arm);s["phase"]="primary";s["block"]=block;s["llc_pressure"]=pressure_mode;s["rep"]=rep;report["samples"].push_back(s);
                        }
                        need(compare(b.sum(arm),gold[arm])["bit_different"]==0,"primary block reproducibility");
                        save(out/"RUN.json",report);
                    }
                }
            }
        }
        report["cuda_free_after_all_graphs_B"]=free_gpu();
    }
    llama_model_free(shell);ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);
    ggml_backend_free(cpu);ggml_backend_free(gpu);llama_backend_free();
    if(trace)need(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)==CUPTI_SUCCESS,"CUPTI shutdown");
    report["passed"]=true;save(out/"RUN.json",report);return 0;
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
