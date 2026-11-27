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

struct Layer {std::array<ggml_tensor*,3>w{};float limit=0;};
static ggml_tensor *ffn(Context &ctx,Layer &l,ggml_tensor *x,ggml_tensor *ids){
    auto gate=ggml_mul_mat_id(ctx.p,l.w[0],x,ids);auto up=ggml_mul_mat_id(ctx.p,l.w[1],x,ids);
    auto act=ggml_swiglu_clamp(ctx.p,gate,up,l.limit);auto down=ggml_mul_mat_id(ctx.p,l.w[2],act,ids);
    ggml_set_name(down,"local_down");return down;
}
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
