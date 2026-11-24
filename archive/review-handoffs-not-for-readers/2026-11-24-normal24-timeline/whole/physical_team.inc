static std::vector<int> declared_cpus() {
    const char *text=std::getenv("XEON_L7_STUDY_CPUS");
    require(text,"explicit test CPU list");
    std::istringstream input(text);std::vector<int> ids;int cpu;
    std::set<std::pair<int,int>> cores;std::map<int,int> packages;
    while(input>>cpu) {
        require(cpu>=0 && cpu<GGML_MAX_N_THREADS && (ids.empty() || cpu>ids.back()),"ascending CPU list");
        const auto base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        std::ifstream s(base+"physical_package_id"),c(base+"core_id");int socket,core;
        require(bool(s>>socket) && bool(c>>core) && cores.emplace(socket,core).second,"distinct physical workers");
        ++packages[socket];ids.push_back(cpu);
    }
    require(input.eof() && ids.size()==24 && packages.size()==2 && packages[0]==12 && packages[1]==12,"24 physical 12+12");
    return ids;
}
struct WorkerPoint {int cpu=-1, nth=0;long tid=-1;bool pinned=false;};
static void probe_private(ggml_tensor *dst,int ith,int nth,void *data) {
    if(ith<0 || ith>=24)return;
    auto &point=static_cast<std::array<WorkerPoint,24>*>(data)->at(ith);
    point.cpu=sched_getcpu();point.nth=nth;point.tid=syscall(SYS_gettid);
    cpu_set_t allowed;CPU_ZERO(&allowed);
    point.pinned=sched_getaffinity(0,sizeof(allowed),&allowed)==0 && point.cpu>=0 && CPU_COUNT(&allowed)==1 && CPU_ISSET(point.cpu,&allowed);
    static_cast<float*>(dst->data)[ith]=float(ith);
}
static void verify_private_team(ggml_backend_t cpu,const std::vector<int> &ids) {
    std::array<WorkerPoint,24> points{};
    auto *ctx=ggml_init({65536,nullptr,true});require(ctx,"worker check metadata");
    auto *x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,24);
    ggml_tensor *args[]={x};
    auto *y=ggml_custom_4d(ctx,GGML_TYPE_F32,24,1,1,1,args,1,probe_private,24,&points);
    auto *graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,cpu);require(buf,"worker check buffer");
    const auto status=ggml_backend_graph_compute(cpu,graph);
    ggml_backend_buffer_free(buf);ggml_free(ctx);
    require(status==GGML_STATUS_SUCCESS,"worker check graph");
    std::set<long> tids;
    for(size_t i=0;i<ids.size();++i) {
        const auto &p=points[i];
        require(p.pinned && p.cpu==ids[i] && p.nth==24 && tids.insert(p.tid).second,"private worker mask");
        LLAMA_LOG_INFO("l7-private-worker ordinal=%zu cpu=%d tid=%ld\n",i,p.cpu,p.tid);
    }
}
