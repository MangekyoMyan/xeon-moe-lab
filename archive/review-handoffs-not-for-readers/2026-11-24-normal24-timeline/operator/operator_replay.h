#pragma once
// Isolated operator test. This TU owns a dormant bank, not a model graph hook.
#include "hot-backend.cpp"
#include "capture_io.h"
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

struct WorkerReceipt {int nth=0,cpu=-1,rc=-1;long tid=0;cpu_set_t mask{};};
static void worker_probe(ggml_tensor *dst,int ith,int nth,void *data){
    if(ith<0||ith>=private_threads)return;
    auto &w=static_cast<std::array<WorkerReceipt,private_threads>*>(data)->at(ith);
    w.nth=nth;w.cpu=sched_getcpu();w.tid=syscall(SYS_gettid);CPU_ZERO(&w.mask);w.rc=sched_getaffinity(0,sizeof(w.mask),&w.mask);
    static_cast<float*>(dst->data)[ith]=float(ith);
}
static json verify_workers(ggml_backend_t backend){
    std::array<WorkerReceipt,private_threads>workers{};auto ctx=ggml_init({65536,nullptr,true});need(ctx,"worker probe metadata");
    auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,private_threads);ggml_tensor *args[]={x};
    auto y=ggml_custom_4d(ctx,GGML_TYPE_F32,private_threads,1,1,1,args,1,worker_probe,private_threads,&workers);
    auto g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,y);auto b=ggml_backend_alloc_ctx_tensors(ctx,backend);need(b,"worker probe buffer");
    need(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"worker probe compute");
    json result=json::array();std::set<int>cpus;std::set<long>tids;
    for(int i=0;i<private_threads;++i){auto &w=workers[i];need(w.nth==private_threads&&w.rc==0&&CPU_COUNT(&w.mask)==1&&w.cpu==physical_ids.at(i)&&CPU_ISSET(w.cpu,&w.mask),"private worker reached declared physical CPU");cpus.insert(w.cpu);tids.insert(w.tid);result.push_back({{"worker",i},{"tid",w.tid},{"cpu",w.cpu},{"allowed_cpu",w.cpu}});}
    need(cpus.size()==physical_ids.size()&&tids.size()==physical_ids.size(),"unique physical workers");ggml_backend_buffer_free(b);ggml_free(ctx);return result;
}

static json err_metrics(const std::vector<float>& got,const std::vector<float>& ref) {
    need(got.size()==ref.size()&&!got.empty(),"metric shape");double err=0,norm=0,gnorm=0,maxabs=0;
    for(size_t i=0;i<got.size();++i){need(std::isfinite(got[i])&&std::isfinite(ref[i]),"finite output");
        double d=double(got[i])-ref[i];err+=d*d;norm+=double(ref[i])*ref[i];gnorm+=double(got[i])*got[i];maxabs=std::max(maxabs,std::abs(d));}
    return {{"nmse",norm?json(err/norm):json(nullptr)},{"sum_squared_error",err},{"reference_norm2",norm},
        {"output_norm2",gnorm},{"max_abs",maxabs},{"count",got.size()},{"byte_equal",!std::memcmp(got.data(),ref.data(),got.size()*sizeof(float))}};
}
static std::vector<float> merge_slots(const std::vector<float>&v,const std::array<float,TOP>&w){
    std::vector<float> result(D);for(int k=0;k<D;++k){float a=v[k]*w[0];for(int j=1;j<TOP;++j){float b=v[j*D+k]*w[j];a+=b;}result[k]=a;}return result;
}
template<typename T> static void read_data(const std::filesystem::path&p,T *dst,size_t n){
    std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f)&&size_t(f.tellg())==n*sizeof(T),"fixture byte count");f.seekg(0);f.read(reinterpret_cast<char*>(dst),n*sizeof(T));need(bool(f),"fixture read");
}
static void write_f32(const std::filesystem::path&p,const std::vector<float>&v){
    std::ofstream f(p,std::ios::binary);need(bool(f),"output file");f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));f.close();need(!f.fail(),"output write");
}
// Same native GGUF dequantization / F64 dot reference used in the prior probe.
// F32 projection rounding and F32 SwiGLU; not the unquantized model's truth.
static std::vector<float> reference_ffn(const Layer&l,const std::array<int32_t,TOP>&ids,const std::array<float,D>&x){
    std::vector<float> result(D*TOP),gate(F),up(F),act(F),row(D);
    for(int slot=0;slot<TOP;++slot){
        for(int p=0;p<2;++p){auto w=l.original[p];auto fn=ggml_get_type_traits(w->type)->to_float;need(fn,"dequant trait");
            const auto *raw=static_cast<const char*>(w->data)+size_t(ids[slot])*w->nb[2];
            for(int m=0;m<F;++m){fn(raw+size_t(m)*w->nb[1],row.data(),D);double sum=0;for(int k=0;k<D;++k)sum+=double(row[k])*x[k];(p?up:gate)[m]=float(sum);}}
        for(int m=0;m<F;++m){float g=std::min(gate[m],l.clamp),u=std::clamp(up[m],-l.clamp,l.clamp);act[m]=(g/(1.f+std::exp(-g)))*u;}
        auto w=l.original[2];auto fn=ggml_get_type_traits(w->type)->to_float;need(fn,"down dequant trait");const auto *raw=static_cast<const char*>(w->data)+size_t(ids[slot])*w->nb[2];
        for(int m=0;m<D;++m){fn(raw+size_t(m)*w->nb[1],row.data(),F);double sum=0;for(int k=0;k<F;++k)sum+=double(row[k])*act[k];result[slot*D+m]=float(sum);}
    }
    return result;
}
struct OriginalOwner {
    ggml_backend_t cpu=nullptr;ggml_threadpool_t pool=nullptr;
    OriginalOwner(){cpu=ggml_backend_cpu_init();need(cpu,"O CPU");auto p=private_pool_params();
        pool=ggml_threadpool_new(&p);need(pool,"O pool");ggml_backend_cpu_set_n_threads(cpu,private_threads);ggml_backend_cpu_set_threadpool(cpu,pool);}
    ~OriginalOwner(){if(cpu)ggml_backend_cpu_set_threadpool(cpu,nullptr);if(pool)ggml_threadpool_free(pool);if(cpu)ggml_backend_free(cpu);}
};
static void operator_replay(ggml_backend_t bank,const llama_model&model,const std::filesystem::path&out,const std::filesystem::path&fixture){
    auto &state=*static_cast<State*>(bank->context);need(state.ready&&state.planned_count==3 && state.payload==36218880,"layer 7 three-expert resident bank");
    auto meta=read_json(fixture/"CAPTURE.json");need(meta["cells"].size()==8,"8cells");OriginalOwner original;
    Dl_info provider{};need(dladdr(reinterpret_cast<void*>(original.cpu->iface.graph_compute),&provider),"O provider");
    Dl_info priv{};need(dladdr(reinterpret_cast<void*>(state.private_cpu->iface.graph_compute),&priv),"C provider");need(std::filesystem::canonical(provider.dli_fname)==std::filesystem::canonical(priv.dli_fname),"O/C CPU provider");
    json report={{"passed",false},{"callback",nullptr},{"collector","closed"},{"cells",json::array()},
        {"boundary","host frozen x/IDs ready to canonical host unweighted6 output; O independent FFN graph, NOT main scheduler"},
        {"GPU_ready",nullptr},{"CPU_provider",std::filesystem::canonical(provider.dli_fname).string()},
        {"threads",private_threads},{"O_C_affinity",physical_ids},{"resident_ids_in_physical_order",state.layers[7]->selected},{"bank_native_bytes",state.payload},{"bank_allocated_bytes",state.allocation},
        {"private_initialized_layers",json::array()},{"empty_layers_bypassed",json::array()},{"GPU37_39_unchanged",true},{"GPU_operation_count_scope","source call counts, not hardware trace"}};
    for(auto &lp:state.layers)if(lp){if(lp->selected.empty()){need(!lp->cpu,"empty private skipped");if(lp->index<37)report["empty_layers_bypassed"].push_back(lp->index);}else{need(bool(lp->cpu),"positive private initialized");report["private_initialized_layers"].push_back(lp->index);}}
    report["O_worker_receipt"]=verify_workers(original.cpu);report["CSP_worker_receipt"]=verify_workers(state.private_cpu);
    size_t verified=0;
    for(const auto &lp:state.layers)if(lp&&!lp->selected.empty())for(int p=0;p<3;++p){auto w=lp->original[p];size_t bytes=ggml_row_size(w->type,w->ne[0])*w->ne[1];std::vector<char> got(ggml_nbytes(lp->gpu_weights[p]));
        ggml_backend_tensor_get(lp->gpu_weights[p],got.data(),0,got.size());for(size_t e=0;e<lp->selected.size();++e)need(!std::memcmp(got.data()+e*bytes,static_cast<const char*>(w->data)+size_t(lp->selected[e])*w->nb[2],bytes),"bank uploaded bytes");verified+=got.size();}
    need(verified==state.payload,"all native bytes verified");report["bank_native_bytes_verified"]=verified;
    save_json(out/"REPLAY.json",report);
    for(const auto &c:meta["cells"]){int ci=c["cell"],li=c["layer"];if(ci>2) continue;need(li==7,"fixed layer");auto &l=*state.layers[li];need(!l.selected.empty(),"selected cell layer");
        std::array<float,D>x{};std::array<int32_t,TOP>ids{};std::array<float,TOP>w{};std::vector<float>captured(D*TOP),result(D*TOP);
        read_data(fixture/c["x_file"].get<std::string>(),x.data(),x.size());read_data(fixture/c["ids_file"].get<std::string>(),ids.data(),ids.size());
        read_data(fixture/c["routing_weights_file"].get<std::string>(),w.data(),w.size());read_data(fixture/c["down_file"].get<std::string>(),captured.data(),captured.size());
        need(json(ids)==c["ids"]&&l.clamp==c["clamp"],"fixture ids/clamp");auto compact=compact_metadata(model,li,ids);need(compact==c["projections"],"compact/shape/provider matches captured model");
        Graph graph(original.cpu,l.original,TOP,l.clamp);
        // Host input/output tensors used by the actual private dispatcher. All allocated before timing.
        ggml_context *tc=ggml_init({65536,nullptr,true});need(tc,"replay host metadata");
        auto tx=ggml_new_tensor_3d(tc,GGML_TYPE_F32,D,1,1);auto ti=ggml_new_tensor_2d(tc,GGML_TYPE_I32,TOP,1);auto ty=ggml_new_tensor_2d(tc,GGML_TYPE_F32,D,TOP);
        auto buf=ggml_backend_alloc_ctx_tensors(tc,original.cpu);need(buf,"replay host data");
        ggml_backend_tensor_set(tx,x.data(),0,sizeof(x));ggml_backend_tensor_set(ti,ids.data(),0,sizeof(ids));ty->src[0]=tx;ty->src[1]=ti;
        std::array<std::vector<float>,4>baseline;for(auto &v:baseline)v.resize(D*TOP);
        int matched=0;std::vector<int>slots;for(int j=0;j<TOP;++j)if(l.local[ids[j]]>=0){++matched;slots.push_back(j);}
        json cell={{"cell",ci},{"layer",li},{"matched_H",matched},{"gpu_slots",slots},{"ids",ids},{"compact",compact},{"samples",json::array()},{"correctness",json::object()}};
        auto run=[&](int mode){SampleStamp t;t.begin=hot_ns();
            if(mode==0){t.matched=matched;ggml_backend_tensor_set(graph.x,x.data(),0,sizeof(x));ggml_backend_tensor_set(graph.ids,ids.data(),0,sizeof(ids));t.cpu_set_end=hot_ns();t.cpu_begin=hot_ns();
                need(ggml_backend_graph_compute(original.cpu,graph.graph)==GGML_STATUS_SUCCESS,"O graph");t.cpu_end=hot_ns();}
            else {std::lock_guard<std::mutex>lock(state.execution_mutex);state.compute(l,ty,"OCSP"[mode],&t);}
            t.end=hot_ns();
            // Extraction is a test artifact: private output is already canonical in ty at t.end.
            if(mode)std::memcpy(result.data(),ty->data,result.size()*sizeof(float));
            else ggml_backend_tensor_get(graph.y,result.data(),0,result.size()*sizeof(float));
            need(t.executed==((mode>=2)?matched:0),"executed H");need(t.gpu_ops==(t.executed?5:0),"GPU operations only on hits");return t;};
        auto save_stamp=[&](int mode,int block,int order,bool first,const SampleStamp&t){cell["samples"].push_back({{"mode",std::string(1,"OCSP"[mode])},{"block",block},{"order",order},{"first_use",first},
            {"input_ready_ns",t.begin},{"output_ready_ns",t.end},{"partition_end_ns",t.partition_end},{"cpu_set_end_ns",t.cpu_set_end},{"submit_end_ns",t.submit_end},
            {"cpu_begin_ns",t.cpu_begin},{"cpu_end_ns",t.cpu_end},{"join_begin_ns",t.join_begin},{"join_end_ns",t.join_end},{"matched_H",t.matched},{"executed_gpu_H",t.executed},{"gpu_operations",t.gpu_ops}});};
        for(int m=0;m<4;++m){auto t=run(m);baseline[m]=result;save_stamp(m,-1,m,true,t);write_f32(out/("cell"+std::to_string(ci)+"-"+std::string(1,"OCSP"[m])+".f32"),result);}
        auto equal=[](const auto&a,const auto&b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(float));};
        cell["correctness"]["O_C"]=err_metrics(baseline[1],baseline[0]);cell["correctness"]["S_P"]=err_metrics(baseline[3],baseline[2]);cell["correctness"]["O_capture"]=err_metrics(baseline[0],captured);
        cell["correctness"]["P_O"]=err_metrics(baseline[3],baseline[0]);cell["correctness"]["P_O_weighted"]=err_metrics(merge_slots(baseline[3],w),merge_slots(baseline[0],w));
        cell["correctness"]["S_P_weighted_byte_equal"]=equal(merge_slots(baseline[3],w),merge_slots(baseline[2],w));
        cell["correctness"]["per_slot_P_O"]=json::array();for(int j=0;j<TOP;++j){std::vector<float>a(baseline[3].begin()+j*D,baseline[3].begin()+(j+1)*D),b(baseline[0].begin()+j*D,baseline[0].begin()+(j+1)*D);cell["correctness"]["per_slot_P_O"].push_back(err_metrics(a,b));if(l.local[ids[j]]<0)need(equal(a,b),"cold output exact");}
        save_json(out/("cell"+std::to_string(ci)+".json"),cell);
        need(equal(baseline[0],baseline[1])&&equal(baseline[2],baseline[3])&&equal(baseline[0],captured)&&cell["correctness"]["S_P_weighted_byte_equal"]==true,"operator numeric gates");
        for(int warm=0;warm<4;++warm)for(int m=0;m<4;++m){run(m);need(equal(result,baseline[m]),"warmup reproducible");}
        const int sequence[4][4]={{0,1,3,2},{1,2,0,3},{2,3,1,0},{3,0,2,1}};
        for(int block=0;block<12;++block)for(int order=0;order<4;++order){int m=sequence[block%4][order];auto t=run(m);save_stamp(m,block,order,false,t);
            if(!equal(result,baseline[m])){cell["failed_sample"]={{"block",block},{"order",order},{"mode",std::string(1,"OCSP"[m])}};write_f32(out/("cell"+std::to_string(ci)+"-failed.f32"),result);save_json(out/("cell"+std::to_string(ci)+".json"),cell);throw std::runtime_error("timed output reproducibility failed");}}
        need(compact_metadata(model,li,ids)==compact,"compact unchanged");cell["passed"]=true;save_json(out/("cell"+std::to_string(ci)+".json"),cell);report["cells"].push_back(cell);save_json(out/"REPLAY.json",report);
        ggml_backend_buffer_free(buf);ggml_free(tc);std::cout<<"operator cell "<<ci<<" H="<<matched<<" passed\n"<<std::flush;
    }
    report["passed"]=true;save_json(out/"REPLAY.json",report);
}

// Keep allocation and GPU slot constant. Replace only the third resident
// expert between periods, after all native output is ready. All three
// projections of every resident expert are read back by operator_replay.
static void set_third_resident(State &state, int id) {
    Layer &layer=*state.layers.at(7);
    need(id==85 || id==331,"predeclared assignments");
    need(layer.selected.size()==3 && layer.selected.at(0)==19 && layer.selected.at(1)==154,"L7 bank geometry");
    const auto old=layer.selected.at(2);
    if (old==static_cast<unsigned>(id)) return;
    need(layer.local.at(old)==2 && layer.local.at(id)==-1,"prior assignment");
    ggml_backend_synchronize(state.private_gpu);
    for (int i=0;i<3;++i) {
        auto w=layer.original[i];
        const size_t n=ggml_row_size(w->type,w->ne[0])*w->ne[1];
        ggml_backend_tensor_set(layer.gpu_weights[i],static_cast<const char*>(w->data)+size_t(id)*w->nb[2],2*n,n);
    }
    ggml_backend_synchronize(state.private_gpu);
    layer.local.at(old)=-1;
    layer.local.at(id)=2;
    layer.selected.at(2)=id;
}

static void operator_periods(ggml_backend_t bank,const llama_model &model,const std::filesystem::path &out,
                             const std::filesystem::path &fixture) {
    auto &state=*static_cast<State*>(bank->context);
    const std::array<int,4> assignments{331,85,85,331};
    for (size_t i=0;i<assignments.size();++i) {
        set_third_resident(state,assignments[i]);
        const auto dir=out/("period-"+std::to_string(i+1));
        need(std::filesystem::create_directory(dir),"new period directory");
        operator_replay(bank,model,dir,fixture);
    }
    // Native GGUF + F64 accumulation reference outside all O/C/S/P clocks.
    const auto meta=read_json(fixture/"CAPTURE.json");
    for (int c=0;c<3;++c) {
        auto &row=meta["cells"][c];
        std::array<int32_t,TOP>ids{};
        std::array<float,D>x{};
        read_data(fixture/row["ids_file"].get<std::string>(),ids.data(),ids.size());
        read_data(fixture/row["x_file"].get<std::string>(),x.data(),x.size());
        write_f32(out/("cell"+std::to_string(c)+"-f64ref.f32"),reference_ffn(*state.layers.at(7),ids,x));
    }
}
