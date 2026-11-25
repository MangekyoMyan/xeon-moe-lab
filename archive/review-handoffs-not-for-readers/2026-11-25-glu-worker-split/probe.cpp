#include "fixture.hpp"
#include "pool24.h"
#include "ggml-cpu-impl.h"
#include "ops.h"
#include "pointwise.hpp"

static void write_file(const fs::path &p,const void *v,size_t bytes){
    std::ofstream out(p,std::ios::binary);
    out.write(static_cast<const char*>(v),bytes);
    out.close();need(bool(out),"write "+p.string());
}

static std::vector<float> read_tensor(ggml_tensor *t){
    need(t->type==GGML_TYPE_F32 && ggml_is_contiguous(t),"contiguous F32 activation");
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t,result.data(),0,result.size()*sizeof(float));
    need(std::all_of(result.begin(),result.end(),[](float f){return std::isfinite(f);}),"non-finite activation");
    return result;
}

struct Worker { int cpu=-1,nth=0,mask_count=0;long tid=0; };
static void probe_worker(ggml_tensor *dst,int ith,int nth,void *data){
    auto *workers=static_cast<Worker*>(data);
    cpu_set_t allowed; CPU_ZERO(&allowed);
    workers[ith]={sched_getcpu(),nth,sched_getaffinity(0,sizeof(allowed),&allowed)==0?CPU_COUNT(&allowed):-1,syscall(SYS_gettid)};
    static_cast<float*>(dst->data)[ith]=ith;
}

static json team(ggml_backend_t cpu){
    std::array<Worker,24> workers{};
    auto ctx=ggml_init({65536,nullptr,true});
    auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,24);ggml_tensor *args[]={x};
    auto y=ggml_custom_4d(ctx,GGML_TYPE_F32,24,1,1,1,args,1,probe_worker,24,workers.data());
    auto g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,y);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buf,"team buffer");
    need(ggml_backend_graph_compute(cpu,g)==GGML_STATUS_SUCCESS,"team");
    json r=json::array();std::set<long> tids;
    for(int i=0;i<24;++i){
        auto w=workers[i];
        need(w.nth==24 && w.cpu==physical_ids.at(i) && w.mask_count==1 && tids.insert(w.tid).second,"strict physical worker");
        r.push_back({{"ith",i},{"cpu",w.cpu},{"tid",w.tid}});
    }
    ggml_backend_buffer_free(buf);ggml_free(ctx);return r;
}

// The full graph calculation below uses every worker. This separate diagnostic
// calls the same DSO and the same tensors for one worker at a time, revealing
// exactly which output entries it writes, including legitimate zero values.
static json inspect_real_assignment(ggml_tensor *act,bool candidate) {
    const auto gold=read_tensor(act);
    auto *out=static_cast<float*>(act->data);
    need(act->src[0]->data!=out && act->src[1]->data!=out,"real fixture disjoint");
    std::vector<int> coverage(gold.size(),0),counts(24,0);
    for(int ith=0;ith<24;++ith){
        std::fill(out,out+gold.size(),std::numeric_limits<float>::quiet_NaN());
        ggml_compute_params params{};params.ith=ith;params.nth=24;
        ggml_compute_forward_glu(&params,act);
        for(size_t i=0;i<gold.size();++i){
            const int owner=candidate?int(i%2304)/96:int(i/2304);
            const bool written=std::isfinite(out[i]);
            need(written==(owner==ith),"real output ownership");
            if(written){
                need(std::memcmp(&out[i],&gold[i],4)==0,"real direct worker value");
                ++counts[ith];++coverage[i];
            }
        }
    }
    need(std::all_of(coverage.begin(),coverage.end(),[](int n){return n==1;}),"real exact coverage");
    std::copy(gold.begin(),gold.end(),out);
    return counts;
}

int main(int argc,char **argv) try {
    need(argc==3,"SPEC OUTDIR");
    const json spec=readj(argv[1]);
    const fs::path dir=argv[2];
    need(fs::is_directory(dir) && !fs::exists(dir/"RUN.json"),"fresh output");
    const std::string stage=spec.at("stage"), arm=spec.at("arm");
    const bool qual=stage=="quality", diag=stage=="trace", cand=arm=="G";
    need((arm=="O"||cand) && (qual||diag||stage=="timing"),"valid arm/stage");
    json result={{"stage",stage},{"arm",arm},{"entry",read_entry_topology()},{"cells",json::array()}};
    const std::string cpu_provider=provider(reinterpret_cast<void*>(ggml_backend_cpu_init));
    result["providers"]={{"cpu",cpu_provider},{"glu",provider(reinterpret_cast<void*>(ggml_compute_forward_glu))},
                         {"cache",provider(reinterpret_cast<void*>(xeon_iq3_cache_add_selected_v2))},
                         {"base",provider(reinterpret_cast<void*>(ggml_init))}};
    need(cpu_provider==fs::canonical(spec["CPU"].get<std::string>()) && result["providers"]["glu"]==cpu_provider &&
         result["providers"]["cache"]==cpu_provider && result["providers"]["base"]==spec["base"],"provider match");
    if(qual) result["synthetic"]=pointwise_cases(cand);
    const fs::path cap=spec.at("capture_json").get<std::string>();
    const auto capture=readj(cap);
    std::vector<std::unique_ptr<Shard>> shards;
    for(const auto &s:spec.at("shards")) shards.push_back(std::make_unique<Shard>(s.get<std::string>()));
    auto cpu=ggml_backend_cpu_init();need(cpu,"backend");
    auto pp=private_pool_params();auto pool=ggml_threadpool_new(&pp);need(pool,"pool");
    ggml_backend_cpu_set_n_threads(cpu,24);ggml_backend_cpu_set_threadpool(cpu,pool);
    auto begin_trace=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"xeon_worker_trace_begin"));
    auto end_trace=reinterpret_cast<int(*)(const char*)>(dlsym(RTLD_DEFAULT,"xeon_worker_trace_end"));
    need(bool(begin_trace)==diag && bool(end_trace)==diag,"instrumentation excluded from timing");
    for(int ci:spec["cells"].get<std::vector<int>>()){
        need(ci==0||ci==1,"L7 selected cases");
        auto cell=capture["cells"][ci];
        Fixture f(cpu,shards,cell,cap.parent_path(),spec.at("cpu_plan"));
        auto xb=readbytes(cap.parent_path()/cell["x_file"].get<std::string>());
        auto ib=readbytes(cap.parent_path()/cell["ids_file"].get<std::string>());
        auto ids=cell["ids"].get<std::vector<int32_t>>();
        need(xb.size()==20480 && ib.size()==24 && !std::memcmp(ib.data(),ids.data(),24),"frozen operands");
        const std::array<std::string,4> names={"gate","up","glu","down"};
        std::array<ggml_tensor*,4> nodes;
        need(ggml_graph_n_nodes(f.graph)==4,"four nodes");
        for(int i=0;i<4;++i){
            nodes[i]=ggml_graph_node(f.graph,i);
            ggml_set_name(nodes[i],names[i].c_str());
        }
        need(nodes[0]->src[0] && std::string(nodes[0]->src[0]->name)=="blk.7.ffn_gate_exps.weight" &&
             std::string(nodes[1]->src[0]->name)=="blk.7.ffn_up_exps.weight" &&
             nodes[2]->src[0]==nodes[0] && nodes[2]->src[1]==nodes[1] &&
             ggml_get_glu_op(nodes[2])==GGML_GLU_OP_SWIGLU_CLAMP && nodes[3]->src[1]==nodes[2] &&
             nodes[3]==f.y && nodes[0]->src[1]==f.x && nodes[1]->src[1]==f.x,"original graph order");
        json block=f.record;block["stamps"]=json::array();
        auto run=[&](const char *phase,int batch,int call){
            const auto lo=ns();
            ggml_backend_tensor_set(f.x,xb.data(),0,xb.size());
            ggml_backend_tensor_set(f.ids,ib.data(),0,ib.size());
            f.compute();
            const auto hi=ns();
            block["stamps"].push_back({{"phase",phase},{"block",batch},{"call",call},
                                      {"input_ready_ns",lo},{"host_down_ready_ns",hi}});
        };
        run("first",-1,0);f.output();
        for(int i=0;i<4;++i)run("warmup",-1,i);
        f.output();
        if(qual){
            for(int i=0;i<4;++i){
                auto values=read_tensor(nodes[i]);
                const auto name="cell"+std::to_string(ci)+"-"+names[i]+".f32";
                write_file(dir/name,values.data(),values.size()*4);
                block["tensors"][names[i]]={{"file",name},{"elements",values.size()},
                    {"nb1",nodes[i]->nb[1]},{"base_mod64",reinterpret_cast<uintptr_t>(nodes[i]->data)%64}};
            }
            block["actual_written_per_worker"]=inspect_real_assignment(nodes[2],cand);
            f.output();
        }else if(diag){
            for(int i=0;i<spec.at("trace_calls").get<int>();++i){
                begin_trace();run("trace",0,i);
                const auto path=dir/("cell"+std::to_string(ci)+"-"+std::to_string(i)+".tsv");
                need(end_trace(path.c_str())==1,"one CPU graph traced");f.output();
            }
        }else{
            for(int b=0;b<spec.at("blocks").get<int>();++b){
                for(int i=0;i<spec.at("per_block").get<int>();++i)run("timed",b,i);
                f.output();
            }
        }
        block["matched_saved_down"]=true;
        result["cells"].push_back(std::move(block));
    }
    result["team_after_runs"]=team(cpu);
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);
    result["passed"]=true;
    std::ofstream f(dir/"RUN.json");f<<result.dump(2)<<'\n';f.close();need(bool(f),"receipt write");
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
