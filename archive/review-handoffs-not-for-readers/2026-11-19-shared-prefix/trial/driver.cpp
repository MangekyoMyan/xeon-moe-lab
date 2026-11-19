#include "common.h"
#include "arg.h"
#include "log.h"
#include "speculative.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-backend-impl.h"
#include "mtmd.h"
#include "json.hpp"
#include <dlfcn.h>
#include <sys/stat.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
static void need(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static json read_json(const fs::path &p){std::ifstream f(p);need(bool(f),"JSON open");json j;f>>j;return j;}
static void save(const fs::path &p,const json &j){auto t=p;t+=".tmp";std::ofstream f(t);need(bool(f),"JSON output");f<<j.dump(2)<<'\n';f.close();need(!f.fail(),"JSON write");fs::rename(t,p);}
static json providers(){
    json a=json::array();
    for(const char *symbol:{"llama_decode","llama_get_logits_ith","ggml_backend_cpu_init","ggml_backend_sched_graph_compute_async","xeon_iq3_cache_total_v1"}){
        void *p=dlsym(RTLD_DEFAULT,symbol);Dl_info i{};need(p&&dladdr(p,&i)&&i.dli_fname,"provider lookup");auto path=fs::canonical(i.dli_fname);struct stat s{};need(!stat(path.c_str(),&s),"provider stat");
        a.push_back({{"symbol",symbol},{"path",path.string()},{"device",uint64_t(s.st_dev)},{"inode",uint64_t(s.st_ino)},{"bytes",uint64_t(s.st_size)}});
    }return a;
}
static common_speculative_output_limits limits(const common_params &p){
    if(!p.mmproj.path.empty()&&mtmd_get_cap_from_file(p.mmproj.path.c_str()).gen_audio)return {p.n_batch,p.n_batch};
    if(p.embedding||(p.pooling_type!=LLAMA_POOLING_TYPE_UNSPECIFIED&&p.pooling_type!=LLAMA_POOLING_TYPE_NONE))return {p.n_batch,1};
    auto r=common_speculative_get_output_limits(p.n_batch,p.n_parallel,common_speculative_n_max(&p.speculative));r.total=std::max(1,r.total);r.per_seq=std::max(1,r.per_seq);return r;
}
struct Resources{
    common_init_result_ptr init;mtmd_context *vision=nullptr;llama_batch batch{};bool batched=false;
    ~Resources(){if(vision)mtmd_free(vision);if(batched)llama_batch_free(batch);init.reset();}
};
int main(int argc,char **argv){
    Resources resources;fs::path out,cases_path,spec_path;llama_context *ctx=nullptr;
    json report={{"schema",1},{"passed",false},{"blocks",json::array()}};
    try{
        if(argc==3&&std::string(argv[1])=="--check-providers"){need(!fs::exists(argv[2]),"fresh receipt");save(argv[2],{{"model_loaded",false},{"providers",providers()}});return 0;}
        std::vector<char*>args{argv[0]};
        for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--study-output"||a=="--study-cases"||a=="--run-spec"){need(i+1<argc,"argument value");std::string v=argv[++i];if(a=="--study-output")out=v;else if(a=="--study-cases")cases_path=v;else spec_path=v;}else args.push_back(argv[i]);}
        need(fs::is_directory(out)&&!fs::exists(out/"RUN.json"),"fresh run");auto pin=read_json(out/"PROVIDER_EXPECTED.json");auto spec=read_json(spec_path);auto cases=read_json(cases_path);
        report["spec"]=spec;report["providers"]=providers();need(report["providers"]==pin["providers"],"provider pins");
        const std::string stage=spec["stage"];need(stage=="baseline"||stage=="observe"||stage=="timing"||stage=="quality","stage");
        const bool candidate=spec["arm"]=="B";need(candidate||spec["arm"]=="A","arm");
        auto trace_initialize=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"s169_trace_initialize"));if(stage=="observe")need(trace_initialize&&trace_initialize()==0,"early observer init");else need(!trace_initialize,"no observer in control");
        common_params p;need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"args");need(!p.cb_eval,"no eval callback");p.cb_eval=nullptr;p.cb_eval_user_data=nullptr;p.warmup=false;p.sampling.backend_sampling=false;auto cap=limits(p);p.n_outputs_max=cap.total;p.n_outputs_max_per_seq=cap.per_seq;
        need(p.n_ctx==131072&&p.n_batch==1024&&p.n_ubatch==1024&&p.n_parallel==1&&p.cpuparams.n_threads==12&&p.cpuparams_batch.n_threads==24&&p.no_extra_bufts&&!p.no_op_offload&&p.moe_cache_size==0&&cap.total==1&&cap.per_seq==1,"normal fixed profile");
        
        common_log_set_verbosity_thold(LOG_LEVEL_INFO);llama_log_set(common_log_default_callback,nullptr);llama_backend_init();resources.init=common_init_from_params(p);need(resources.init&&resources.init->context()&&resources.init->model(),"init");ctx=resources.init->context();auto *model=resources.init->model();auto sched=ctx->get_sched();const auto &cp=ctx->get_cparams();
        need(cp.n_ctx==131072&&cp.n_ubatch==1024&&cp.n_outputs_max==1&&cp.n_outputs_max_per_seq==1&&cp.n_rs_seq==0,"actual context");report["context"]={{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_outputs_max",cp.n_outputs_max},{"n_rs_seq",cp.n_rs_seq},{"threads",12},{"threads_batch",24}};
        report["active_backend_providers"]=json::array();
        for(int i=0;i<ggml_backend_sched_get_n_backends(sched);++i){auto b=ggml_backend_sched_get_backend(sched,i);Dl_info info{};need(dladdr(reinterpret_cast<void*>(b->iface.graph_compute),&info)&&info.dli_fname,"actual backend");report["active_backend_providers"].push_back({{"name",ggml_backend_name(b)},{"graph_compute_provider",fs::canonical(info.dli_fname).string()}});}
        need(report["active_backend_providers"]==pin["active_backend_providers"],"actual backend pins");need(model->layers.size()==40,"layers");
        using prefix_fn=int(*)(ggml_backend_sched_t,uint64_t*,size_t);auto prefix_stats=reinterpret_cast<prefix_fn>(dlsym(RTLD_DEFAULT,"xeon_shared_prefix_stats_v1"));need(bool(prefix_stats)==candidate,"prefix stats provider presence");
        if(prefix_stats){Dl_info info{};need(dladdr(reinterpret_cast<void*>(prefix_stats),&info)&&info.dli_fname,"prefix provider");report["prefix_provider"]=fs::canonical(info.dli_fname).string();for(const auto &v:pin["providers"])if(v["symbol"]=="ggml_backend_sched_graph_compute_async")need(report["prefix_provider"]==v["path"],"prefix same base provider");}
        auto snapshot=[&](){std::array<uint64_t,7>a{};if(prefix_stats){need(prefix_stats(sched,a.data(),a.size())==0&&a[0]==1,"prefix snapshot");need(a[4]==0,"no failure drains");}return a;};
        auto delta=[&](const std::array<uint64_t,7>&a,const std::array<uint64_t,7>&b,int count){if(candidate){need(b[2]-a[2]==uint64_t(37*count)&&b[3]-a[3]==uint64_t(37*count),"exact per-layer prefix invocations");if(count)need(b[5]>=27648&&b[5]<=65536&&b[6]==37,"scratch bounds and all37 eligibility");}};

        report["placement"]=json::array();for(int l=0;l<40;++l){auto &layer=model->layers[l];for(auto *w:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps})need(w&&w->buffer&&ggml_backend_buffer_is_host(w->buffer)==(l<37),"routed placement");report["placement"].push_back({{"layer",l},{"routed_host",l<37}});}
        report["shared_inventory"]=json::array();for(int l=0;l<37;++l){auto &a=model->layers[l];json ws=json::array();int k=0;for(auto *w:{a.ffn_gate_shexp,a.ffn_up_shexp,a.ffn_down_shexp}){need(w&&w->buffer&&!ggml_backend_buffer_is_host(w->buffer),"shared device resident");need(std::string(ggml_backend_buffer_name(w->buffer))=="CUDA0","shared CUDA0 buffer");need(w->type==(k==1?GGML_TYPE_Q8_0:GGML_TYPE_Q6_K),"shared type");need(w->ne[0]==(k<2?5120:2304)&&w->ne[1]==(k<2?2304:5120)&&w->ne[2]==1&&w->ne[3]==1,"shared shape");ws.push_back({{"name",w->name},{"type",ggml_type_name(w->type)},{"ne",std::vector<int64_t>(w->ne,w->ne+4)},{"buffer",ggml_backend_buffer_name(w->buffer)}});++k;}report["shared_inventory"].push_back({{"layer",l},{"weights",ws}});}
        auto total=reinterpret_cast<size_t(*)()>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_total_v1"));need(total&&total()==127401984000ULL,"actual compact cache");report["compact_bytes"]=total();
        need(!p.mmproj.path.empty()&&p.mmproj_use_gpu,"normal vision");auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.device=p.mmproj_device;vp.n_threads=p.cpuparams.n_threads;vp.flash_attn_type=p.flash_attn_type;vp.image_min_tokens=p.image_min_tokens;vp.image_max_tokens=p.image_max_tokens;vp.batch_max_tokens=p.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;resources.vision=mtmd_init_from_file(p.mmproj.path.c_str(),model,vp);need(resources.vision,"vision init");report["vision_loaded"]=true;
        const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));need(nv==129280,"vocab");report["vocab_size"]=nv;need(cases["cases"].size()==2,"two training cases");
        const std::array<std::string,2>names={"text_cache_budget_ja","code_stable_grouping_python"};const std::array<int,2>lengths={272,240};std::array<std::vector<llama_token>,2>prefix,target;
        for(int c=0;c<2;++c){auto &v=cases["cases"][c];prefix[c]=v["prefix_ids"].get<std::vector<llama_token>>();target[c]=v["next_targets_prefix_and64T1"].get<std::vector<llama_token>>();need(v["case_id"]==names[c]&&v["split"]=="train"&&prefix[c].size()==size_t(lengths[c])&&target[c].size()==65,"frozen case");auto in=v["used_T1_ids"].get<std::vector<llama_token>>();need(in.size()==64&&std::equal(in.begin(),in.end(),target[c].begin()),"teacher inputs");for(auto t:prefix[c])need(t>=0&&t<nv,"prefix token");for(auto t:target[c])need(t>=0&&t<nv,"target token");}
        resources.batch=llama_batch_init(1024,0,1);resources.batched=true;auto &batch=resources.batch;
        auto set=[&](const llama_token *ids,int n,int pos){need(n>0&&n<=1024,"batch");batch.n_tokens=n;for(int i=0;i<n;++i){batch.token[i]=ids[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i==n-1;}};
        auto compute=[&](){need(llama_decode(ctx,batch)==0,"decode");const float *q=llama_get_logits_ith(ctx,-1);need(q,"output ready");return q;};auto finite=[&](const float *q){for(int i=0;i<nv;++i)need(std::isfinite(q[i]),"finite logits");};
        auto clear=[&](){llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
        // Identical normal warmup in every process; no observer active here.
        set(prefix[0].data(),64,0);finite(compute());clear();set(prefix[0].data(),prefix[0].size(),0);finite(compute());for(int i=0;i<4;++i){set(&target[0][i],1,lengths[0]+i);finite(compute());}llama_synchronize(ctx);clear();report["preparation"]={{"JA_first64",true},{"JA_full_prefix",lengths[0]},{"JA_reference_T1",4}};
        // Observer setup/flush stays outside token clocks; no added graph synchronization.
        using window_fn=int(*)();using call_fn=int(*)(unsigned,unsigned);using dump_fn=int(*)(const char*);
        auto window_begin=reinterpret_cast<window_fn>(dlsym(RTLD_DEFAULT,"s169_trace_window_begin"));auto begin=reinterpret_cast<call_fn>(dlsym(RTLD_DEFAULT,"s169_trace_call_begin"));auto end=reinterpret_cast<window_fn>(dlsym(RTLD_DEFAULT,"s169_trace_call_end"));auto window_end=reinterpret_cast<window_fn>(dlsym(RTLD_DEFAULT,"s169_trace_window_end"));auto dump=reinterpret_cast<dump_fn>(dlsym(RTLD_DEFAULT,"s169_trace_dump"));
        const bool observe=stage=="observe";if(observe)need(window_begin&&begin&&end&&window_end&&dump,"observer interface");else need(!window_begin&&!begin&&!end&&!window_end&&!dump,"observer absent in control/timing");
        std::array<bool,2>seen{};for(auto case_id:spec["cases"]){int c=case_id.get<int>();need(c>=0&&c<2&&!seen[c],"case order");seen[c]=true;clear();auto before_prefix=snapshot();set(prefix[c].data(),prefix[c].size(),0);const float*q=compute();finite(q);auto before_T1=snapshot();delta(before_prefix,before_T1,0);json call_stats=json::array();
            const bool quality=stage!="timing";std::vector<float>outputs(quality?size_t(nv)*65:0);if(quality)std::memcpy(outputs.data(),q,nv*sizeof(float));std::array<int64_t,64>starts{},ends{};std::array<int,64>codes{};
            for(int i=0;i<64;++i){auto previous=quality?snapshot():std::array<uint64_t,7>{};set(&target[c][i],1,lengths[c]+i);bool traced=observe&&i<4;if(observe&&i==0)need(window_begin()==0,"trace window begin");if(traced)need(begin(c,i+1)==0,"trace begin");starts[i]=ns();codes[i]=llama_decode(ctx,batch);q=llama_get_logits_ith(ctx,-1);ends[i]=ns();if(traced)need(end()==0,"trace end");if(observe&&i==3)need(window_end()==0,"trace window end");need(codes[i]==0&&q,"T1 status");if(quality){auto current=snapshot();delta(previous,current,1);call_stats.push_back(current);finite(q);std::memcpy(outputs.data()+size_t(i+1)*nv,q,nv*sizeof(float));}}
            llama_synchronize(ctx);auto after_T1=snapshot();delta(before_T1,after_T1,64);if(!quality){finite(q);outputs.assign(q,q+nv);}std::string filename=names[c]+".f32";std::ofstream f(out/filename,std::ios::binary);need(bool(f),"F32 open");f.write(reinterpret_cast<const char*>(outputs.data()),outputs.size()*sizeof(float));f.close();need(!f.fail(),"F32 write");
            report["blocks"].push_back({{"case_id",names[c]},{"case_index",c},{"split","train"},{"prefix_ids",prefix[c]},{"positions_begin",lengths[c]},{"input_tokens",std::vector<llama_token>(target[c].begin(),target[c].begin()+64)},{"start_ns",starts},{"output_ready_ns",ends},{"status",codes},{"file",filename},{"rows",quality?65:1},{"all_finite",true},{"traced_calls",observe?4:0},{"prefix_stats_before_prefill",before_prefix},{"prefix_stats_before_T1",before_T1},{"prefix_stats_after_T1",after_T1},{"per_call_prefix_stats",call_stats}});save(out/"RUN.json",report);std::cout<<names[c]<<" complete\n"<<std::flush;
        }need(seen[0]&&seen[1],"both cases");if(observe)need(dump(out.c_str())==0,"trace dump");report["passed"]=true;save(out/"RUN.json",report);return 0;
    }catch(const std::exception &e){if(ctx)llama_synchronize(ctx);if(!out.empty()&&!fs::exists(out/"trace169-summary.json")){auto partial=reinterpret_cast<int(*)(const char*)>(dlsym(RTLD_DEFAULT,"s169_trace_dump"));if(partial){report["partial_trace_dump_attempted"]=true;report["partial_trace_dump_status"]=partial(out.c_str());}}report["error"]=e.what();if(!out.empty())try{save(out/"RUN.json",report);}catch(...){}std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
