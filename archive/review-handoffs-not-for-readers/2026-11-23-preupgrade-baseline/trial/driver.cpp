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
#include "cache_inspection.h"
static json read_json(const fs::path &p){std::ifstream f(p);need(bool(f),"JSON open");json j;f>>j;return j;}
static void save(const fs::path &p,const json &j){auto t=p;t+=".tmp";std::ofstream f(t);need(bool(f),"JSON output");f<<j.dump(2)<<'\n';f.close();need(!f.fail(),"JSON write");fs::rename(t,p);}
#include "affinity_check.h"
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
    void close() {
        if (vision) { mtmd_free(vision); vision = nullptr; }
        if (batched) { llama_batch_free(batch); batched = false; }
        init.reset();
    }
    ~Resources() { close(); }
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
        const auto &condition=spec["condition"];
        const int expected_decode=condition.at("decode_threads");
        const int expected_batch=condition.at("prefill_threads");
        const int projector_threads=condition.at("projector_threads");
        const auto process_ids=condition.at("process_cpu_ids").get<std::vector<int>>();
        const auto initial_affinity=cpu_ids(0);
        need(!initial_affinity.empty() && initial_affinity==process_ids,"launcher CPU allowance");
        int policy_mode=-1;
        unsigned long policy_nodes=0;
        need(syscall(SYS_get_mempolicy,&policy_mode,&policy_nodes,sizeof(policy_nodes)*8,nullptr,0)==0 &&
             policy_mode==MPOL_INTERLEAVE && policy_nodes==3 && prctl(PR_GET_THP_DISABLE,0,0,0,0)==1,
             "entry memory policy or THP");
        report["requested"]=condition;
        report["linux_entry"]={{"pid",getpid()},{"main_tid",syscall(SYS_gettid)},{"effective_cpu_ids",initial_affinity},
                              {"mempolicy_mode",policy_mode},{"mempolicy_nodes_mask",policy_nodes},{"thp_disabled",true}};
        report["actual_decode_team_observed"]=nullptr;
        report["actual_prefill_team_observed"]=nullptr;
        report["observer_scope"]="Configured counts from context getters. Quality task counters and optional PC samples show phase activity.";
        const std::string stage=spec["stage"];
        need(stage=="quality" || stage=="timing", "stage");
        need(!std::getenv("GGML_CPU_IQ_ROW_LOCAL") && !std::getenv("GGML_CPU_IQ_AUDIT_DIR")
             && !std::getenv("LD_PRELOAD"), "ordinary process environment");
        auto total=reinterpret_cast<size_t(*)()>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_total_v1"));
        need(total && total()==0, "registry before load");
        common_params p;need(common_params_parse(args.size(),args.data(),p,LLAMA_EXAMPLE_SERVER),"args");need(!p.cb_eval,"no eval callback");p.cb_eval=nullptr;p.cb_eval_user_data=nullptr;p.warmup=false;p.sampling.backend_sampling=false;auto cap=limits(p);p.n_outputs_max=cap.total;p.n_outputs_max_per_seq=cap.per_seq;
        need(p.n_ctx==131072&&p.n_batch==1024&&p.n_ubatch==1024&&p.n_parallel==1&&p.cpuparams.n_threads==expected_decode&&p.cpuparams_batch.n_threads==expected_batch&&p.no_extra_bufts&&!p.no_op_offload&&p.moe_cache_size==0&&cap.total==1&&cap.per_seq==1,"normal fixed profile");
        
        need(p.cpuparams.strict_cpu && p.cpuparams_batch.strict_cpu && p.cpuparams.mask_valid && p.cpuparams_batch.mask_valid &&
             parsed_ids(p.cpuparams)==condition.at("decode_cpu_ids").get<std::vector<int>>() &&
             parsed_ids(p.cpuparams_batch)==condition.at("prefill_cpu_ids").get<std::vector<int>>(),"parsed strict CPU masks");
        report["parsed_cpu"]={{"decode_count",p.cpuparams.n_threads},{"prefill_count",p.cpuparams_batch.n_threads},
                              {"decode_ids",parsed_ids(p.cpuparams)},{"prefill_ids",parsed_ids(p.cpuparams_batch)},
                              {"decode_strict",p.cpuparams.strict_cpu},{"prefill_strict",p.cpuparams_batch.strict_cpu}};
        report["io_before_load"]=io_readings();
        common_log_set_verbosity_thold(LOG_LEVEL_INFO);llama_log_set(common_log_default_callback,nullptr);llama_backend_init();
        const auto load_begin = ns();
        resources.init=common_init_from_params(p);
        report["text_model_context_load_ns"] = ns()-load_begin;
        report["io_after_load"]=io_readings();
        need(resources.init&&resources.init->context()&&resources.init->model(),"init");ctx=resources.init->context();auto *model=resources.init->model();auto sched=ctx->get_sched();const auto &cp=ctx->get_cparams();
        need(llama_n_threads(ctx)==expected_decode && llama_n_threads_batch(ctx)==expected_batch && cp.n_ctx==131072&&cp.n_batch==1024&&cp.n_ubatch==1024&&cp.n_outputs_max==1&&cp.n_outputs_max_per_seq==1&&cp.n_rs_seq==0,"actual context");report["context"]={{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_outputs_max",cp.n_outputs_max},{"n_rs_seq",cp.n_rs_seq},{"threads",llama_n_threads(ctx)},{"threads_batch",llama_n_threads_batch(ctx)},{"source","llama_n_threads / llama_n_threads_batch after context construction"}};
        report["active_backend_providers"]=json::array();
        for(int i=0;i<ggml_backend_sched_get_n_backends(sched);++i){auto b=ggml_backend_sched_get_backend(sched,i);Dl_info info{};need(dladdr(reinterpret_cast<void*>(b->iface.graph_compute),&info)&&info.dli_fname,"actual backend");report["active_backend_providers"].push_back({{"name",ggml_backend_name(b)},{"graph_compute_provider",fs::canonical(info.dli_fname).string()}});}
        need(report["active_backend_providers"]==pin["active_backend_providers"],"actual backend pins");need(model->layers.size()==40,"layers");
        report["placement"]=json::array();for(int l=0;l<40;++l){auto &layer=model->layers[l];for(auto *w:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps})need(w&&w->buffer&&ggml_backend_buffer_is_host(w->buffer)==(l<37),"routed placement");report["placement"].push_back({{"layer",l},{"routed_host",l<37}});}
        report["shared_inventory"]=json::array();for(int l=0;l<37;++l){auto &a=model->layers[l];json ws=json::array();int k=0;for(auto *w:{a.ffn_gate_shexp,a.ffn_up_shexp,a.ffn_down_shexp}){need(w&&w->buffer&&!ggml_backend_buffer_is_host(w->buffer),"shared device resident");need(std::string(ggml_backend_buffer_name(w->buffer))=="CUDA0","shared CUDA0 buffer");need(w->type==(k==1?GGML_TYPE_Q8_0:GGML_TYPE_Q6_K),"shared type");need(w->ne[0]==(k<2?5120:2304)&&w->ne[1]==(k<2?2304:5120)&&w->ne[2]==1&&w->ne[3]==1,"shared shape");ws.push_back({{"name",w->name},{"type",ggml_type_name(w->type)},{"ne",std::vector<int64_t>(w->ne,w->ne+4)},{"buffer",ggml_backend_buffer_name(w->buffer)}});++k;}report["shared_inventory"].push_back({{"layer",l},{"weights",ws}});}
        need(total()==127401984000ULL,"actual compact cache");
        report["compact_bytes"]=total();
        if (stage=="quality") report["registration"] = compact_registration(*model);
        report["loaded_text_memory_KiB"] = process_memory();
        const auto vision_begin = ns();
        need(!p.mmproj.path.empty()&&p.mmproj_use_gpu,"normal vision");auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.device=p.mmproj_device;vp.n_threads=projector_threads;report["projector_init_threads_argument"]=vp.n_threads;vp.flash_attn_type=p.flash_attn_type;vp.image_min_tokens=p.image_min_tokens;vp.image_max_tokens=p.image_max_tokens;vp.batch_max_tokens=p.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;resources.vision=mtmd_init_from_file(p.mmproj.path.c_str(),model,vp);need(resources.vision,"vision init");report["vision_loaded"]=true;
        report["vision_init_ns"]=ns()-vision_begin;
        const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));need(nv==129280,"vocab");report["vocab_size"]=nv;need(cases["cases"].size()==2,"two training cases");
        const std::array<std::string,2>names={"text_cache_budget_ja","code_stable_grouping_python"};const std::array<int,2>lengths={272,240};std::array<std::vector<llama_token>,2>prefix,target;
        for(int c=0;c<2;++c){auto &v=cases["cases"][c];prefix[c]=v["prefix_ids"].get<std::vector<llama_token>>();target[c]=v["next_targets_prefix_and64T1"].get<std::vector<llama_token>>();need(v["case_id"]==names[c]&&v["split"]=="train"&&prefix[c].size()==size_t(lengths[c])&&target[c].size()==65,"frozen case");auto in=v["used_T1_ids"].get<std::vector<llama_token>>();need(in.size()==64&&std::equal(in.begin(),in.end(),target[c].begin()),"teacher inputs");for(auto t:prefix[c])need(t>=0&&t<nv,"prefix token");for(auto t:target[c])need(t>=0&&t<nv,"target token");}
        resources.batch=llama_batch_init(1024,0,1);resources.batched=true;auto &batch=resources.batch;
        auto set=[&](const llama_token *ids,int n,int pos){need(n>0&&n<=1024,"batch");batch.n_tokens=n;for(int i=0;i<n;++i){batch.token[i]=ids[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i==n-1;}};
        auto compute=[&](){need(llama_decode(ctx,batch)==0,"decode");const float *q=llama_get_logits_ith(ctx,-1);need(q,"output ready");return q;};auto finite=[&](const float *q){for(int i=0;i<nv;++i)need(std::isfinite(q[i]),"finite logits");};
        auto clear=[&](){llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
        // Identical normal warmup in every process; no observer active here.
        const auto warm_begin=ns();
        set(prefix[0].data(),64,0);finite(compute());clear();set(prefix[0].data(),prefix[0].size(),0);finite(compute());for(int i=0;i<4;++i){set(&target[0][i],1,lengths[0]+i);finite(compute());}llama_synchronize(ctx);clear();report["preparation"]={{"JA_first64",true},{"JA_full_prefix",lengths[0]},{"JA_reference_T1",4},{"warmup_start_ns",warm_begin},{"warmup_end_ns",ns()}};
        std::array<bool,2>seen{};for(auto case_id:spec["cases"]){int c=case_id.get<int>();need(c>=0&&c<2&&!seen[c],"case order");seen[c]=true;clear();set(prefix[c].data(),prefix[c].size(),0);
            const bool observe=stage=="quality" && c==0 && spec.value("quality_host_observation",false);
            if(observe) {
                await_quality_observer(out);
                report["ja_quality_task_observations"]["before_prefill"]=task_activity();
            }
            const auto prefill_begin=ns();const float*q=compute();const auto prefill_end=ns();
            if(observe) report["ja_quality_task_observations"]["after_prefill"]=task_activity();
            finite(q);
            const bool quality=stage!="timing";std::vector<float>outputs(quality?size_t(nv)*65:0);if(quality)std::memcpy(outputs.data(),q,nv*sizeof(float));std::array<int64_t,64>starts{},ends{};std::array<int,64>codes{};
            if(observe) report["ja_quality_task_observations"]["before_decode"]=task_activity();
            for(int i=0;i<64;++i){set(&target[c][i],1,lengths[c]+i);starts[i]=ns();codes[i]=llama_decode(ctx,batch);q=llama_get_logits_ith(ctx,-1);ends[i]=ns();need(codes[i]==0&&q,"T1 status");if(quality){finite(q);std::memcpy(outputs.data()+size_t(i+1)*nv,q,nv*sizeof(float));}}
            if(observe) {
                report["ja_quality_task_observations"]["after_decode"]=task_activity();
                save(out/"quality-observer-finished.json",{{"pid",getpid()},{"monotonic_ns",ns()}});
            }
            llama_synchronize(ctx);if(!quality){finite(q);outputs.assign(q,q+nv);}std::string filename=names[c]+".f32";std::ofstream f(out/filename,std::ios::binary);need(bool(f),"F32 open");f.write(reinterpret_cast<const char*>(outputs.data()),outputs.size()*sizeof(float));f.close();need(!f.fail(),"F32 write");
            report["blocks"].push_back({{"case_id",names[c]},{"case_index",c},{"split","train"},{"prefix_ids",prefix[c]},{"positions_begin",lengths[c]},{"input_tokens",std::vector<llama_token>(target[c].begin(),target[c].begin()+64)},{"start_ns",starts},{"output_ready_ns",ends},{"status",codes},{"file",filename},{"rows",quality?65:1},{"all_finite",true},{"traced_calls",0},{"prefill_begin_ns",prefill_begin},{"prefill_ready_ns",prefill_end}});save(out/"RUN.json",report);std::cout<<names[c]<<" complete\n"<<std::flush;
        }need(seen[0]&&seen[1],"both cases");report["before_unload_KiB"]=process_memory();
        llama_synchronize(ctx);
        ctx=nullptr;
        const auto unload_begin=ns();
        resources.close();
        report["unload_ns"]=ns()-unload_begin;
        report["compact_bytes_after_unload"]=total();
        need(total()==0,"registry after unload");
        report["after_unload_KiB"]=process_memory();
        report["passed"]=true;
        save(out/"RUN.json",report);
        return 0;
    } catch(const std::exception &error) {
        if(ctx) llama_synchronize(ctx);
        report["error"]=error.what();
        if(!out.empty()) try {save(out/"RUN.json",report);} catch(...) {}
        std::cerr<<"FAIL "<<error.what()<<'\n';
        return 1;
    }
}
