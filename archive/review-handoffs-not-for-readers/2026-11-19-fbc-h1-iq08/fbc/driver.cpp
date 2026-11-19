#include "fbc-trace.h"
#include "log.h"
#include "route_client.h"
#include "call_control.h"
#include "common.h"
#include "arg.h"
#include "speculative.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"
#include "mtmd.h"
#include "hot-backend.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <map>
#include <set>

static int64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static json read_json(const std::filesystem::path &p) {
    std::ifstream f(p); need(bool(f), "input JSON open"); json j; f >> j; return j;
}
// Same output reservation rule as tools/server/server-context.cpp:48-67.
static common_speculative_output_limits output_limits(const common_params &p) {
    if (!p.mmproj.path.empty() && mtmd_get_cap_from_file(p.mmproj.path.c_str()).gen_audio)
        return {p.n_batch, p.n_batch};
    if (p.embedding || (p.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && p.pooling_type != LLAMA_POOLING_TYPE_NONE))
        return {p.n_batch, 1};
    auto result = common_speculative_get_output_limits(p.n_batch, p.n_parallel, common_speculative_n_max(&p.speculative));
    result.total = std::max(1, result.total); result.per_seq = std::max(1, result.per_seq); return result;
}
struct Resources {
    common_init_result_ptr init;
    mtmd_context *vision = nullptr;
    ggml_backend_t capacity_bank = nullptr;
    llama_batch batch{};
    bool has_batch = false, abandon = false;
    ~Resources() {
        if (abandon) { init.release(); return; }
        if (capacity_bank) ggml_backend_free(capacity_bank);
        if (vision) mtmd_free(vision);
        if (has_batch) llama_batch_free(batch);
        init.reset();
    }
};
int main(int argc,char **argv) {
    Resources resources;std::unique_ptr<API>api;llama_context *ctx=nullptr;
    std::filesystem::path out,cases_path,spec_path;std::string arm;
    json report={{"schema",1},{"passed",false},{"memory_phases",json::array()},{"cases",json::array()}};
    try{
        if(argc==3&&std::string(argv[1])=="--check-providers"){
            need(!std::filesystem::exists(argv[2]),"fresh output");save_json(argv[2],{{"model_loaded",false},{"providers",providers(true)}});return 0;
        }
        std::vector<char *>args{argv[0]};
        for(int i=1;i<argc;++i){std::string a=argv[i];
            if(a=="--study-output"||a=="--study-cases"||a=="--run-spec"){
                need(i+1<argc,"study value");std::string value=argv[++i];
                if(a=="--study-output")out=value;
                if(a=="--study-cases")cases_path=value;
                if(a=="--run-spec")spec_path=value;
            }else args.push_back(argv[i]);
        }
        need(!out.empty()&&std::filesystem::is_directory(out)&&!std::filesystem::exists(out/"RUN.json"),"fresh directory");
        auto pin=read_json(out/"PROVIDER_EXPECTED.json");report["providers"]=providers(true);need(report["providers"]==pin["providers"],"pinned providers");
        auto spec=read_json(spec_path);arm=spec["mode"].get<std::string>();std::string stage=spec["stage"];
        need((stage=="quality"||stage=="timing")&&(arm=="F"||arm=="B"||arm=="C"),"FBC declaration");
        const bool has_bank=arm!="F",uses_custom=arm=="C";
        report["spec"]=spec;report["arm"]=arm;report["stage"]=stage;report["has_bank"]=has_bank;report["uses_custom"]=uses_custom;
        const char *flag=std::getenv("GGML_XEON_HOT_TRIAL_KEEP_ORIGINAL_GRAPH");
        need(flag&&std::string(flag)==(arm=="B"?"1":"0"),"context-fixed B flag");
        common_params params;need(common_params_parse(args.size(),args.data(),params,LLAMA_EXAMPLE_SERVER),"parse arguments");
        need(!params.cb_eval,"no callback in replay");params.cb_eval=nullptr;params.cb_eval_user_data=nullptr;params.warmup=false;params.sampling.backend_sampling=false;auto limits=output_limits(params);params.n_outputs_max=limits.total;params.n_outputs_max_per_seq=limits.per_seq;
        need(params.n_ctx==131072&&params.n_batch==1024&&params.n_ubatch==1024&&params.n_parallel==1&&params.cpuparams.n_threads==12&&params.cpuparams_batch.n_threads==24&&params.no_extra_bufts&&!params.no_op_offload&&params.moe_cache_size==0&&limits.total==1&&limits.per_seq==1,"fixed profile");
        report["requested_ubatch"]=params.n_ubatch;report["scope"]="Same new FBC engine; fixed original35, diagnosticCPU602a, trainingJA/Python. B retains bank but original main graph. No adoption.";save_json(out/"RUN.json",report);
        common_log_set_verbosity_thold(stage=="quality"?LOG_LEVEL_DEBUG:LOG_LEVEL_INFO);llama_log_set(common_log_default_callback,nullptr);
        llama_backend_init();resources.init=common_init_from_params(params);need(resources.init&&resources.init->context()&&resources.init->model(),"model initialization");
        ctx=resources.init->context();auto *model=resources.init->model();const auto &cp=ctx->get_cparams();
        report["cparams"]={{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_seq_max",cp.n_seq_max},{"n_outputs_max",cp.n_outputs_max},{"n_outputs_max_per_seq",cp.n_outputs_max_per_seq},{"threads",12},{"threads_batch",24},{"n_rs_seq",cp.n_rs_seq}};
        need(cp.n_ubatch==uint32_t(params.n_ubatch)&&cp.n_outputs_max==1&&cp.n_rs_seq==0,"actual cparams");
        auto scheduler=ctx->get_sched();ggml_backend_dev_t gpu=nullptr;report["active_backend_providers"]=json::array();
        for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto b=ggml_backend_sched_get_backend(scheduler,i);Dl_info info{};
            need(dladdr(reinterpret_cast<void *>(b->iface.graph_compute),&info)&&info.dli_fname,"backend provider");
            report["active_backend_providers"].push_back({{"name",ggml_backend_name(b)},{"graph_compute_provider",std::filesystem::canonical(info.dli_fname).string()}});
            if(ggml_backend_dev_type(ggml_backend_get_device(b))==GGML_BACKEND_DEVICE_TYPE_GPU){need(!gpu,"singleGPU");gpu=ggml_backend_get_device(b);}
        }
        need(gpu&&report["active_backend_providers"]==pin["active_backend_providers"],"actual scheduler pin");
        need(model->layers.size()==40,"40 layers");report["placement"]=json::array();
        for(int i=0;i<40;++i){auto &l=model->layers[i];for(auto *w:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps})need(w&&w->buffer&&ggml_backend_buffer_is_host(w->buffer)==(i<37),"expert placement");report["placement"].push_back({{"layer",i},{"host",i<37}});}
        report["native_expert_inventory"]=json::array();
        for(int i=0;i<40;++i){const auto &l=model->layers[i];json item={{"layer",i},{"host",i<37},{"projections",json::array()}};size_t total=0;
            for(auto *w:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps}){size_t n=ggml_row_size(w->type,w->ne[0])*w->ne[1];total+=n;item["projections"].push_back({{"name",w->name},{"type",ggml_type_name(w->type)},{"ne",std::vector<int64_t>(w->ne,w->ne+4)},{"native_expert_bytes",n},{"source_expert_stride_bytes",w->nb[2]},{"contiguous",ggml_is_contiguous(w)}});}
            item["native_triplet_bytes"]=total;report["native_expert_inventory"].push_back(item);
        }
        report["memory_scope"]="M0 is after context initialization, including hot initialization for B/C. Gpre/Gpost remain logged in hot-backend guards. Bank custom path is used only for C T1, executed GPU H0. M1 vision warmup builds/reserves dummy shape, not real-image compute. CUDA then NVML are sequential queries; M3 after main-model T1 is retained allocation, not instantaneous peak.";
        auto memory=[&](const char *phase){
            llama_synchronize(ctx);for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i)ggml_backend_synchronize(ggml_backend_sched_get_backend(scheduler,i));
            size_t free=0,total=0;auto begin=ns();ggml_backend_dev_memory(gpu,&free,&total);auto cuda_end=ns();
            const char *cmd="/usr/bin/timeout 8s /usr/bin/nvidia-smi -i GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3 --query-gpu=timestamp,uuid,pci.bus_id,memory.total,memory.used,memory.free --format=csv,nounits,noheader 2>&1";
            auto nv_begin=ns();std::string raw;FILE *p=popen(cmd,"r");need(p,"NVML query pipe");char buf[512];while(fgets(buf,sizeof(buf),p))raw+=buf;int code=pclose(p);auto nv_end=ns();
            report["memory_phases"].push_back({{"phase",phase},{"cuda",{{"begin_ns",begin},{"end_ns",cuda_end},{"free_bytes",free},{"total_bytes",total},{"API","ggml_backend_dev_memory"}}},{"nvml_cli",{{"begin_ns",nv_begin},{"end_ns",nv_end},{"status",code},{"command",cmd},{"raw_csv",raw},{"memory_unit","MiB"}}},{"bank_allocated",has_bank}});
            save_json(out/"RUN.json",report);need(code==0,"NVML milestone query failed");std::cout<<"memory "<<phase<<" CUDA free MiB="<<free/1048576.0<<'\n'<<std::flush;
        };
        memory("M0_after_context_before_vision");
        ggml_backend_t outer_cpu=nullptr,hot=nullptr;
        for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto backend=ggml_backend_sched_get_backend(scheduler,i);if(ggml_backend_dev_type(ggml_backend_get_device(backend))==GGML_BACKEND_DEVICE_TYPE_CPU)outer_cpu=backend;if(std::string(ggml_backend_name(backend))=="XEON_HOT_FFN")hot=backend;}
        need(outer_cpu&&bool(hot)==has_bank,"integrated backend presence");
        report["bank_used_by_model"]=uses_custom;
        need(!params.mmproj.path.empty()&&params.mmproj_use_gpu,"vision allocation");auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.device=params.mmproj_device;vp.n_threads=params.cpuparams.n_threads;vp.flash_attn_type=params.flash_attn_type;vp.image_min_tokens=params.image_min_tokens;vp.image_max_tokens=params.image_max_tokens;vp.batch_max_tokens=params.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;
        resources.vision=mtmd_init_from_file(params.mmproj.path.c_str(),model,vp);need(resources.vision,"vision initialization");report["vision_weights_loaded"]=true;memory("M1_after_vision_warmup");
        const auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);need(nv==129280,"vocab");report["vocab_size"]=nv;
        auto input=read_json(cases_path);need(input["cases"].size()==2,"two training cases");
        std::vector<std::vector<llama_token>>prefixes,continuations;std::map<std::string,int>index;
        const std::array<std::string,2>names={"text_cache_budget_ja","code_stable_grouping_python"};
        const std::array<int,2>lengths={272,240};
        for(int i=0;i<2;++i){const auto &c=input["cases"][i];
            need(c["case_id"]==names[i]&&c["split"]=="train"&&c["prefix_count"]==lengths[i],"fixed training cases");
            std::vector<llama_token>pr=c["prefix_ids"],co=c["next_targets_prefix_and64T1"],in=c["used_T1_ids"];
            need(pr.size()==size_t(lengths[i])&&co.size()==65&&in.size()==64&&std::equal(in.begin(),in.end(),co.begin()),"input boundaries");
            for(auto t:pr){need(t>=0&&t<nv,"prefix token");}
            for(auto t:co){need(t>=0&&t<nv,"target token");}
            index[names[i]]=i;prefixes.push_back(pr);continuations.push_back(co);
            report["cases"].push_back({{"case_id",names[i]},{"family_id",c["family_id"]},{"split","train"},{"prefix_ids",pr},{"prefix_count",pr.size()},{"teacher_forced_input64",in},{"next_targets65",co},{"prefix_i32_sha256",c["prefix_ids_i32_sha256"]},{"input_i32_sha256",c["used_T1_ids_i32_sha256"]}});
        }
        using cache_fn=size_t(*)();auto cache_total=reinterpret_cast<cache_fn>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_total_v1"));need(cache_total,"actual cache count");
        Dl_info cache_info{};need(dladdr(reinterpret_cast<void*>(cache_total),&cache_info),"cache provider");
        report["cache_provider"]=std::filesystem::canonical(cache_info.dli_fname).string();
        std::string cache_expected;for(const auto &v:pin["providers"])if(v["symbol"]=="ggml_backend_cpu_init")cache_expected=v["path"].get<std::string>();need(report["cache_provider"]==cache_expected,"cache same CPU provider");report["cache_allocated_bytes"]=cache_total();need(cache_total()==127401984000ULL,"fixed actual compact allocation");
        const std::set<int>selected_layers={1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,19,20,22,28,29,33,35,36};
        std::vector<int>order;std::set<int>seen;for(const auto &name:spec["cases"]){need(index.count(name.get<std::string>())==1,"selected case");int i=index.at(name.get<std::string>());need(seen.insert(i).second,"duplicate case");order.push_back(i);}need(order.size()==2,"both fixed cases required");
        report["input_policy"]="Frozen prefix/continuation IDs only; no template application and no tokenization. No free answers in this diagnostic.";
        resources.batch=llama_batch_init(1024,0,1);resources.has_batch=true;auto &batch=resources.batch;api=std::make_unique<API>();
        auto closed=[&](){need(snapshot(*api)["stats"]["enabled"]==false,"collector closed; internal-origin separation not implemented");};
        auto set_batch=[&](const std::vector<llama_token>&tok,int position){need(!tok.empty()&&tok.size()<=1024,"batch bounds");batch.n_tokens=tok.size();for(int i=0;i<batch.n_tokens;++i){batch.token[i]=tok[i];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==batch.n_tokens-1);}};
        auto set_one=[&](llama_token token,int position){batch.n_tokens=1;batch.token[0]=token;batch.pos[0]=position;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=true;};
        auto finite=[&](const float *p){if(!p)return false;for(int i=0;i<nv;++i)if(!std::isfinite(p[i]))return false;return true;};
        auto stats=[&](){xeon_hot_trial_stats x{};json j;
            const int present=xeon_hot_trial_snapshot(outer_cpu,&x);need(bool(present)==has_bank,"bank state presence");
            if(has_bank){need(x.ready&&x.payload==475453440&&x.allocation>=x.payload,"bank state/payload");
                j={{"ready",x.ready},{"mode",std::string(1,x.mode)},{"custom_nodes_built",x.builds},{"native_bytes",x.payload},{"bank_allocation_bytes",x.allocation},{"calls",json::array()},{"executed_H",json::array()}};
                for(int l=0;l<40;++l){j["calls"].push_back(x.calls[l]);j["executed_H"].push_back(std::vector<uint64_t>(x.hist[l],x.hist[l]+7));if(!uses_custom){need(x.builds==0&&x.calls[l]==0,"B has no custom execution");for(auto h:x.hist[l])need(h==0,"B has no histogram calls");}}
            }return j;
        };
        auto validate_calls=[&](const json&a,const json&b,int calls){if(!has_bank)return;
            for(int l=0;l<40;++l){const uint64_t want=uses_custom&&selected_layers.count(l)?calls:0;
                need(b["calls"][l].get<uint64_t>()-a["calls"][l].get<uint64_t>()==want,"exact selected layer calls");
                for(int h=0;h<7;++h)need(b["executed_H"][l][h].get<uint64_t>()-a["executed_H"][l][h].get<uint64_t>()==(h==0?want:0),"C executed H0 only");
            }
        };
        auto select=[&](char mode){llama_synchronize(ctx);closed();need(std::string(1,mode)==arm,"single arm per context");if(uses_custom)need(xeon_hot_trial_set_mode(outer_cpu,'C')==1,"C mode");};
        report["initial_bank"]=stats();
        auto begin_prefix=[&](int ci){llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);set_batch(prefixes[ci],0);};
        auto compute=[&](){int rc=llama_decode(ctx,batch);need(rc==0,"decode failed");const float *p=llama_get_logits_ith(ctx,-1);need(p,"normal output ready");return p;};
        auto write_f32=[&](const std::string &name,const std::vector<float>&v){need(!std::filesystem::exists(out/name),"no output overwrite");std::ofstream f(out/name,std::ios::binary);need(bool(f),"F32 open");f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));f.close();need(!f.fail(),"F32 write");};
        // The same preparation in every process: JA prefix64, then fullJA plus4 reference inputs.
        closed();select(arm[0]);std::vector<llama_token>warm(prefixes[0].begin(),prefixes[0].begin()+64);set_batch(warm,0);need(finite(compute()),"text64 finite");memory("M2_after_text64_warmup");
        json prep={{"case_id",names[0]},{"prefix_ids",prefixes[0]},{"T1_inputs",std::vector<llama_token>(continuations[0].begin(),continuations[0].begin()+4)},{"count",4},{"timed",false},{"before",stats()}};
        begin_prefix(0);need(finite(compute()),"preparation prefix finite");for(int i=0;i<4;++i){set_one(continuations[0][i],lengths[0]+i);need(finite(compute()),"preparation T1 finite");}llama_synchronize(ctx);prep["after"]=stats();validate_calls(prep["before"],prep["after"],4);report["preparation"]=prep;
        llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);report["blocks"]=json::array();report["answers"]=json::array();save_json(out/"RUN.json",report);
        auto block=[&](int ci,char mode,bool quality){select(mode);auto before=stats();begin_prefix(ci);auto pre0=ns();const float *ptr=compute();auto pre1=ns();need(finite(ptr),"prefix finite");auto after_prefix=stats();validate_calls(before,after_prefix,0);
            std::vector<float>values;if(quality){values.resize(size_t(65)*nv);std::memcpy(values.data(),ptr,size_t(nv)*sizeof(float));}
            std::array<int64_t,64>start{},ready{};std::array<int,64>code{};std::array<uint64_t,64>builds{};
            for(int i=0;i<64;++i){set_one(continuations[ci][i],lengths[ci]+i);
                if(quality&&i==0){common_log_flush(common_log_main());LOG_INF("FBC_TRACE_BEGIN case=%s arm=%s\n",names[ci].c_str(),arm.c_str());common_log_flush(common_log_main());xeon_fbc_trace_arm((out/(names[ci]+"-graph")).c_str());}
                start[i]=ns();code[i]=llama_decode(ctx,batch);ptr=llama_get_logits_ith(ctx,-1);ready[i]=ns();need(code[i]==0&&ptr,"bounded T1 status");if(quality&&i==0){need(!xeon_fbc_trace_pending(),"trace produced without forcing reuse");common_log_flush(common_log_main());LOG_INF("FBC_TRACE_END case=%s arm=%s\n",names[ci].c_str(),arm.c_str());common_log_flush(common_log_main());}
                if(quality){need(finite(ptr),"quality finite");std::memcpy(values.data()+size_t(i+1)*nv,ptr,size_t(nv)*sizeof(float));if(has_bank)builds[i]=stats()["custom_nodes_built"].get<uint64_t>()-after_prefix["custom_nodes_built"].get<uint64_t>();}}
            llama_synchronize(ctx);auto after=stats();validate_calls(after_prefix,after,64);
            json b={{"case_id",names[ci]},{"family_id",input["cases"][ci]["family_id"]},{"split","train"},{"mode",std::string(1,mode)},{"quality",quality},{"prefix_ids",prefixes[ci]},
                {"prefix_count",lengths[ci]},{"input_tokens",std::vector<llama_token>(continuations[ci].begin(),continuations[ci].begin()+64)},{"next_targets",continuations[ci]},
                {"positions",[&](){std::vector<int>p;for(int i=0;i<64;++i)p.push_back(lengths[ci]+i);return p;}()},{"start_ns",start},{"output_ready_ns",ready},{"status",code},{"prefill_begin_ns",pre0},{"prefill_ready_ns",pre1},
                {"before_prefix",before},{"before_T1",after_prefix},{"after",after},{"custom_builds_after_each_T1",quality&&has_bank?json(builds):json(nullptr)}};
            if(quality){std::string name="quality-"+names[ci]+"-"+mode+".f32";write_f32(name,values);b["file"]=name;b["shape"]={65,nv};b["all_finite"]=true;}
            if(!quality){std::vector<float>last(ptr,ptr+nv);write_f32("last-"+names[ci]+".f32",last);b["last_file"]="last-"+names[ci]+".f32";}
            report["blocks"].push_back(b);save_json(out/"RUN.json",report);std::cout<<stage<<' '<<names[ci]<<' '<<mode<<" complete\n"<<std::flush;
        };
        for(int ci:order)block(ci,arm[0],stage=="quality");
        report["final_bank"]=stats();
        memory("M3_after_stage");closed();report["passed"]=true;save_json(out/"RUN.json",report);
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}if(resources.has_batch){llama_batch_free(batch);resources.has_batch=false;}resources.init.reset();llama_backend_free();std::cout<<"FBC stage passed\n";return 0;
    }catch(const std::exception&e){report["error"]=e.what();try{if(ctx&&!resources.abandon)llama_synchronize(ctx);}catch(...){resources.abandon=true;report["cleanup_incomplete"]=true;}report["unsafe_cleanup_avoided"]=resources.abandon;if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
