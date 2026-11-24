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
// F/D0 take the same coordinator path without calling any collector function.
struct InactiveAPI {
    int label(const void *, const xeon_route37_label *) { return 1; }
    int finish(const void *, int, int) { return 1; }
};
struct Step {
    std::array<int64_t, 7> stamp{};
    CallResult result{};
};
#include "pool24.h"
#include "operator_replay.h"

int main(int argc,char **argv) {
    Resources resources;std::unique_ptr<API>api;llama_context *ctx=nullptr;bool session=false;
    std::filesystem::path out,cases_path,old_reference,fixture;
    json report={{"schema",1},{"passed",false},{"memory_phases",json::array()},{"cases",json::array()}};
    try{
        if(argc==3&&std::string(argv[1])=="--check-providers"){
            need(!std::filesystem::exists(argv[2]),"fresh output");save_json(argv[2],{{"model_loaded",false},{"providers",providers(true)}});return 0;
        }
        report["entry"]=read_entry_topology();
        std::vector<char *>args{argv[0]};
        for(int i=1;i<argc;++i){std::string a=argv[i];
            if(a=="--study-output"||a=="--study-cases"||a=="--wiring-reference"||a=="--fixture"){
                need(i+1<argc,"study value");std::string value=argv[++i];
                if(a=="--study-output")out=value;
                if(a=="--fixture")fixture=value;
                if(a=="--study-cases")cases_path=value;
                if(a=="--wiring-reference")old_reference=value;
            }else args.push_back(argv[i]);
        }
        need(!out.empty()&&std::filesystem::is_directory(out)&&!std::filesystem::exists(out/"RUN.json"),"fresh directory");
        auto pin=read_json(out/"PROVIDER_EXPECTED.json");report["providers"]=providers(true);need(report["providers"]==pin["providers"],"pinned providers");
        common_params params;need(common_params_parse(args.size(),args.data(),params,LLAMA_EXAMPLE_SERVER),"parse arguments");
        need(!params.cb_eval,"no callback in replay");params.cb_eval=nullptr;params.cb_eval_user_data=nullptr;params.warmup=false;params.sampling.backend_sampling=false;auto limits=output_limits(params);params.n_outputs_max=limits.total;params.n_outputs_max_per_seq=limits.per_seq;
        need(params.n_ctx==131072&&params.n_batch==1024&&(params.n_ubatch==1024||params.n_ubatch==512||params.n_ubatch==256)&&params.n_parallel==1&&params.cpuparams.n_threads==private_threads&&params.cpuparams_batch.n_threads==private_threads&&params.no_extra_bufts&&!params.no_op_offload&&params.moe_cache_size==0&&limits.total==1&&limits.per_seq==1,"fixed profile");
        need(params.cpuparams.mask_valid && params.cpuparams_batch.mask_valid && params.cpuparams.strict_cpu && params.cpuparams_batch.strict_cpu,"outer strict masks");
        for (int cpu=0;cpu<GGML_MAX_N_THREADS;++cpu) {
            const bool selected=std::find(physical_ids.begin(),physical_ids.end(),cpu)!=physical_ids.end();
            need(bool(params.cpuparams.cpumask[cpu])==selected && bool(params.cpuparams_batch.cpumask[cpu])==selected,"outer CPU mask");
        }
        report["requested_ubatch"]=params.n_ubatch;report["scope"]="Layer 7, three resident expert triplets; O/C/S/P in two memberships. Bank unused by main model; not integrated tok/s.";save_json(out/"RUN.json",report);
        llama_backend_init();resources.init=common_init_from_params(params);need(resources.init&&resources.init->context()&&resources.init->model(),"model initialization");
        ctx=resources.init->context();auto *model=resources.init->model();const auto &cp=ctx->get_cparams();
        need(llama_n_threads(ctx)==private_threads && llama_n_threads_batch(ctx)==private_threads,"actual context threads");report["cparams"]={{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_seq_max",cp.n_seq_max},{"n_outputs_max",cp.n_outputs_max},{"n_outputs_max_per_seq",cp.n_outputs_max_per_seq},{"threads",llama_n_threads(ctx)},{"threads_batch",llama_n_threads_batch(ctx)},{"n_rs_seq",cp.n_rs_seq}};
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
        report["memory_scope"]="M0 is a corresponding point. Actual Gpre/Gpost are separately recorded inside unchanged guards in hot-backend.cpp. The bank is held but is not used by the model graph. M1 vision warmup builds/reserves dummy shape, not real-image compute. CUDA then NVML are sequential queries; M3 after operator replay is retained allocation, not instantaneous peak.";
        auto memory=[&](const char *phase){
            llama_synchronize(ctx);for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i)ggml_backend_synchronize(ggml_backend_sched_get_backend(scheduler,i));
            size_t free=0,total=0;auto begin=ns();ggml_backend_dev_memory(gpu,&free,&total);auto cuda_end=ns();
            const char *cmd="/usr/bin/timeout 8s /usr/bin/nvidia-smi -i GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3 --query-gpu=timestamp,uuid,pci.bus_id,memory.total,memory.used,memory.free --format=csv,nounits,noheader 2>&1";
            auto nv_begin=ns();std::string raw;FILE *p=popen(cmd,"r");need(p,"NVML query pipe");char buf[512];while(fgets(buf,sizeof(buf),p))raw+=buf;int code=pclose(p);auto nv_end=ns();
            report["memory_phases"].push_back({{"phase",phase},{"cuda",{{"begin_ns",begin},{"end_ns",cuda_end},{"free_bytes",free},{"total_bytes",total},{"API","ggml_backend_dev_memory"}}},{"nvml_cli",{{"begin_ns",nv_begin},{"end_ns",nv_end},{"status",code},{"command",cmd},{"raw_csv",raw},{"memory_unit","MiB"}}},{"bank_allocated",resources.capacity_bank!=nullptr}});
            save_json(out/"RUN.json",report);need(code==0,"NVML milestone query failed");std::cout<<"memory "<<phase<<" CUDA free MiB="<<free/1048576.0<<'\n'<<std::flush;
        };
        memory("M0_after_context_before_vision");
        ggml_backend_t outer_cpu=nullptr,outer_gpu=nullptr;
        for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto backend=ggml_backend_sched_get_backend(scheduler,i);auto type=ggml_backend_dev_type(ggml_backend_get_device(backend));if(type==GGML_BACKEND_DEVICE_TYPE_CPU)outer_cpu=backend;if(type==GGML_BACKEND_DEVICE_TYPE_GPU)outer_gpu=backend;}
        need(outer_cpu&&outer_gpu,"outer backends for capacity fixture");
        resources.capacity_bank=xeon_hot_backend_create(*model,outer_cpu,outer_gpu,private_threads,private_threads);need(resources.capacity_bank,"capacity bank requested explicitly");
        xeon_hot_backend_finish(outer_cpu);
        report["capacity_bank_initialized"]=true;report["bank_used_by_model"]=false;
        memory("Gpost_bank_initialized_before_vision");
        need(!params.mmproj.path.empty()&&params.mmproj_use_gpu,"vision allocation");auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.device=params.mmproj_device;vp.n_threads=params.cpuparams.n_threads;report["projector_threads_argument"]=vp.n_threads;vp.flash_attn_type=params.flash_attn_type;vp.image_min_tokens=params.image_min_tokens;vp.image_max_tokens=params.image_max_tokens;vp.batch_max_tokens=params.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;
        resources.vision=mtmd_init_from_file(params.mmproj.path.c_str(),model,vp);need(resources.vision,"vision initialization");report["vision_weights_loaded"]=true;memory("M1_after_vision_warmup");
        const auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);need(nv==129280,"vocab");report["vocab_size"]=nv;
        auto input=read_json(cases_path);std::vector<std::vector<llama_token>>tokens;report["tokenizer"]={{"form","raw text, no chat template"},{"add_special",true},{"parse_special",true}};
        for(const auto &c:input["cases"]){std::string text=c["text_utf8"];int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);need(n>=1088,"at least1088 token IDs required");std::vector<llama_token>tok(n);need(llama_tokenize(vocab,text.data(),text.size(),tok.data(),n,true,true)==n,"tokenizer");need(json(tok)==c["expected_token_ids"],"token IDs differ from pinned API tokenizer");report["cases"].push_back({{"case_id",c["case_id"]},{"text_sha256",c["sha256_utf8"]},{"full_tokens",tok},{"full_token_count",n}});tokens.push_back(std::move(tok));}
        need(tokens.size()==2&&!old_reference.empty(),"two capacity fixtures and wiring reference");auto old=read_json(old_reference/"RUN.json");need(old["passed"]==true&&old["arm"]=="F-before"&&old["cases"][0]["full_tokens"]==report["cases"][0]["full_tokens"],"old wiring input identity");
        resources.batch=llama_batch_init(1024,0,1);resources.has_batch=true;auto &batch=resources.batch;api=std::make_unique<API>();
        auto set_batch=[&](const std::vector<llama_token>&tok,int first,int count){need(count<=1024,"logical batch bound");batch.n_tokens=count;for(int i=0;i<count;++i){batch.token[i]=tok[first+i];batch.pos[i]=first+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==count-1);}};
        auto finite=[&](const float *p){if(!p)return false;for(int i=0;i<nv;++i)if(!std::isfinite(p[i]))return false;return true;};
        set_batch(tokens[0],0,64);int rc=llama_decode(ctx,batch);if(rc){llama_synchronize(ctx);throw std::runtime_error("warmup decode");}need(finite(llama_get_logits_ith(ctx,-1)),"warmup finite");memory("M2_after_text64_warmup");
        need(!fixture.empty(),"verified activation fixture required");
        need(snapshot(*api)["stats"]["enabled"]==false,"collector closed for replay");
        operator_periods(resources.capacity_bank,*model,out,fixture);
        memory("M3_after_operator_replay");
        report["passed"]=true;report["session_closed_before_owner_destruction"]=!session;save_json(out/"RUN.json",report);
        if(resources.capacity_bank){ggml_backend_free(resources.capacity_bank);resources.capacity_bank=nullptr;}
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}if(resources.has_batch){llama_batch_free(batch);resources.has_batch=false;}resources.init.reset();llama_backend_free();std::cout<<"capacity stage passed\n";return 0;
    }catch(const std::exception &e){report["error"]=e.what();try{if(ctx&&!resources.abandon)llama_synchronize(ctx);if(session&&api){try{report["failure_trace"]=snapshot(*api);}catch(...){}if(!api->close(ctx))resources.abandon=true;else session=false;}}catch(...){resources.abandon=true;report["cleanup_incomplete"]=true;}report["unsafe_cleanup_avoided"]=resources.abandon;if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
