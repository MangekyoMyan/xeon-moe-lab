#include "route_client.h"
#include "call_control.h"
#include "common.h"
#include "arg.h"
#include "speculative.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"
#include "mtmd.h"
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
    llama_batch batch{};
    bool has_batch = false, abandon = false;
    ~Resources() {
        if (abandon) { init.release(); return; }
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
static void validate_trace(const json &t, int count, bool observing) {
    const auto &s = t["stats"];
    need(s["flags"] == 0 && s["dropped_events"] == 0 && s["enabled"] == false, "closed stats");
    need(s["observing"] == observing && s["calls"] == count && s["attempts"] == count, "ledger count");
    need(s["records"] == (observing ? count*37 : 0) && s["complete_decode_attempts"] == (observing ? count : 0), "record count");
    need(t["records"].size() == size_t(observing ? count*37 : 0) && t["calls"].size() == size_t(count) && t["attempts"].size() == size_t(count), "array count");
    for (int i = 0; i < count; ++i) {
        const auto &a = t["attempts"][i], &c = t["calls"][i];
        json label = {{"request",0},{"sequence",0},{"decode_call",i},{"phase",XR_DECODE},{"logical_input_token",i}};
        need(a["label"] == label && c["label"] == label, "labels");
        need(a["id"] == i+1 && a["position"] == 64+i && a["context"] == 1 && a["ubatch"] == 0 && a["n_tokens"] == 1, "attempt identity");
        need(a["flags"] == 0 && a["graph_status"] == 0 && a["graph_returned"] == true && a["caller_completed"] == true && a["scope_t1"] == true && a["observing"] == observing, "attempt completion");
        need(c["first_attempt"] == i && c["attempts"] == 1 && c["flags"] == 0 && c["public_or_driver_status"] == 0 && c["completed"] == true && c["synchronized"] == true, "call completion");
        need(a["first_record"] == (observing ? 37*i : 0) && a["records"] == (observing ? 37 : 0) && a["layers"] == (observing ? (UINT64_C(1)<<37)-1 : 0), "attempt coverage");
        if (!observing) continue;
        std::array<bool,37> seen{};
        for (int j = 37*i; j < 37*(i+1); ++j) {
            const auto &v = t["records"][j]; int layer = v["layer"];
            need(layer >= 0 && layer < 37 && !seen[layer], "unique layer"); seen[layer] = true;
            need(v["ordinal"] == j && v["attempt_id"] == i+1 && v["position"] == 64+i && v["generation"] == a["generation"] && v["context"] == 1, "record identity");
            need(v["request"] == 0 && v["sequence"] == 0 && v["decode_call"] == i && v["logical_input_token"] == i && v["ubatch"] == 0 && v["phase"] == XR_DECODE && v["projection"] == 1 && v["n_tokens"] == 1 && v["flags"] == 0, "record tags");
            std::vector<int> ids = v["ids"]; std::sort(ids.begin(),ids.end());
            need(ids.size() == 6 && ids.front() >= 0 && ids.back() < 384 && std::adjacent_find(ids.begin(),ids.end()) == ids.end(), "expert IDs");
        }
    }
}

int main(int argc, char **argv) {
    Resources resources; std::unique_ptr<API> api; InactiveAPI inactive;
    llama_context *ctx = nullptr; bool session = false;
    std::filesystem::path out, cases_path, reference;
    std::string arm;
    json report = {{"schema",1},{"passed",false},{"blocks",json::array()}};
    try {
        if (argc == 4 && std::string(argv[1]) == "--check-providers") {
            need(!std::filesystem::exists(argv[3]), "new provider output");
            save_json(argv[3], {{"model_loaded",false},{"providers",providers(std::string(argv[2]) == "D")}}); return 0;
        }
        std::vector<char *> args{argv[0]};
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--study-output" || a == "--study-arm" || a == "--study-cases" || a == "--study-reference") {
                need(i+1 < argc, "study argument value"); std::string value = argv[++i];
                if (a == "--study-output") out = value;
                if (a == "--study-arm") arm = value;
                if (a == "--study-cases") cases_path = value;
                if (a == "--study-reference") reference = value;
            } else args.push_back(argv[i]);
        }
        const bool diagnostic = arm == "D";
        need(diagnostic || arm == "F-before" || arm == "F-after", "explicit arm");
        need(!out.empty() && std::filesystem::is_directory(out) && !std::filesystem::exists(out/"RUN.json"), "fresh output");
        need(arm == "F-before" || !reference.empty(), "frozen reference required");
        report["arm"] = arm; report["providers"] = providers(diagnostic);
        const auto provider_pin = read_json(out/"PROVIDER_EXPECTED.json");
        need(provider_pin["arm"] == arm && report["providers"] == provider_pin["providers"], "actual function providers differ from pinned arm");
        report["scope"] = "Fixed teacher-forced numerical check, caller overhead timing, and six raw-text calibration cases. No sampled answers, quality or hot-bank test.";
        report["clock"] = "steady_clock nanoseconds; host boundaries only";
        save_json(out/"RUN.json",report);
        common_params params;
        need(common_params_parse(args.size(),args.data(),params,LLAMA_EXAMPLE_SERVER), "model arguments");
        params.warmup = false; params.sampling.backend_sampling = false;
        auto limits = output_limits(params);
        params.n_outputs_max = limits.total; params.n_outputs_max_per_seq = limits.per_seq;
        need(params.n_ctx == 131072 && params.n_parallel == 1 && params.cpuparams.n_threads == 12 && params.cpuparams_batch.n_threads == 24 && params.no_extra_bufts && !params.no_op_offload && params.moe_cache_size == 0, "profile");
        need(limits.total == 1 && limits.per_seq == 1, "unsupported output reservation profile");
        llama_backend_init();
        resources.init = common_init_from_params(params);
        need(resources.init && resources.init->context() && resources.init->model(), "model initialization");
        ctx = resources.init->context(); auto *model = resources.init->model();
        const auto &cp = ctx->get_cparams();
        report["cparams"] = {{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_seq_max",cp.n_seq_max},{"n_outputs_max",cp.n_outputs_max},{"n_outputs_max_per_seq",cp.n_outputs_max_per_seq},{"n_threads",params.cpuparams.n_threads},{"n_threads_batch",params.cpuparams_batch.n_threads},{"backend_sampling",false}};
        need(cp.n_outputs_max == 1 && cp.n_outputs_max_per_seq == 1, "actual output cap");
        auto scheduler = ctx->get_sched(); report["active_backend_providers"] = json::array();
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto backend=ggml_backend_sched_get_backend(scheduler,i); Dl_info info{};
            need(dladdr(reinterpret_cast<void *>(backend->iface.graph_compute),&info)&&info.dli_fname,"backend provider");
            report["active_backend_providers"].push_back({{"name",ggml_backend_name(backend)},{"graph_compute_provider",std::filesystem::canonical(info.dli_fname).string()}});
        }
        need(report["active_backend_providers"] == provider_pin["active_backend_providers"], "active scheduler providers differ from pinned arm");
        report["placement"] = json::array(); need(model->layers.size()==40,"40 layers");
        for (int i=0;i<40;++i) {
            auto &l=model->layers[i];
            for (auto *w:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps}) need(w&&w->buffer&&ggml_backend_buffer_is_host(w->buffer)==(i<37),"expert placement");
            report["placement"].push_back({{"layer",i},{"expert_weights_host",i<37}});
        }
        need(!params.mmproj.path.empty()&&params.mmproj_use_gpu,"vision allocation");
        auto vp=mtmd_context_params_default(); vp.use_gpu=true; vp.device=params.mmproj_device;
        vp.n_threads=params.cpuparams.n_threads; vp.flash_attn_type=params.flash_attn_type;
        vp.image_min_tokens=params.image_min_tokens; vp.image_max_tokens=params.image_max_tokens; vp.batch_max_tokens=params.mtmd_batch_max_tokens;
        vp.warmup=true; vp.print_timings=false;
        resources.vision=mtmd_init_from_file(params.mmproj.path.c_str(),model,vp); need(resources.vision,"vision initialization");
        report["vision_weights_loaded"]=true;
        const auto *vocab=llama_model_get_vocab(model); const int nv=llama_vocab_n_tokens(vocab);
        need(nv==129280,"expected vocabulary"); report["vocab_size"]=nv;
        auto input=read_json(cases_path); std::vector<std::vector<llama_token>> tokens;
        report["tokenizer"]={{"input_form","raw text; no chat template applied"},{"add_special",true},{"parse_special",true}};
        report["cases"]=json::array();
        for (const auto &c:input["cases"]) {
            std::string text=c["text_utf8"]; int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);
            need(n>=128,"fixture too short; no repetition padding"); std::vector<llama_token> tok(n);
            need(llama_tokenize(vocab,text.data(),text.size(),tok.data(),n,true,true)==n,"tokenization");
            report["cases"].push_back({{"case_id",c["case_id"]},{"family_id",c["family_id"]},{"text_sha256",c["sha256_utf8"]},{"full_tokens",tok},{"full_token_count",n},{"used_prefix_tokens",64},{"maximum_used_continuation",64}});
            tokens.push_back(std::move(tok));
        }
        need(tokens.size()==7,"seven fixtures");
        if (!reference.empty()) {
            auto ref=read_json(reference/"RUN.json"); need(ref["passed"]==true && ref["arm"]=="F-before", "completed F-before reference required");
            need(ref["providers"] == read_json(reference/"PROVIDER_EXPECTED.json")["providers"], "reference provider pin");
            need(ref["cases"]==report["cases"] && ref["cparams"]==report["cparams"] && ref["placement"]==report["placement"],"reference inputs/profile");
        }
        if (diagnostic) api=std::make_unique<API>();
        resources.batch=llama_batch_init(64,0,1); resources.has_batch=true; auto &batch=resources.batch;
        auto set_batch=[&](const std::vector<llama_token>&tok,int first,int count){
            batch.n_tokens=count;
            for(int i=0;i<count;++i){batch.token[i]=tok[first+i];batch.pos[i]=first+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==count-1);}
        };
        auto finite=[&](const float *p){if(!p)return false;for(int i=0;i<nv;++i)if(!std::isfinite(p[i]))return false;return true;};
        // Explicit warmup before any collector session; no ignored errors.
        set_batch(tokens[0],0,64); int rc=llama_decode(ctx,batch);
        if(rc){llama_synchronize(ctx);throw std::runtime_error("warmup decode");}
        need(finite(llama_get_logits_ith(ctx,-1)),"warmup finite");
        report["memory_after_warmup"]=json::array();
        for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){
            auto backend=ggml_backend_sched_get_backend(scheduler,i); auto dev=ggml_backend_get_device(backend);
            if(ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_GPU){size_t free=0,total=0;ggml_backend_dev_memory(dev,&free,&total);report["memory_after_warmup"].push_back({{"backend",ggml_backend_name(backend)},{"free_bytes",free},{"total_bytes",total},{"API","ggml_backend_dev_memory; backend CUDA memory query, not NVML reserved accounting"}});}
        }
        save_json(out/"RUN.json",report);
        auto block=[&](const std::string &name,const std::string &mode,int ci,int count,bool numerical){
            need(count>0&&count<=64,"bounded block"); const bool collect=mode=="D1"||mode=="D2", observe=mode=="D2";
            std::vector<Step> steps(count); std::vector<float> logits(numerical?size_t(count)*nv:0);
            llama_memory_clear(llama_get_memory(ctx),true); llama_synchronize(ctx);
            set_batch(tokens[ci],0,64); auto prefill_start=ns(); int status=llama_decode(ctx,batch);
            if(status){llama_synchronize(ctx);throw std::runtime_error("prefill decode");}
            need(finite(llama_get_logits_ith(ctx,-1)),"prefill finite"); auto prefill_ready=ns();
            auto setup_start=ns();
            if(collect){need(api&&api->open(ctx,1,observe,4096,512,128),"open collector");session=true;}
            auto setup_end=ns();
            report["active_block"]=name; save_json(out/"RUN.json",report); // outside timing block
            for(int i=0;i<count;++i){
                auto &s=steps[i]; s.stamp[0]=ns(); set_batch(tokens[ci],64+i,1);
                xeon_route37_label tag{};tag.phase=XR_DECODE;tag.decode_call=i;tag.logical_token=i;
                const float *ptr=nullptr;
                auto submit=[&](){s.stamp[1]=ns();int result=llama_decode(ctx,batch);s.stamp[2]=ns();return result;};
                auto output=[&](){ptr=llama_get_logits_ith(ctx,-1);s.stamp[3]=ns();bool good=numerical?finite(ptr):ptr!=nullptr;s.stamp[4]=ns();return good;};
                auto drain=[&](){llama_synchronize(ctx);};
                s.result=collect?drive_call(*api,ctx,tag,submit,output,drain):drive_call(inactive,ctx,tag,submit,output,drain);
                s.stamp[5]=ns();
                if(!s.result.synchronized&&s.result.label_accepted)resources.abandon=true;
                if(!s.result.ok()){
                    report["failed_call"]={{"block",name},{"index",i},{"decode_status",s.result.decode_status},{"synchronized",s.result.synchronized},{"exception_stage",s.result.exception_stage}};
                    throw std::runtime_error("call failed; no retry");
                }
                if(numerical)std::memcpy(logits.data()+size_t(i)*nv,ptr,nv*sizeof(float));
                s.stamp[6]=ns();
            }
            const auto block_end=ns(); // all trace/JSON/export begins after this boundary
            json result={{"name",name},{"mode",mode},{"case_id",input["cases"][ci]["case_id"]},{"calls",count},{"prefix_tokens",64},{"position_first",64},{"position_last",63+count},{"collector_session",collect},{"observing",observe},{"output_validation",numerical?"all-vocab finite and saved F32":"synchronized pointer nonnull only; no full finite scan or logits copy"},{"prefill_begin_ns",prefill_start},{"prefill_ready_ns",prefill_ready},{"session_setup_begin_ns",setup_start},{"session_setup_end_ns",setup_end},{"block_end_ns",block_end},{"steps",json::array()}};
            if(collect){need(api->close(ctx),"close collector");session=false;result["trace"]=snapshot(*api);validate_trace(result["trace"],count,observe);}
            for(int i=0;i<count;++i){const auto &s=steps[i];result["steps"].push_back({{"index",i},{"position",64+i},{"step_begin_ns",s.stamp[0]},{"decode_begin_ns",s.stamp[1]},{"decode_return_ns",s.stamp[2]},{"output_ready_ns",s.stamp[3]},{"validation_end_ns",s.stamp[4]},{"finalize_end_ns",s.stamp[5]},{"step_end_ns",s.stamp[6]},{"decode_status",s.result.decode_status},{"synchronized",s.result.synchronized},{"collector_finalize_called",collect&&s.result.finalize_called},{"ok",s.result.ok()}});}
            if(numerical){
                auto path=out/(name+".f32");std::ofstream f(path,std::ios::binary);need(bool(f),"logits output open");f.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));f.close();need(!f.fail(),"logits write");
                result["logits_file"]=path.filename().string();result["logits_shape"]={count,nv};result["logits_type"]="native x86 little-endian F32";
                if(!reference.empty()){
                    std::ifstream fref(reference/"correct_F.f32",std::ios::binary|std::ios::ate);need(bool(fref),"reference logits open");need(size_t(fref.tellg())==logits.size()*sizeof(float),"reference logits size");std::vector<float> baseline(logits.size());fref.seekg(0);fref.read(reinterpret_cast<char*>(baseline.data()),baseline.size()*sizeof(float));need(bool(fref),"reference logits read");
                    bool equal=std::memcmp(baseline.data(),logits.data(),logits.size()*sizeof(float))==0;result["frozen_reference_byte_equal"]=equal;
                    save_json(out/(name+".json"),result);need(equal,"numerical mismatch; preserve outputs and stop before timing/screen");
                }
            }
            save_json(out/(name+".json"),result);report["blocks"].push_back({{"name",name},{"mode",mode},{"case_id",input["cases"][ci]["case_id"]},{"calls",count},{"numerical",numerical},{"passed",true},{"result_file",name+".json"}});report.erase("active_block");save_json(out/"RUN.json",report);
            std::cout<<"completed "<<name<<" calls="<<count<<'\n'<<std::flush;
        };
        if(!diagnostic){
            block("correct_F","F",0,4,true);
            block("timing_F_0","F",0,64,false);block("timing_F_1","F",0,64,false);
        }else{
            for(const auto &m:{"D0","D1","D2"})block(std::string("correct_")+m,m,0,4,true);
            int i=0;for(const auto &m:{"D0","D1","D2","D2","D1","D0"})block("timing_"+std::to_string(i++)+"_"+m,m,0,64,false);
            for(int ci=1;ci<7;++ci)block(input["cases"][ci]["case_id"].get<std::string>(),"D2",ci,64,false);
        }
        report["passed"]=true;report["session_closed_before_owner_destruction"]=!session;save_json(out/"RUN.json",report);
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}
        if(resources.has_batch){llama_batch_free(batch);resources.has_batch=false;}
        resources.init.reset();llama_backend_free();std::cout<<"study passed "<<arm<<'\n';return 0;
    }catch(const std::exception &e){
        report["error"]=e.what();
        try{if(ctx&&!resources.abandon)llama_synchronize(ctx);if(session&&api){try{report["failure_trace"]=snapshot(*api);}catch(...){}if(!api->close(ctx))resources.abandon=true;else session=false;}}
        catch(...){resources.abandon=true;report["cleanup_incomplete"]=true;}
        report["unsafe_cleanup_avoided"]=resources.abandon;
        if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
