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
static void validate_ledger(const json &t,int prefix,int count,bool prefill) {
    const auto &s=t["stats"];
    need(s["flags"]==0&&s["dropped_events"]==0&&s["enabled"]==false,"closed ledger flags");
    const int calls=prefill?1:count;
    need(s["calls"]==calls&&t["calls"].size()==size_t(calls),"call count");
    need(s["observing"]==!prefill&&s["records"]==(prefill?0:37*count)&&s["complete_decode_attempts"]==(prefill?0:count),"record count");
    int position=0;
    for(size_t i=0;i<t["attempts"].size();++i){
        const auto &a=t["attempts"][i];int nt=a["n_tokens"];
        need(nt>0&&a["id"]==i+1&&a["context"]==1&&a["flags"]==0&&a["graph_status"]==0&&a["graph_returned"]==true&&a["caller_completed"]==true,"attempt completed");
        need(a["observing"]==!prefill&&a["position"]==(prefill?position:prefix+int(i)),"attempt position");
        need(a["label"]["phase"]==(prefill?XR_PREFILL:XR_DECODE)&&a["label"]["decode_call"]==(prefill?0:int(i))&&a["label"]["sequence"]==0&&a["label"]["request"]==0,"attempt label");
        if(prefill){need(a["records"]==0&&a["layers"]==0&&a["ubatch"]==i,"prefill attempt");position+=nt;}
        else{need(nt==1&&a["ubatch"]==0&&a["first_record"]==37*i&&a["records"]==37&&a["layers"]==((UINT64_C(1)<<37)-1),"T1 coverage");}
    }
    need(s["attempts"]==t["attempts"].size(),"attempt stats");
    if(prefill)need(position==prefix,"actual ubatch sum");else need(t["attempts"].size()==size_t(count),"T1 attempts");
    for(int i=0;i<calls;++i){const auto &c=t["calls"][i];
        need(c["flags"]==0&&c["public_or_driver_status"]==0&&c["synchronized"]==true&&c["completed"]==true,"public call status");
        need(c["attempts"]==(prefill?t["attempts"].size():size_t(1))&&c["first_attempt"]==(prefill?0:i),"call attempts");
    }
    if(!prefill)for(int i=0;i<count;++i){std::array<bool,37>seen{};
        for(int j=37*i;j<37*(i+1);++j){const auto &v=t["records"][j];int l=v["layer"];
            need(l>=0&&l<37&&!seen[l],"unique layer");seen[l]=true;
            need(v["ordinal"]==j&&v["attempt_id"]==i+1&&v["position"]==prefix+i&&v["flags"]==0&&v["phase"]==XR_DECODE&&v["n_tokens"]==1&&v["projection"]==1,"T1 record");
            need(v["request"]==0&&v["sequence"]==0&&v["context"]==1&&v["decode_call"]==i&&v["logical_input_token"]==i&&v["ubatch"]==0&&v["generation"]==t["attempts"][i]["generation"],"record metadata");
            std::vector<int>ids=v["ids"];std::sort(ids.begin(),ids.end());need(ids.size()==6&&ids.front()>=0&&ids.back()<384&&std::adjacent_find(ids.begin(),ids.end())==ids.end(),"route IDs");
        }
    }
}

int main(int argc,char **argv) {
    Resources resources;std::unique_ptr<API>api;llama_context *ctx=nullptr;bool session=false;
    std::filesystem::path out,cases_path,old_reference;
    json report={{"schema",1},{"passed",false},{"memory_phases",json::array()},{"cases",json::array()}};
    try{
        if(argc==3&&std::string(argv[1])=="--check-providers"){
            need(!std::filesystem::exists(argv[2]),"fresh output");save_json(argv[2],{{"model_loaded",false},{"providers",providers(true)}});return 0;
        }
        std::vector<char *>args{argv[0]};
        for(int i=1;i<argc;++i){std::string a=argv[i];
            if(a=="--study-output"||a=="--study-cases"||a=="--wiring-reference"){
                need(i+1<argc,"study value");std::string value=argv[++i];
                if(a=="--study-output")out=value;
                if(a=="--study-cases")cases_path=value;
                if(a=="--wiring-reference")old_reference=value;
            }else args.push_back(argv[i]);
        }
        need(!out.empty()&&std::filesystem::is_directory(out)&&!std::filesystem::exists(out/"RUN.json"),"fresh directory");
        auto pin=read_json(out/"PROVIDER_EXPECTED.json");report["providers"]=providers(true);need(report["providers"]==pin["providers"],"pinned providers");
        common_params params;need(common_params_parse(args.size(),args.data(),params,LLAMA_EXAMPLE_SERVER),"parse arguments");
        params.warmup=false;params.sampling.backend_sampling=false;auto limits=output_limits(params);params.n_outputs_max=limits.total;params.n_outputs_max_per_seq=limits.per_seq;
        need(params.n_ctx==131072&&params.n_batch==1024&&(params.n_ubatch==1024||params.n_ubatch==512||params.n_ubatch==256)&&params.n_parallel==1&&params.cpuparams.n_threads==12&&params.cpuparams_batch.n_threads==24&&params.no_extra_bufts&&!params.no_op_offload&&params.moe_cache_size==0&&limits.total==1&&limits.per_seq==1,"fixed profile");
        report["requested_ubatch"]=params.n_ubatch;report["scope"]="Bank-free first5 training route capture; full prompt plus64 teacher-forced reference inputs; first5 output vectors saved, all64 finite-scanned; no holdout or sampled quality";save_json(out/"RUN.json",report);
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
        report["memory_scope"]="M0 is a bank-free corresponding point, not actual Gpre/Gpost admission. M1 vision warmup builds/reserves dummy shape, not real-image compute. CUDA then NVML are sequential queries; M3 after five training cases and exports is retained allocation, not instantaneous peak.";
        auto memory=[&](const char *phase){
            llama_synchronize(ctx);for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i)ggml_backend_synchronize(ggml_backend_sched_get_backend(scheduler,i));
            size_t free=0,total=0;auto begin=ns();ggml_backend_dev_memory(gpu,&free,&total);auto cuda_end=ns();
            const char *cmd="/usr/bin/timeout 8s /usr/bin/nvidia-smi -i GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3 --query-gpu=timestamp,uuid,pci.bus_id,memory.total,memory.used,memory.free --format=csv,nounits,noheader 2>&1";
            auto nv_begin=ns();std::string raw;FILE *p=popen(cmd,"r");need(p,"NVML query pipe");char buf[512];while(fgets(buf,sizeof(buf),p))raw+=buf;int code=pclose(p);auto nv_end=ns();
            report["memory_phases"].push_back({{"phase",phase},{"cuda",{{"begin_ns",begin},{"end_ns",cuda_end},{"free_bytes",free},{"total_bytes",total},{"API","ggml_backend_dev_memory"}}},{"nvml_cli",{{"begin_ns",nv_begin},{"end_ns",nv_end},{"status",code},{"command",cmd},{"raw_csv",raw},{"memory_unit","MiB"}}},{"bank_allocated",false}});
            save_json(out/"RUN.json",report);need(code==0,"NVML milestone query failed");std::cout<<"memory "<<phase<<" CUDA free MiB="<<free/1048576.0<<'\n'<<std::flush;
        };
        memory("M0_after_context_before_vision");
        need(!params.mmproj.path.empty()&&params.mmproj_use_gpu,"vision allocation");auto vp=mtmd_context_params_default();vp.use_gpu=true;vp.device=params.mmproj_device;vp.n_threads=params.cpuparams.n_threads;vp.flash_attn_type=params.flash_attn_type;vp.image_min_tokens=params.image_min_tokens;vp.image_max_tokens=params.image_max_tokens;vp.batch_max_tokens=params.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;
        resources.vision=mtmd_init_from_file(params.mmproj.path.c_str(),model,vp);need(resources.vision,"vision initialization");report["vision_weights_loaded"]=true;memory("M1_after_vision_warmup");
        const auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);need(nv==129280,"vocab");report["vocab_size"]=nv;
        auto input=read_json(cases_path);std::vector<std::vector<llama_token>>tokens;report["tokenizer"]={{"form","raw controls; pilot inputs are pre-rendered API chat template plus reference, no re-rendering in caller"},{"add_special",true},{"parse_special",true}};
        for(const auto &c:input["cases"]){std::string text=c["text_utf8"];int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);need(n>=(c.contains("prefix_tokens")?c["prefix_tokens"].get<int>()+65:1088),"enough tokens for full prefix and65 targets");std::vector<llama_token>tok(n);need(llama_tokenize(vocab,text.data(),text.size(),tok.data(),n,true,true)==n,"tokenizer");need(json(tok)==c["expected_token_ids"],"token IDs differ from pinned API tokenizer");report["cases"].push_back({{"case_id",c["case_id"]},{"text_sha256",c["sha256_utf8"]},{"full_tokens",tok},{"full_token_count",n}});tokens.push_back(std::move(tok));}
        need(tokens.size()==7&&!old_reference.empty(),"two capacity controls and five training cases");auto old=read_json(old_reference/"RUN.json");need(old["passed"]==true&&old["arm"]=="F-before"&&old["cases"][0]["full_tokens"]==report["cases"][0]["full_tokens"],"old wiring input identity");
        resources.batch=llama_batch_init(1024,0,1);resources.has_batch=true;auto &batch=resources.batch;api=std::make_unique<API>();
        auto set_batch=[&](const std::vector<llama_token>&tok,int first,int count){need(count<=1024,"logical batch bound");batch.n_tokens=count;for(int i=0;i<count;++i){batch.token[i]=tok[first+i];batch.pos[i]=first+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==count-1);}};
        auto finite=[&](const float *p){if(!p)return false;for(int i=0;i<nv;++i)if(!std::isfinite(p[i]))return false;return true;};
        set_batch(tokens[0],0,64);int rc=llama_decode(ctx,batch);if(rc){llama_synchronize(ctx);throw std::runtime_error("warmup decode");}need(finite(llama_get_logits_ith(ctx,-1)),"warmup finite");memory("M2_after_text64_warmup");
        auto output_rows=[&](const std::string &name,int ci,int prefix,int count=4){
            need(count==4||count==64,"bounded continuation count");
            std::vector<float> values(size_t(nv)*5);std::vector<Step>steps(count);const float *ptr=nullptr;
            llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);set_batch(tokens[ci],0,prefix);
            need(api->open(ctx,1,0,4096,512,128),"prefill session");session=true;
            xeon_route37_label tag{};tag.phase=XR_PREFILL;std::array<int64_t,4> ts{};
            auto pr=drive_call(*api,ctx,tag,[&](){ts[0]=ns();int z=llama_decode(ctx,batch);ts[1]=ns();return z;},[&](){ptr=llama_get_logits_ith(ctx,-1);ts[2]=ns();bool ok=finite(ptr);ts[3]=ns();return ok;},[&](){llama_synchronize(ctx);});
            if(!pr.synchronized&&pr.label_accepted)resources.abandon=true;
            need(pr.ok(),"prefill result");need(api->close(ctx),"prefill close");session=false;auto prefill_trace=snapshot(*api);validate_ledger(prefill_trace,prefix,1,true);
            std::memcpy(values.data(),ptr,nv*sizeof(float));
            json b={{"case_id",input["cases"][ci]["case_id"]},{"name",name},{"prefix_tokens",prefix},{"continuation_inputs",std::vector<llama_token>(tokens[ci].begin()+prefix,tokens[ci].begin()+prefix+count)},{"next_token_targets_prefix_and_all_T1",std::vector<llama_token>(tokens[ci].begin()+prefix,tokens[ci].begin()+prefix+count+1)},{"logical_prefill_output_positions",{prefix-1}},{"prefill_trace",prefill_trace},{"prefill_decode_begin_ns",ts[0]},{"prefill_decode_return_ns",ts[1]},{"prefill_output_ready_ns",ts[2]},{"prefill_finite_end_ns",ts[3]},{"actual_ubatches",json::array()},{"expected_ubatches",json::array()},{"T1",json::array()}};
            for(const auto &a:prefill_trace["attempts"])b["actual_ubatches"].push_back(a["n_tokens"]);
            for(int rem=prefix;rem>0;){int n=std::min(rem,params.n_ubatch);b["expected_ubatches"].push_back(n);rem-=n;}
            need(b["actual_ubatches"]==b["expected_ubatches"],"actual split differs; investigate, do not round");
            need(api->open(ctx,1,1,4096,512,128),"T1 session");session=true;
            for(int i=0;i<count;++i){auto &s=steps[i];s.stamp[0]=ns();set_batch(tokens[ci],prefix+i,1);xeon_route37_label label{};label.phase=XR_DECODE;label.decode_call=i;label.logical_token=i;
                s.result=drive_call(*api,ctx,label,[&](){s.stamp[1]=ns();int z=llama_decode(ctx,batch);s.stamp[2]=ns();return z;},[&](){ptr=llama_get_logits_ith(ctx,-1);s.stamp[3]=ns();bool ok=finite(ptr);s.stamp[4]=ns();return ok;},[&](){llama_synchronize(ctx);});s.stamp[5]=ns();
                if(!s.result.synchronized&&s.result.label_accepted)resources.abandon=true;
                need(s.result.ok(),"T1 result");if(i<4)std::memcpy(values.data()+size_t(i+1)*nv,ptr,nv*sizeof(float));s.stamp[6]=ns();
            }
            need(api->close(ctx),"T1 close");session=false;b["T1_trace"]=snapshot(*api);validate_ledger(b["T1_trace"],prefix,count,false);
            for(int i=0;i<count;++i)b["T1"].push_back({{"index",i},{"position",prefix+i},{"next_target_token_id",tokens[ci][prefix+i+1]},{"boundaries_ns",steps[i].stamp},{"decode_status",steps[i].result.decode_status},{"full_logits_finite",true},{"synchronized",steps[i].result.synchronized}});
            b["next_token_targets_for_saved_rows"]=std::vector<llama_token>(tokens[ci].begin()+prefix,tokens[ci].begin()+prefix+5);b["all_saved_logits_finite"]=true;b["logits_shape"]={5,nv};b["logits_type"]="native little-endian F32; row0 final prefix, rows1..4 after first four continuation inputs; later T1 F32 vectors not saved";
            std::ofstream f(out/(name+".f32"),std::ios::binary);need(bool(f),"logits open");f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));f.close();need(!f.fail(),"logits write");b["logits_file"]=name+".f32";
            if(prefix==64&&ci==0){std::ifstream fref(old_reference/"correct_F.f32",std::ios::binary|std::ios::ate);need(bool(fref)&&size_t(fref.tellg())==size_t(4)*nv*sizeof(float),"old logits size");std::vector<float>old_values(size_t(4)*nv);fref.seekg(0);fref.read(reinterpret_cast<char*>(old_values.data()),old_values.size()*sizeof(float));need(bool(fref),"old logits read");b["T1_byte_equal_previous_F_before"]=std::memcmp(old_values.data(),values.data()+nv,old_values.size()*sizeof(float))==0;save_json(out/(name+".json"),b);need(b["T1_byte_equal_previous_F_before"]==true,"short64 numeric gate");}
            save_json(out/(name+".json"),b);report["completed_cases"].push_back(name);save_json(out/"RUN.json",report);std::cout<<"completed "<<name<<" prefix="<<prefix<<'\n'<<std::flush;
        };
        report["completed_cases"]=json::array();output_rows("short64",0,64);for(int ci=2;ci<7;++ci)output_rows(input["cases"][ci]["case_id"].get<std::string>(),ci,input["cases"][ci]["prefix_tokens"],64);memory("M3_after_train_first5");
        report["passed"]=true;report["session_closed_before_owner_destruction"]=!session;save_json(out/"RUN.json",report);
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}if(resources.has_batch){llama_batch_free(batch);resources.has_batch=false;}resources.init.reset();llama_backend_free();std::cout<<"capacity stage passed\n";return 0;
    }catch(const std::exception &e){report["error"]=e.what();try{if(ctx&&!resources.abandon)llama_synchronize(ctx);if(session&&api){try{report["failure_trace"]=snapshot(*api);}catch(...){}if(!api->close(ctx))resources.abandon=true;else session=false;}}catch(...){resources.abandon=true;report["cleanup_incomplete"]=true;}report["unsafe_cleanup_avoided"]=resources.abandon;if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
