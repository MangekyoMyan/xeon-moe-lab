#include "route_client.h"
#include "call_control.h"
#include "common.h"
#include "arg.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"
#include "mtmd.h"
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

static bool coverage(const json &t,size_t calls){
    if(t["stats"]["flags"]!=0||t["stats"]["dropped_events"]!=0||t["stats"]["records"]!=calls*37||t["stats"]["attempts"]!=calls||t["stats"]["calls"]!=calls||t["stats"]["complete_decode_attempts"]!=calls)return false;
    for(size_t i=0;i<calls;++i){
        const auto &a=t["attempts"][i],&c=t["calls"][i];
        if(a["id"]!=i+1||a["records"]!=37||a["layers"]!=((UINT64_C(1)<<37)-1)||a["flags"]!=0||a["label"]["phase"]!=XR_DECODE||a["graph_returned"]!=true||a["caller_completed"]!=true||a["n_tokens"]!=1||c["attempts"]!=1||c["flags"]!=0||c["public_or_driver_status"]!=0||c["completed"]!=true||c["synchronized"]!=true)return false;
        if(c["label"]["request"]!=0||c["label"]["sequence"]!=0||c["label"]["phase"]!=XR_DECODE||c["label"]["decode_call"]!=i||c["label"]["logical_input_token"]!=i||a["label"]!=c["label"]||a["position"]!=64+i||a["context"]!=1||a["ubatch"]!=0||a["first_record"]!=i*37||c["first_attempt"]!=i)return false;
        std::array<bool,37> seen{};
        for(size_t j=i*37;j<(i+1)*37;++j){const auto&r=t["records"][j];int l=r["layer"];
            if(l<0||l>=37||seen[l]||r["attempt_id"]!=i+1||r["phase"]!=XR_DECODE||r["flags"]!=0)return false;
            if(r["ordinal"]!=j||r["position"]!=a["position"]||r["generation"]!=a["generation"]||r["sequence"]!=0||r["request"]!=0||r["context"]!=1||r["decode_call"]!=i||r["logical_input_token"]!=i||r["n_tokens"]!=1||r["projection"]!=1||r["ubatch"]!=0)return false;
            seen[l]=true;
            std::vector<int> ids=r["ids"];if(ids.size()!=6)return false;std::sort(ids.begin(),ids.end());if(ids[0]<0||ids.back()>=384||std::adjacent_find(ids.begin(),ids.end())!=ids.end())return false;
        }
    }
    return true;
}
struct Resources {
    common_init_result_ptr init;mtmd_context *vision=nullptr;llama_batch batch{};bool has_batch=false,abandon=false;
    ~Resources(){
        if(abandon){init.release();return;} // no user-space frees of possibly undrained device work
        if(vision)mtmd_free(vision);
        if(has_batch)llama_batch_free(batch);
        init.reset();
    }
};
int main(int argc,char **argv){
    std::filesystem::path out;bool continue64=false;json report={{"schema",1},{"passed",false},{"model_routes",false},{"call_results",json::array()}};
    Resources resources;API *api=nullptr;bool session=false,backend_started=false;llama_context *ctx=nullptr;
    try{
        if(argc==3&&std::string(argv[1])=="--check-providers"){
            need(!std::filesystem::exists(argv[2]),"fresh provider output required");
            save_json(argv[2],{{"backend_init_called",false},{"model_loaded",false},{"providers",providers()}});return 0;
        }
        std::vector<char*> args{argv[0]};
        for(int i=1;i<argc;++i){std::string x=argv[i];if(x=="--route-output"){need(i+1<argc,"route output");out=argv[++i];}else if(x=="--route-continue64")continue64=true;else args.push_back(argv[i]);}
        need(!out.empty()&&std::filesystem::is_directory(out)&&!std::filesystem::exists(out/"RUN.json"),"new prepared output directory required");
        report["session_id"]=out.filename().string();report["providers"]=providers();report["planned_capture_calls"]=continue64?68:4;report["scope"]="Teacher-forced text continuation; route coverage, not speed/quality or hot-bank admission";save_json(out/"RUN.json",report);
        common_params params;need(common_params_parse(args.size(),args.data(),params,LLAMA_EXAMPLE_SERVER),"model arguments");
        params.warmup=false;params.sampling.backend_sampling=false;
        need(params.n_ctx==131072&&params.n_parallel==1&&params.cpuparams.n_threads==12&&params.cpuparams_batch.n_threads==24&&params.no_extra_bufts&&!params.no_op_offload&&params.moe_cache_size==0,"diagnostic profile mismatch");
        llama_backend_init();backend_started=true;
        resources.init=common_init_from_params(params);need(resources.init&&resources.init->model()&&resources.init->context(),"model/context load");ctx=resources.init->context();auto *model=resources.init->model();
        report["active_backend_providers"]=json::array();auto scheduler=ctx->get_sched();bool cpu_seen=false;
        for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto backend=ggml_backend_sched_get_backend(scheduler,i);Dl_info info{};
            need(dladdr(reinterpret_cast<void*>(backend->iface.graph_compute),&info)&&info.dli_fname,"active backend provider");
            auto path=std::filesystem::canonical(info.dli_fname).string();auto type=ggml_backend_dev_type(ggml_backend_get_device(backend));
            if(type==GGML_BACKEND_DEVICE_TYPE_CPU){need(path==report["providers"][2]["path"].get<std::string>(),"active CPU provider differs");cpu_seen=true;}
            report["active_backend_providers"].push_back({{"name",ggml_backend_name(backend)},{"device_type",type},{"graph_compute_provider",path}});
        }
        need(cpu_seen,"active CPU backend missing");
        need(model->layers.size()==40,"model layer count");
        json placement=json::array();
        for(int i=0;i<40;++i){const auto &l=model->layers[i];for(auto *w:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps}){
            need(w&&w->buffer,"expert tensor/buffer");need(ggml_backend_buffer_is_host(w->buffer)==(i<37),"expert layer placement differs");}
            placement.push_back({{"layer",i},{"expert_weights_host",i<37}});}
        report["placement"]=placement;
        need(!params.mmproj.path.empty()&&params.mmproj_use_gpu,"preserve configured vision model allocation");
        auto vp=mtmd_context_params_default();vp.use_gpu=params.mmproj_use_gpu;vp.device=params.mmproj_device;vp.n_threads=params.cpuparams.n_threads;vp.flash_attn_type=params.flash_attn_type;vp.image_min_tokens=params.image_min_tokens;vp.image_max_tokens=params.image_max_tokens;vp.batch_max_tokens=params.mtmd_batch_max_tokens;vp.warmup=true;vp.print_timings=false;
        resources.vision=mtmd_init_from_file(params.mmproj.path.c_str(),model,vp);need(resources.vision,"vision load/warmup");report["vision_weights_loaded"]=true;
        const auto *vocab=llama_model_get_vocab(model);std::string text;
        for(int i=0;i<64;++i)text+="A careful experiment records the selected experts and preserves every observation. The same fixed input is used for the comparison. ";
        int n=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);need(n>132,"token fixture too short");std::vector<llama_token> tokens(n);need(llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),true,true)==n,"token fixture");
        const int prefix=64,total=continue64?68:4;tokens.resize(prefix+total);report["tokens"]=tokens;report["prefix_tokens"]=prefix;report["logical_token_definition"]="Zero-based index of current teacher-forced input within the continuation; position=64+logical token, not index of predicted output";
        resources.batch=llama_batch_init(prefix,0,1);resources.has_batch=true;auto &batch=resources.batch;
        auto set_batch=[&](int first,int count){batch.n_tokens=count;for(int i=0;i<count;++i){batch.token[i]=tokens[first+i];batch.pos[i]=first+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==count-1);}};
        auto output_valid=[&](){float *logits=llama_get_logits_ith(ctx,-1);if(!logits)return false;for(int i=0;i<llama_vocab_n_tokens(vocab);++i)if(!std::isfinite(logits[i]))return false;return true;};
        // Explicit warmup before opening the collector; failures are not ignored.
        set_batch(0,prefix);int rc=llama_decode(ctx,batch);if(rc){llama_synchronize(ctx);throw std::runtime_error("warmup decode failed");}need(output_valid(),"warmup output");
        llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);
        set_batch(0,prefix);rc=llama_decode(ctx,batch);if(rc){llama_synchronize(ctx);throw std::runtime_error("prefill failed");}need(output_valid(),"prefill output");
        report["starting_KV"]="Cleared after checked warmup; fixed64-token prefill synchronized; sequence0, next position64";
        API client;api=&client;need(api->open(ctx,1,1,4096,512,128),"session open");session=true;
        report["model_routes"]=true;save_json(out/"RUN.json",report);
        for(int i=0;i<total;++i){
            set_batch(prefix+i,1);xeon_route37_label tag{};tag.request=0;tag.sequence=0;tag.decode_call=i;tag.phase=XR_DECODE;tag.logical_token=i;
            auto result=drive_call(*api,ctx,tag,[&](){return llama_decode(ctx,batch);},output_valid,[&](){llama_synchronize(ctx);});
            report["call_results"].push_back({{"logical_input_token",i},{"position",prefix+i},{"label_accepted",result.label_accepted},{"decode_returned",result.decode_returned},{"decode_status",result.decode_status},{"output_called",result.output_called},{"output_valid",result.output_valid},{"synchronized",result.synchronized},{"drain_called",result.drain_called},{"finalize_called",result.finalize_called},{"finalize_ok",result.finalize_ok},{"exception_stage",result.exception_stage},{"ok",result.ok()}});
            if(!result.synchronized&&result.label_accepted){resources.abandon=true;report["unsafe_cleanup_avoided"]=true;}
            report["trace"]=snapshot(*api);save_json(out/"RUN.json",report);
            need(result.ok(),"tagged call failed; evidence retained");need(coverage(report["trace"],i+1),"route coverage/ledger failed");
            if(i==3){save_json(out/"SMOKE4.json",report);std::cout<<"route37: 4-call coverage passed (148 records)\n"<<std::flush;}
        }
        need(api->close(ctx),"close session");session=false;report["trace"]=snapshot(*api);report["session_closed_before_owner_destruction"]=true;report["passed"]=true;save_json(out/"RUN.json",report);api=nullptr;
        std::cout<<json({{"passed",true},{"calls",total},{"records",total*37},{"output",(out/"RUN.json").string()}}).dump()<<'\n';
        // Resource destruction occurs before backend release; do not destroy ctx
        // while collector ownership or device work is still open.
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}
        if(resources.has_batch){llama_batch_free(resources.batch);resources.has_batch=false;}
        resources.init.reset();llama_backend_free();return 0;
    }catch(const std::exception&e){
        report["error"]=e.what();
        // Recover API pointers independently: the local client may have unwound.
        try{if(ctx&&!resources.abandon)llama_synchronize(ctx);API client;
            if(session){report["trace"]=snapshot(client);bool closed=client.close(ctx);report["session_close_on_error"]=closed;if(!closed)resources.abandon=true;session=false;}
        }catch(...){resources.abandon=true;report["cleanup_or_snapshot_incomplete"]=true;}
        if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}
        std::cerr<<"FAIL: "<<e.what()<<'\n';
        if(resources.abandon){resources.init.release();std::cerr<<"Undrained/unfinished state retained until process exit; no manual device frees\n";}
        // backend release is deferred to process exit on failure, after resource
        // cleanup; no retry or second model load is attempted.
        (void)backend_started;return 1;
    }
}
