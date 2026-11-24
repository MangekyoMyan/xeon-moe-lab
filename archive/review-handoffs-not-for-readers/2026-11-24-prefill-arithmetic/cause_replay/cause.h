// Matched counterfactual at layer zero; all subsequent computation uses the ordinary CPU-128 policy.
#include <atomic>
#include <thread>

struct FirstLayerChange {
    fs::path base, candidate;
    std::string mode,error;
    const llama_model *model=nullptr;
    json visits=json::array();

    static std::vector<uint8_t> file(const fs::path &path,size_t bytes) {
        need(fs::file_size(path)==bytes,"matched capture size");
        std::vector<uint8_t> v(bytes);std::ifstream f(path,std::ios::binary);f.read((char*)v.data(),bytes);need(bool(f),"capture read");return v;
    }
    std::vector<uint8_t> correct(const std::string &name,ggml_tensor *t,const std::vector<uint8_t>&before) {
        const auto xin=PrecisionCapture::bytes(t->src[1]),ein=PrecisionCapture::bytes(t->src[2]);
        need(xin==file(candidate/(name+"-input.bin"),xin.size()) && ein==file(candidate/(name+"-ids.bin"),ein.size()),"causal original inputs");
        auto xs=PrecisionCapture::values<float>(xin);auto ids=PrecisionCapture::values<int32_t>(ein);auto y=PrecisionCapture::values<float>(before);
        auto w=name=="ffn_moe_gate-0"?model->layers.at(0).ffn_gate_exps:model->layers.at(0).ffn_up_exps;
        const int nt=96,dim=5120,rows=2304,ns=6,nb=dim/256;
        need(w->type==GGML_TYPE_IQ2_S && ggml_backend_buffer_is_host(w->buffer) && w->ne[0]==dim && w->ne[1]==rows && xs.size()==size_t(nt)*dim && y.size()==size_t(ns)*nt*rows && ids.size()==size_t(ns)*nt,"residual dimensions");
        const auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
        const auto dot=ggml_get_type_traits_cpu(w->type)->vec_dot;
        std::vector<block_q8_K> encoded(nb),residual(nt*nb);std::vector<float> xr(dim);
        for(int token=0;token<nt;++token){
            auto x=xs.data()+size_t(token)*dim;
            quant(x,encoded.data(),dim);dequantize_row_q8_K(encoded.data(),xr.data(),dim);
            for(int i=0;i<dim;i++)xr[i]=x[i]-xr[i];
            quant(xr.data(),residual.data()+token*nb,dim);
        }
        for(int id:ids)need(id>=0 && id<w->ne[2],"expert bounds");
        std::atomic<bool> ok{true};std::vector<std::thread> pool;
        for(int worker=0;worker<24;++worker)pool.emplace_back([&,worker]{
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(worker,&mask);
            if(sched_setaffinity(0,sizeof(mask),&mask)!=0 || sched_getcpu()!=worker){ok=false;return;}
            const int begin=worker*rows/24,end=(worker+1)*rows/24;
            for(int token=0;token<nt;++token)for(int slot=0;slot<ns;++slot){
                const int e=ids[token*ns+slot];
                auto src=static_cast<const char*>(w->data)+size_t(e)*w->nb[2];
                auto dst=y.data()+size_t(token*ns+slot)*rows;
                for(int i=begin;i<end;i++){
                    float delta=0;dot(dim,&delta,0,src+size_t(i)*w->nb[1],0,residual.data()+token*nb,0,1);
                    dst[i]+=delta;
                }
            }
        });
        for(auto &th:pool)th.join();need(ok,"residual worker topology");
        std::vector<uint8_t> v(before.size());std::memcpy(v.data(),y.data(),v.size());return v;
    }
    bool observe(ggml_tensor *t,bool ask) {
        const std::string n=t->name;
        const bool lout=mode=="gpu-layer-output";
        const bool wanted=lout?(n=="l_out-0"):(n=="ffn_moe_gate-0" || n=="ffn_moe_up-0");
        if(ask)return wanted;
        need(wanted && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t),"causal target layout");
        const auto before=PrecisionCapture::bytes(t);
        need(before==file(candidate/(n+".bin"),before.size()),"causal tensor before change differs from captured CPU");
        if(!lout){
            for(int k:{1,2}){
                auto b=PrecisionCapture::bytes(t->src[k]);std::string f=n+(k==1?"-input.bin":"-ids.bin");
                need(b==file(candidate/f,b.size()) && b==file(base/f,b.size()),"identical independent operands");
            }
        }
        auto start=ns();double sampleerr=0;int checks=0;
        if(mode!="control"){
            auto data=mode=="residual"?correct(n,t,before):file(base/(n+".bin"),before.size());
            if(mode=="residual"){
                auto y=PrecisionCapture::values<float>(data);
                for(auto &r:read_json(candidate/"CAPTURE.json")["F64_samples"].get<json::array_t>()){
                    if(r[0].get<std::string>()!=n)continue;
                    int tok=r[1],slot=r[2],row=r[4];double err=double(y[(size_t(tok)*6+slot)*2304+row])-r[5].get<double>();
                    sampleerr+=err*err;++checks;
                }
                need(checks==1152 && std::isfinite(sampleerr),"residual sample audit");
            }
            ggml_backend_tensor_set(t,data.data(),0,data.size());
            need(PrecisionCapture::bytes(t)==data,"counterfactual tensor readback");
        }
        visits.push_back({{"node",n},{"bytes",before.size()},{"elapsed_ns",ns()-start},{"F64_samples",checks},{"F64_squared_error",sampleerr}});
        return true;
    }
    static bool callback(ggml_tensor*t,bool ask,void*raw){
        auto &c=*static_cast<FirstLayerChange*>(raw);
        if(!c.error.empty())return false;
        try{return c.observe(t,ask);}catch(const std::exception&e){c.error=e.what();return false;}
    }
};

static void run_changes(llama_context *ctx,const llama_model*model,llama_batch& batch,const fs::path& out,const json& cases,const json& spec,json& report,int nv){
    const auto root=fs::path(spec.at("endpoint_runs").get<std::string>());
    auto &params=const_cast<llama_cparams&>(ctx->get_cparams());
    need(std::string(getenv("GGML_OP_OFFLOAD_MIN_BATCH"))=="128" && std::string(getenv("GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"))=="128","causal base policy");
    auto set=[&](const int*ids,int count,int pos){batch.n_tokens=count;for(int i=0;i<count;i++){batch.token[i]=ids[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i+1==count;}};
    auto clear=[&](){llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto step=[&](){need(llama_decode(ctx,batch)==0,"causal llama_decode");const float*y=llama_get_logits_ith(ctx,-1);need(y,"causal logits");return y;};
    auto warm=cases.at("warmup_ja64").get<std::vector<int>>();set(warm.data(),64,0);step();clear();
    for(const auto &item:cases.at("cases")){
        std::string id=item["id"];if(id!="JA96" && id!="PY96")continue;
        auto prefix=item["prefix"].get<std::vector<int>>(),forced=item["forced"].get<std::vector<int>>();
        for(const std::string mode:{"control","gpu-gates","residual","gpu-layer-output"}){
            clear();FirstLayerChange c;c.base=root/"precision-64-64"/(id+"-early");c.candidate=root/"precision-128-128"/(id+"-early");c.mode=mode;c.model=model;
            params.cb_eval=c.callback;params.cb_eval_user_data=&c;ggml_backend_sched_set_eval_callback(ctx->get_sched(),c.callback,&c);
            set(prefix.data(),int(prefix.size()),0);auto t=ns();const float*y=step();auto end=ns();llama_synchronize(ctx);
            params.cb_eval=nullptr;params.cb_eval_user_data=nullptr;ggml_backend_sched_set_eval_callback(ctx->get_sched(),nullptr,nullptr);
            need(c.error.empty(),c.error.c_str());need(c.visits.size()==(mode=="gpu-layer-output"?1:2),"causal visits");
            std::ofstream file(out/(id+"-"+mode+".f32"),std::ios::binary);file.write((const char*)y,nv*4);
            for(int i=0;i<64;i++){set(&forced[i],1,int(prefix.size())+i);y=step();file.write((const char*)y,nv*4);}
            file.close();need(!file.fail(),"causal scores write");
            report["counterfactual"][id][mode]={{"rows",65},{"prefill_ns",end-t},{"visits",c.visits}};
            save(out/"RUN.json",report);
        }
    }
}
