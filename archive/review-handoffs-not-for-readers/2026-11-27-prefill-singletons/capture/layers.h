#include <map>

struct LayerCapture {
    fs::path folder;
    json tensors=json::object();
    std::string error;
    std::map<std::string, int> hits;

    void store(const std::string &name,const ggml_tensor *t) {
        need(++hits[name]==1,"unexpected extra ubatch or duplicate capture");
        need(t && t->nb[0]==4 && (t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_I32),"capture dtype");
        const size_t size=ggml_nbytes(t),row=size_t(t->ne[0])*4;
        need(size>0 && size<=96u*1024*1024,"bounded snapshot");
        std::vector<char> src(size);
        ggml_backend_tensor_get(t,src.data(),0,size);
        std::ofstream f(folder/(name+".bin"),std::ios::binary);
        for(int64_t k=0;k<t->ne[3];++k) for(int64_t j=0;j<t->ne[2];++j) for(int64_t i=0;i<t->ne[1];++i) {
            const size_t off=k*t->nb[3]+j*t->nb[2]+i*t->nb[1];
            need(off+row<=size,"snapshot stride");f.write(src.data()+off,row);
        }
        f.close();need(!f.fail(),"capture write");
        auto buf=t->buffer?t->buffer:t->view_src?t->view_src->buffer:nullptr;
        tensors[name]={{"name",t->name},{"op",ggml_op_name(t->op)},{"type",ggml_type_name(t->type)},
                       {"ne",std::vector<int64_t>(t->ne,t->ne+4)},{"nb",std::vector<size_t>(t->nb,t->nb+4)},
                       {"buffer",buf?ggml_backend_buffer_name(buf):"none"},{"file",name+".bin"}};
    }
    bool see(ggml_tensor *t,bool ask) {
        const std::string n=t->name;
        for(int l:{4,24}){
            const auto suf="-"+std::to_string(l);
            if(n=="ffn_moe_gate"+suf && t->op==GGML_OP_MUL_MAT_ID){
                if(!ask){store("x"+suf,t->src[1]);store("ids"+suf,t->src[2]);}
                return true;
            }
            if(n=="ffn_moe_weights_scaled"+suf || n=="ffn_moe_down"+suf || n=="ffn_moe_out"+suf){
                if(!ask)store(n,t);
                return true;
            }
        }
        return false;
    }
    static bool callback(ggml_tensor *t,bool ask,void *opaque){
        auto &self=*static_cast<LayerCapture*>(opaque);
        if(!self.error.empty())return false;
        try{return self.see(t,ask);}catch(const std::exception &e){self.error=e.what();return false;}
    }
};

static void run_capture(llama_context *ctx,const llama_model *model,llama_batch &b,const fs::path &out,
                        const json &fixtures,json &report,int nv){
    auto &params=const_cast<llama_cparams&>(ctx->get_cparams());
    auto run=[&](const std::vector<llama_token>&ids){
        llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);
        b.n_tokens=ids.size();
        for(int i=0;i<b.n_tokens;++i){b.token[i]=ids[i];b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i+1==b.n_tokens);}
        auto a=ns();need(llama_decode(ctx,b)==0,"decode");auto *p=llama_get_logits_ith(ctx,-1);llama_synchronize(ctx);
        auto z=ns();need(p,"logits");std::vector<float> row(p,p+nv);
        for(float f:row)need(std::isfinite(f),"finite logits");
        return std::make_pair(row,z-a);
    };
    auto warm=fixtures.at("warm").get<std::vector<llama_token>>();run(warm);
    auto ids=fixtures.at("tokens").get<std::vector<llama_token>>();need(ids.size()==512,"512 fixture");
    auto ref=run(ids);
    auto write_logits=[&](const char *name,const std::vector<float>&data){
        std::ofstream f(out/name,std::ios::binary);f.write((const char*)data.data(),data.size()*4);f.close();need(!f.fail(),"write logits");};
    write_logits("native-logits.f32",ref.first);
    LayerCapture cap;cap.folder=out;params.cb_eval=LayerCapture::callback;params.cb_eval_user_data=&cap;
    ggml_backend_sched_set_eval_callback(ctx->get_sched(),params.cb_eval,params.cb_eval_user_data);
    auto found=run(ids);
    params.cb_eval=nullptr;params.cb_eval_user_data=nullptr;ggml_backend_sched_set_eval_callback(ctx->get_sched(),nullptr,nullptr);
    write_logits("capture-logits.f32",found.first);
    const bool same=std::memcmp(ref.first.data(),found.first.data(),nv*4)==0;
    report["capture"]={{"plain_prefix_ns",ref.second},{"observed_prefix_ns",found.second},{"vocab",nv},
                       {"native_logits_equal",same},{"tensors",cap.tensors},{"hits",cap.hits},{"error",cap.error}};
    for(int l:{4,24}){
        auto &layer=model->layers.at(l);
        for(auto *w:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps}){
            report["weight_meta"][std::to_string(l)].push_back({{"name",w->name},{"type",ggml_type_name(w->type)},
                {"ne",std::vector<int64_t>(w->ne,w->ne+4)},{"nb",std::vector<size_t>(w->nb,w->nb+4)},
                {"buffer",ggml_backend_buffer_name(w->buffer)}});
        }
    }
    save(out/"RUN.json",report);
    need(cap.error.empty(),cap.error.c_str());need(cap.hits.size()==10 && same,"capture completeness or prefix logits");
    for(int l:{4,24}){
        const std::string s="-"+std::to_string(l);
        need(cap.tensors["x"+s]["ne"]==json::array({5120,1,512,1}) &&
             cap.tensors["ids"+s]["ne"]==json::array({6,512,1,1}) &&
             cap.tensors["ffn_moe_down"+s]["ne"]==json::array({5120,6,512,1}) &&
             cap.tensors["ffn_moe_out"+s]["ne"]==json::array({5120,512,1,1}),"actual single 512 ubatch");
    }
}
