// Diagnostic copies and reference arithmetic execute only on a separate prefix replay.
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <map>

struct PrecisionCapture {
    fs::path dest;
    std::string mode;
    const llama_model *model=nullptr;
    json log=json::array();
    json placements=json::object();
    json references=json::array();
    std::string error;

    static std::vector<uint8_t> bytes(const ggml_tensor *t) {
        need(t && (t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_I32) && t->nb[0]==4, "capture element layout");
        const size_t total=ggml_nelements(t)*4, span=ggml_nbytes(t);
        need(span>=total && span<=96*1024*1024, "capture extent");
        std::vector<uint8_t> raw(span);
        ggml_backend_tensor_get(t,raw.data(),0,span);
        if (ggml_is_contiguous(t)) return raw;
        std::vector<uint8_t> packed(total);
        size_t next=0;
        for (int64_t k=0;k<t->ne[3];++k) for(int64_t j=0;j<t->ne[2];++j) for(int64_t i=0;i<t->ne[1];++i) {
            const size_t at=k*t->nb[3]+j*t->nb[2]+i*t->nb[1],len=t->ne[0]*4;
            need(at+len<=span,"strided read");std::memcpy(packed.data()+next,raw.data()+at,len);next+=len;
        }
        need(next==total,"capture count");return packed;
    }
    static std::string buffer(const ggml_tensor *t) {
        auto b=t->buffer?t->buffer:(t->view_src?t->view_src->buffer:nullptr);
        return b?ggml_backend_buffer_name(b):"none";
    }
    static json describe(const ggml_tensor *t) {
        return {{"type",ggml_type_name(t->type)},{"shape",std::vector<int64_t>(t->ne,t->ne+4)},
                {"nb",std::vector<size_t>(t->nb,t->nb+4)},{"buffer",buffer(t)},{"tensor_name",t->name}};
    }
    std::vector<uint8_t> snapshot(const std::string &label,const ggml_tensor *t) {
        auto data=bytes(t);
        auto path=dest/(label+".bin");
        need(!fs::exists(path),"snapshot name collision");
        std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size());
        out.close();need(!out.fail(),"snapshot write");
        auto meta=describe(t);meta["file"]=path.filename().string();log.push_back(meta);
        return data;
    }
    template<typename T> static std::vector<T> values(const std::vector<uint8_t> &data) {
        need(data.size()%sizeof(T)==0,"typed bytes");std::vector<T> out(data.size()/sizeof(T));
        std::memcpy(out.data(),data.data(),data.size());return out;
    }
    void reference(const std::string &label,ggml_tensor *t,const std::vector<uint8_t>&ob) {
        const bool gate=label=="ffn_moe_gate-0";
        auto w=gate?model->layers.at(0).ffn_gate_exps:model->layers.at(0).ffn_up_exps;
        need(w && ggml_backend_buffer_is_host(w->buffer) && w->type==t->src[0]->type,"original native weights");
        auto x=values<float>(snapshot(label+"-input",t->src[1]));
        auto ids=values<int32_t>(snapshot(label+"-ids",t->src[2]));
        auto y=values<float>(ob);
        const int dim=w->ne[0],rows=w->ne[1],slots=t->ne[1],nt=t->ne[2],width=t->src[1]->ne[1];
        need(dim==5120 && rows==2304 && slots==6 && nt==96 && (width==1 || width==6)
            && x.size()==size_t(dim)*width*nt && ids.size()==size_t(slots)*nt
            && y.size()==size_t(rows)*slots*nt,"reference dimensions");
        const auto deq=ggml_get_type_traits(w->type)->to_float;
        const auto qt=ggml_get_type_traits_cpu(w->type)->vec_dot_type;
        need(deq && qt==GGML_TYPE_Q8_K,"first projection native Q8_K pairing");
        const auto quant=ggml_get_type_traits_cpu(qt)->from_float;need(quant,"native activation quantizer");
        std::vector<float> weights(dim),xq(dim);std::vector<block_q8_K> encoded(dim/256);
        for (int tok:{0,1,23,47,71,95}) for(int slot=0;slot<slots;++slot) {
            const float *v=x.data()+size_t(tok*width+(slot%width))*dim;
            quant(v,encoded.data(),dim);dequantize_row_q8_K(encoded.data(),xq.data(),dim);
            const int expert=ids[tok*slots+slot];need(expert>=0 && expert<w->ne[2],"real expert ID");
            for(int j=0;j<32;++j) {
                const int m=(j*(rows-1))/31;
                auto raw=static_cast<const char*>(w->data)+size_t(expert)*w->nb[2]+size_t(m)*w->nb[1];
                deq(raw,weights.data(),dim);
                double full=0.0,q8=0.0;
                for(int k=0;k<dim;++k){full+=double(weights[k])*v[k];q8+=double(weights[k])*xq[k];}
                const float result=y[(size_t(tok)*slots+slot)*rows+m];
                need(std::isfinite(full)&&std::isfinite(q8)&&std::isfinite(result),"finite reference");
                references.push_back({label,tok,slot,expert,m,full,q8,result});
            }
        }
    }
    bool observe(ggml_tensor *t,bool ask) {
        const std::string n=t->name;
        const bool mm=t->op==GGML_OP_MUL_MAT_ID && n.rfind("ffn_moe_",0)==0;
        if (ask && mm) placements[n]={buffer(t),ggml_type_name(t->src[0]->type),buffer(t->src[1]),buffer(t->src[2])};
        if (mode=="early") {
            static const std::array<const char*,8> names={"attn_out-0","ffn_norm-0","ffn_moe_logits-0","ffn_moe_gate-0","ffn_moe_up-0",
                                                        "ffn_moe_swiglu_limited-0","ffn_moe_down-0","l_out-0"};
            const bool want=std::any_of(names.begin(),names.end(),[&](const char*s){return n==s;});
            if (ask)return want;
            need(want,"early capture dispatch");auto v=snapshot(n,t);
            if(n=="ffn_moe_gate-0" || n=="ffn_moe_up-0")reference(n,t,v);
        } else {
            // Read the IDs while their last routed dot product still owns its input.
            const bool want=mm && n.rfind("ffn_moe_down-",0)==0;
            if(ask)return want;
            need(want,"route capture dispatch");snapshot(n+"-ids",t->src[2]);
        }
        return true;
    }
    static bool callback(ggml_tensor *t,bool ask,void *opaque) {
        auto &self=*static_cast<PrecisionCapture*>(opaque);
        if(!self.error.empty())return false;
        try{return self.observe(t,ask);}catch(const std::exception &e){self.error=e.what();return false;}
    }
};

static void capture_prefixes(llama_context *ctx,const llama_model *model,llama_batch &batch,
            const fs::path &out,const json &cases,const json &spec,json &report,int nv) {
    if(!spec.value("capture",false))return;
    auto &params=const_cast<llama_cparams&>(ctx->get_cparams());
    need(!params.cb_eval,"default evaluation callback");
    for(const auto &item:cases["cases"]) {
        const std::string id=item["id"];
        if(id!="JA96" && id!="PY96")continue;
        const auto prefix=item["prefix"].get<std::vector<llama_token>>();
        for(const std::string mode:{"early","routes"}) {
            llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);
            PrecisionCapture c;c.mode=mode;c.model=model;c.dest=out/(id+"-"+mode);
            fs::create_directory(c.dest);
            params.cb_eval=PrecisionCapture::callback;params.cb_eval_user_data=&c;
            ggml_backend_sched_set_eval_callback(ctx->get_sched(),params.cb_eval,params.cb_eval_user_data);
            batch.n_tokens=int(prefix.size());
            for(int i=0;i<batch.n_tokens;++i){batch.token[i]=prefix[i];batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i+1==batch.n_tokens;}
            const auto rc=llama_decode(ctx,batch);
            const float *row=rc==0?llama_get_logits_ith(ctx,-1):nullptr;
            llama_synchronize(ctx);
            params.cb_eval=nullptr;params.cb_eval_user_data=nullptr;
            ggml_backend_sched_set_eval_callback(ctx->get_sched(),nullptr,nullptr);
            need(c.error.empty(),c.error.c_str());need(row,"capture prefix output");
            std::vector<float> reference(nv);std::ifstream f(out/(id+".f32"),std::ios::binary);f.read((char*)reference.data(),nv*4);need(bool(f),"ordinary prefix reference");
            const bool exact=std::memcmp(row,reference.data(),nv*4)==0;
            std::ofstream last(c.dest/"final.f32",std::ios::binary);last.write((const char*)row,nv*4);last.close();need(!last.fail(),"final capture");
            save(c.dest/"CAPTURE.json",{{"mode",mode},{"case",id},{"prefix_logits_byte_equal_to_no_callback",exact},
                  {"observations",c.log},{"placements",c.placements},{"F64_samples",c.references}});
            report["capture_comparison"][id][mode]=exact;
            save(out/"RUN.json",report);
        }
    }
}
