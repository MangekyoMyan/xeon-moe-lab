extern "C" void pp512_ledger_label(const char*);
struct ModelBoundary{
    std::string phase;std::map<int,std::pair<int64_t,int64_t>> times;
    int patch_layer=-1;std::vector<float> original,candidate;int injected=0;bool valid=true;
    bool node(ggml_tensor*t,bool ask){
        for(int l:{4,24}){
            const auto suffix="-"+std::to_string(l);const std::string name=t->name;
            bool start=name=="ffn_moe_weights_scaled"+suffix,finish=name=="ffn_moe_out"+suffix;
            if(!start&&!finish)continue;
            if(phase=="boundary"){
                if(!ask){if(start)times[l].first=ns();else times[l].second=ns();}
                return true;
            }
            if(l==patch_layer&&finish){
                if(!ask){
                    const size_t bytes=candidate.size()*4;std::vector<float>check(candidate.size());
                    if(ggml_nbytes(t)!=bytes || !ggml_is_contiguous(t)){valid=false;return false;}
                    ggml_backend_tensor_get(t,check.data(),0,bytes);
                    if(std::memcmp(check.data(),original.data(),bytes)){valid=false;return false;}
                    ggml_backend_tensor_set(t,candidate.data(),0,bytes);ggml_backend_tensor_get(t,check.data(),0,bytes);
                    valid &= !std::memcmp(check.data(),candidate.data(),bytes);++injected;
                }
                return true;
            }
        }
        return false;
    }
    static bool callback(ggml_tensor*t,bool ask,void*opaque){return static_cast<ModelBoundary*>(opaque)->node(t,ask);}
};
static std::vector<float> load_float_file(const fs::path&p,size_t n){
    need(fs::file_size(p)==n*4,"F32 extent");std::vector<float>x(n);std::ifstream f(p,std::ios::binary);
    f.read((char*)x.data(),n*4);need(bool(f),"read F32");return x;
}
static void run_audit(llama_context*ctx,const llama_model*model,llama_batch&b,const fs::path&out,const json&cases,json&report,int nv){
    auto clear=[&]{llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto call=[&](const std::vector<llama_token>&tokens,int pos){
        b.n_tokens=tokens.size();
        for(int i=0;i<b.n_tokens;++i){b.token[i]=tokens[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=i+1==b.n_tokens;}
        need(llama_decode(ctx,b)==0,"decode");auto row=llama_get_logits_ith(ctx,-1);llama_synchronize(ctx);need(row,"logits");
        return std::vector<float>(row,row+nv);
    };
    auto set=[&](ModelBoundary *m){
        auto &cp=const_cast<llama_cparams&>(ctx->get_cparams());cp.cb_eval=m?ModelBoundary::callback:nullptr;cp.cb_eval_user_data=m;
        ggml_backend_sched_set_eval_callback(ctx->get_sched(),cp.cb_eval,m);
    };
    const auto vocab=llama_model_get_vocab(model);pp512_ledger_label("warm");clear();call(cases.at("warm"),0);
    for(const auto &c:cases["cases"]){
        std::string id=c["id"];std::vector<llama_token> tokens=c["tokens"],forced=c["forced"];
        if(forced.empty()){
            const std::string prompt=c["rendered"];int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
            need(n>=528,"complete holdout tokenization");std::vector<llama_token> all(n);
            need(llama_tokenize(vocab,prompt.data(),prompt.size(),all.data(),all.size(),true,true)==n &&
                 std::equal(tokens.begin(),tokens.end(),all.begin()),"published 512 tokens match exact tokenizer");
            forced.assign(all.begin()+512,all.begin()+528);
        }
        need(tokens.size()==512 && forced.size()==16,"fixed sizes");report["forced_inputs"][id]=forced;
        const fs::path cap=c["capture"].get<std::string>();auto ref=load_float_file(cap/"native-logits.f32",nv);
        for(std::string phase:{"plain","boundary","P4","P24"}){
            clear();ModelBoundary cb;cb.phase=phase;std::string label=id+"-"+phase;pp512_ledger_label(label.c_str());
            if(phase[0]=='P'){
                cb.patch_layer=std::stoi(phase.substr(1));
                cb.original=load_float_file(cap/("ffn_moe_out-"+std::to_string(cb.patch_layer)+".bin"),512*5120);
                cb.candidate=load_float_file(c["candidates"][std::to_string(cb.patch_layer)].get<std::string>(),512*5120);
            }
            set(phase=="plain"?nullptr:&cb);
            const auto lo=ns();auto row=call(tokens,0);const auto hi=ns();set(nullptr);
            need(cb.valid && cb.injected==(phase[0]=='P'?1:0),"actual single-layer injection verified");
            bool original=std::memcmp(row.data(),ref.data(),nv*4)==0;
            if(phase=="plain"||phase=="boundary")need(original,"whole-model control logits");
            json j={{"phase",phase},{"case",id},{"start",lo},{"end",hi},{"prefix_matches_native",original},
                    {"layer_times",json::object()},{"injected",cb.injected},{"forced_argmax",json::array()},{"greedy_ids",json::array()}};
            for(auto [l,clock]:cb.times){need(clock.second>clock.first && clock.first>lo && clock.second<hi,"native layer boundaries");j["layer_times"][std::to_string(l)]={clock.first,clock.second};}
            if(phase=="boundary")need(cb.times.size()==2,"boundary count");
            pp512_ledger_label("T1");
            if(phase!="boundary"){
                auto top=[](const std::vector<float>&v){for(auto x:v)need(std::isfinite(x),"finite");return int(std::max_element(v.begin(),v.end())-v.begin());};
                std::ofstream f(out/(label+".f32"),std::ios::binary);
                for(int n=0;n<=16;++n){j["forced_argmax"].push_back(top(row));f.write((const char*)row.data(),nv*4);if(n<16)row=call({forced[n]},512+n);}
                f.close();need(!f.fail(),"write distributions");
                clear();cb.injected=0;set(phase=="plain"?nullptr:&cb);row=call(tokens,0);set(nullptr);
                need(cb.valid && cb.injected==(phase[0]=='P'?1:0),"free greedy verified injection");
                for(int n=0;n<16;++n){int t=top(row);j["greedy_ids"].push_back(t);if(llama_vocab_is_eog(vocab,t)||n==15)break;row=call({t},512+n);}
            }
            report["audit"].push_back(j);save(out/"RUN.json",report);
        }
    }
}
