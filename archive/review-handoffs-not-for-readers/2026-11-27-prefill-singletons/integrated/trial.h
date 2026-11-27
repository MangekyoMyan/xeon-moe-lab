extern "C" void pp512_live_begin(llama_context*,const llama_model*);
extern "C" void pp512_live_mode(int,int,bool);
extern "C" const char *pp512_live_receipt();
extern "C" void pp512_live_end();
struct LiveOwner{LiveOwner(llama_context*c,const llama_model*m){pp512_live_begin(c,m);}~LiveOwner(){pp512_live_end();}};
static std::vector<float> from_file(const fs::path&p,size_t n){
    need(fs::file_size(p)==n*4,"reference size");std::vector<float> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);need(bool(f),"reference read");return v;
}
static void actual_trial(llama_context*ctx,const llama_model*model,llama_batch&b,const fs::path&out,const json &cases,json&report,int nv){
    LiveOwner scope(ctx,model);const fs::path audit=cases["audit_folder"].get<std::string>();
    const json evidence=read_json(audit/"RUN.json");need(evidence["passed"],"prior true-model audit");
    auto clear=[&]{llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto evaluate=[&](const std::vector<llama_token>&tokens,int pos){
        b.n_tokens=tokens.size();for(int i=0;i<b.n_tokens;++i){b.token[i]=tokens[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=(i+1==b.n_tokens);}
        auto a=ns();need(llama_decode(ctx,b)==0,"decode");auto row=llama_get_logits_ith(ctx,-1);llama_synchronize(ctx);auto z=ns();
        need(row,"whole logits");std::vector<float> v(row,row+nv);for(auto x:v)need(std::isfinite(x),"finite");return std::make_tuple(std::move(v),a,z);
    };
    const std::map<std::string,std::pair<int,int>> map={{"O",{0,0}},{"GP",{1,3}},{"P4",{3,1}},{"P24",{3,2}},{"S",{2,3}},{"P",{3,3}}};
    auto mode=[&](const std::string &arm,bool qual){auto [m,k]=map.at(arm);pp512_live_mode(m,k,qual);};
    auto obtain=[&](const json&c,const std::string&arm,bool qual){
        mode(arm,qual);clear();const auto tokens=c.at("tokens").get<std::vector<llama_token>>();
        auto [v,a,z]=evaluate(tokens,0);auto receipt=json::parse(pp512_live_receipt());
        const int mask=map.at(arm).second,n=bool(mask&1)+bool(mask&2);
        if(receipt["layers"].size()!=size_t(n)){save(out/"MISSING_LAYER.json",receipt);throw std::runtime_error("integrated dispatch count");}
        if(n)need(receipt["matched4"]&&receipt["matched24"],"verified graph contract");
        for(int i=0;i<n;++i){int l=receipt["layers"][i]["layer"];need((i==0?(mask&1?4:24):24)==l,"layer order");}
        json j={{"case",c["id"]},{"arm",arm},{"start_ns",a},{"end_ns",z},{"receipt",receipt}};
        return std::make_pair(std::move(v),j);
    };
    mode("O",false);clear();evaluate(cases["warm"],0);
    for(const auto &c:cases["cases"]){
        const std::string id=c["id"];std::vector<llama_token> forced=evidence.at("forced_inputs").at(id);
        need(forced.size()==16,"fixed targets");std::map<std::string,std::vector<float>> all;
        for(const std::string arm:{"O","GP","P4","P24","S","P"}){
            auto [v,j]=obtain(c,arm,true);std::vector<float> block;block.reserve(size_t(nv)*17);
            for(int k=0;k<=16;++k){block.insert(block.end(),v.begin(),v.end());
                if(k<16){mode(arm,false);v=std::get<0>(evaluate({forced[k]},512+k));need(json::parse(pp512_live_receipt())["layers"].empty(),"T1 fallback");}}
            std::string ref=arm=="O"||arm=="GP"?"plain":arm;bool ok=true;
            if(arm=="O"||arm=="GP"||arm=="P4"||arm=="P24"){
                const auto exact=from_file(audit/(id+"-"+ref+".f32"),block.size());
                ok=std::memcmp(exact.data(),block.data(),block.size()*4)==0;
                j["compared_to"]=ref;
            }else if(arm=="P"){ok=std::memcmp(all.at("S").data(),block.data(),block.size()*4)==0;j["compared_to"]="integrated S";}
            if(arm=="S")all["S"]=block;
            j["same_as_reference_when_compared"]=ok;
            std::ofstream f(out/(id+"-"+arm+".f32"),std::ios::binary);f.write((const char*)block.data(),block.size()*4);f.close();need(!f.fail(),"store logits");
            report["quality"].push_back(j);save(out/"RUN.json",report);
            need(ok,"integrated reference mismatch (see quality receipt)");
        }
    }
    // One fresh 512-token prefill per call; normal memory state cleared before the clock.
    // Alternate native/candidate order, and reverse the order of the two fixed prompts.
    const int first[]={0,1,1,0};
    for(int pair=0;pair<4;++pair)for(int index=0;index<2;++index){
        const auto &c=cases["cases"][(pair%2)?1-index:index];
        for(int i=0;i<2;++i){
            std::string arm=(first[pair]^i)?"P":"O";auto [row,j]=obtain(c,arm,false);
            auto refpath=out/(c["id"].get<std::string>()+"-"+arm+".f32");std::vector<float> expected(nv);
            std::ifstream f(refpath,std::ios::binary);f.read((char*)expected.data(),nv*4);need(bool(f)&&!std::memcmp(expected.data(),row.data(),nv*4),"timed prefix full vocabulary exact");
            j["pair"]=pair;j["order"]=i;report["timings"].push_back(j);save(out/"RUN.json",report);
        }
    }
    for(const auto &c:cases["cases"])for(std::string arm:{"O","P"}){
        auto [row,j]=obtain(c,arm,false);std::vector<llama_token> seq;
        for(int k=0;k<24;++k){
            auto t=llama_token(std::max_element(row.begin(),row.end())-row.begin());seq.push_back(t);
            if(llama_vocab_is_eog(llama_model_get_vocab(model),t)||k==23)break;
            mode(arm,false);row=std::get<0>(evaluate({t},512+k));
        }
        j["greedy"]=seq;j["text"]=common_detokenize(ctx,seq,false);report["free_greedy"].push_back(j);
    }
    save(out/"RUN.json",report);pp512_live_mode(0,0,false);
}
