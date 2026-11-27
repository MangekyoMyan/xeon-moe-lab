extern "C" void pp512_live_begin(llama_context*,const llama_model*);
extern "C" void pp512_live_mode(int,int,bool);
extern "C" const char *pp512_live_receipt();
extern "C" void pp512_live_end();
struct ExtensionOwner{ExtensionOwner(llama_context*c,const llama_model*m){pp512_live_begin(c,m);}~ExtensionOwner(){pp512_live_end();}};
static void put_f32(const fs::path&p,const std::vector<float>&v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*4);f.close();need(!f.fail(),"F32 output");}
static void run_extended(llama_context*ctx,const llama_model*model,llama_batch&b,const fs::path&out,const json&cases,json&report,int nv){
    ExtensionOwner o(ctx,model);const auto *vocab=llama_model_get_vocab(model);
    auto clear=[&]{llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto call=[&](const std::vector<llama_token>&ids,int pos,int m,bool audit){
        need(!ids.empty()&&ids.size()<=1024,"real ubatch");pp512_live_mode(m,3,audit);b.n_tokens=ids.size();
        for(int i=0;i<b.n_tokens;++i){b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=i+1==b.n_tokens;}
        auto t0=ns();need(llama_decode(ctx,b)==0,"decode");auto p=llama_get_logits_ith(ctx,-1);llama_synchronize(ctx);auto t1=ns();
        need(p,"output");std::vector<float>v(p,p+nv);for(auto x:v)need(std::isfinite(x),"finite");
        auto rec=json::parse(pp512_live_receipt());auto layers=rec["layers"];
        if(m==0||ids.size()!=1024)need(layers.empty(),"normal/fallback scope");
        else need(layers.size()==2,"both eligible 1024 layers");
        json j={{"at",pos},{"n",ids.size()},{"start",t0},{"end",t1},{"layers",layers},{"free_B",rec["free_gpu_B"]}};
        return std::make_pair(std::move(v),std::move(j));
    };
    clear();call(cases["warm"],0,0,false);
    const std::map<std::string,int>mode={{"O",0},{"GP",1},{"S",2},{"P",3}};
    for(const auto &c:cases["cases"]){
        const std::string id=c["id"];std::map<std::string,std::vector<float>> gold;
        for(const std::string arm:{"O","GP","S","P"}){
            clear();auto [v,j]=call(c["tokens"],0,mode.at(arm),true);
            std::vector<float> rows;for(int i=0;i<=16;++i){rows.insert(rows.end(),v.begin(),v.end());if(i<16)v=call({c["forced"][i].get<llama_token>()},1024+i,mode.at(arm),false).first;}
            put_f32(out/(id+"-"+arm+".f32"),rows);gold[arm]=rows;
            j["case"]=id;j["arm"]=arm;j["identical_to_O"]=std::memcmp(rows.data(),gold["O"].data(),rows.size()*4)==0;
            if(arm=="P")need(!std::memcmp(rows.data(),gold["S"].data(),rows.size()*4),"joined full distributions");
            report["quality"].push_back(j);save(out/"RUN.json",report);
        }
    }
    for(int pair=0;pair<2;++pair)for(int ci=0;ci<2;++ci)for(int i=0;i<2;++i){
        const auto &c=cases["cases"][pair?1-ci:ci];const std::string id=c["id"],arm=(pair^i)?"P":"O";
        clear();auto [v,j]=call(c["tokens"],0,mode.at(arm),false);
        std::ifstream f(out/(id+"-"+arm+".f32"),std::ios::binary);std::vector<float> gold(nv);f.read((char*)gold.data(),nv*4);
        need(bool(f)&&!std::memcmp(v.data(),gold.data(),nv*4),"timed logits");
        j["case"]=id;j["arm"]=arm;j["pair"]=pair;j["order"]=i;report["timings"].push_back(j);save(out/"RUN.json",report);
    }
    const auto &chat=cases["conversation"];std::vector<llama_token> prior=chat["common_prefix_tokens"],suffix=chat["suffix_tokens"];
    need(prior.size()>=16000&&suffix.size()>0&&suffix.size()<160,"shared history");
    for(int pair=0;pair<2;++pair)for(int i=0;i<2;++i){
        const std::string arm=(pair^i)?"P":"O";const int m=mode.at(arm);
        clear();json j={{"arm",arm},{"pair",pair},{"order",i},{"prefix_chunks",json::array()},{"suffix_ids",suffix}};
        std::vector<float> v;
        for(size_t pos=0;pos<prior.size();pos+=1024){
            auto end=std::min(pos+1024,prior.size());std::vector<llama_token> a(prior.begin()+pos,prior.begin()+end);
            auto data=call(a,pos,m,false);v=std::move(data.first);j["prefix_chunks"].push_back(data.second);
        }
        const std::string path="long-"+std::to_string(pair)+"-"+arm;
        put_f32(out/(path+"-prefix.f32"),v);
        auto follow=call(suffix,prior.size(),m,false);v=std::move(follow.first);j["suffix"]=follow.second;put_f32(out/(path+"-suffix.f32"),v);
        std::vector<llama_token> generated;
        for(int k=0;k<24;++k){auto t=llama_token(std::max_element(v.begin(),v.end())-v.begin());generated.push_back(t);
            if(llama_vocab_is_eog(vocab,t)||k==23)break;
            v=call({t},prior.size()+suffix.size()+k,m,false).first;}
        j["greedy"]=generated;j["text"]=common_detokenize(ctx,generated,false);report["conversation"].push_back(j);save(out/"RUN.json",report);
        std::cout<<"long-history pair "<<pair<<" arm "<<arm<<" complete\n"<<std::flush;
    }
    pp512_live_mode(0,0,false);
}
