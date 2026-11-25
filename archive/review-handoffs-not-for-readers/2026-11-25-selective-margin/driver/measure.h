#include "margin_shared.h"
#include <algorithm>
#include <array>

struct MarginCalls {
    using Init=void(*)(float);using Begin=void(*)(int,int);using Receipt=void(*)(margin_row*,int*);using Close=void(*)();
    Init init=nullptr;Begin begin=nullptr;Receipt receipt=nullptr;Close close=nullptr;
    explicit MarginCalls(const json &spec){
        init=(Init)dlsym(RTLD_DEFAULT,"xeon_margin_open");
        begin=(Begin)dlsym(RTLD_DEFAULT,"xeon_margin_begin");
        receipt=(Receipt)dlsym(RTLD_DEFAULT,"xeon_margin_receipt");
        close=(Close)dlsym(RTLD_DEFAULT,"xeon_margin_close");
        if(!spec.at("candidate").get<bool>()){need(!init&&!begin&&!receipt&&!close,"ordinary DSO");return;}
        Dl_info inf{};need(init&&begin&&receipt&&close&&dladdr((void*)init,&inf)&&fs::canonical(inf.dli_fname)==fs::path(spec["cuda_path"].get<std::string>()),"native diagnostic provider");
        init(spec["cut"].get<float>());
    }
    ~MarginCalls(){if(close)close();}
};

static int arm_code(const std::string &arm){
    if(arm=="D")return 1;if(arm=="S")return 2;if(arm=="R")return 3;if(arm=="A")return 4;return 0;
}

static void margin_trials(llama_context*ctx,llama_batch &batch,const fs::path&out,const json &fixtures,const json &spec,json &report,int nv,MarginCalls &hooks){
    const auto cases=fixtures.at("cases");const float cut=spec.at("cut");
    const bool capable=spec.at("candidate");
    auto vocab=llama_model_get_vocab(&ctx->get_model());
    auto set=[&](const int *ids,int n,int pos){
        need(n>0 && n<=1024,"batch bounds");batch.n_tokens=n;
        for(int i=0;i<n;i++){batch.token[i]=ids[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i+1==n;}
    };
    auto compute=[&](){need(llama_decode(ctx,batch)==0,"decode status");auto y=llama_get_logits_ith(ctx,-1);need(y,"output ready");return y;};
    auto clear=[&](){llama_synchronize(ctx);llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto warm=cases.at("JA96").at("prefix").get<std::vector<int>>();
    clear();if(capable)hooks.begin(1,int(warm.size()));set(warm.data(),int(warm.size()),0);compute();
    std::vector<margin_row> watch(40*128);int counts[6]{};
    if(capable)hooks.receipt(watch.data(),counts);
    clear();
    std::ofstream timeline(out/"measurements.jsonl");need(bool(timeline),"timeline file");
    for(const auto &entry:spec.at("trials")){
        std::string id=entry["case"],arm=entry["arm"],name=entry["name"];bool quality=entry["quality"],freegen=entry.value("free",false);
        const auto c=cases.at(id);const auto prefix=c["prefix"].get<std::vector<int>>(),forced=c["forced"].get<std::vector<int>>();
        const int n=int(prefix.size());const bool eligible=n>=64 && n<128;
        clear();if(capable)hooks.begin(arm_code(arm),n);set(prefix.data(),n,0);
        auto t=ns();auto y=compute();auto z=ns();
        json row={{"name",name},{"case",id},{"arm",arm},{"quality",quality},{"prefix",n},{"start_ns",t},{"ready_ns",z}};
        std::vector<float> first(y,y+nv);
        if(capable) {
            hooks.receipt(watch.data(),counts);
            int nf=0;
            if(eligible){
                for(int l=0;l<40;++l)for(int i=0;i<n;++i){
                    auto &r=watch[l*128+i];
                    need(r.sixth>=0&&r.sixth<384&&r.seventh>=0&&r.seventh<384&&r.sixth!=r.seventh&&std::isfinite(r.gap)&&r.gap>=0&&r.near_boundary==(r.gap<=cut),"valid live top-six boundary");
                    if(!l)nf+=r.near_boundary;
                }
                need(counts[0]==1&&counts[1]==1,"gate/up entered native grouped route");
                for(int p=0;p<2;p++){
                    const int expected=6*(arm=="A"?n:nf);
                    need(counts[2+p]==expected&&counts[4+p]==(arm=="D"?0:expected),"exact policy work count");
                }
                if(quality){
                    std::ofstream g(out/(name+".gaps"),std::ios::binary);
                    for(int l=0;l<40;l++)g.write((const char*)&watch[128*l],n*sizeof(margin_row));
                    g.close();need(!g.fail(),"write gap receipts");
                }
            }else for(int x:counts)need(x==0,"out-of-band fallthrough");
            row["layer0_near_tokens"]=nf;row["native_counts"]=std::vector<int>(counts,counts+6);
        }
        if(quality){
            std::ofstream f(out/(name+".f32"),std::ios::binary);need(bool(f),"logits file");int rows=0;
            auto save_row=[&](const float*x){need(std::all_of(x,x+nv,[](float v){return std::isfinite(v);}),"non-finite logits");f.write((const char*)x,nv*4);++rows;};
            save_row(first.data());std::vector<int> greedy;
            greedy.push_back(int(std::max_element(first.begin(),first.end())-first.begin()));
            for(size_t i=0;i<forced.size();i++){
                set(&forced[i],1,n+i);y=compute();save_row(y);greedy.push_back(int(std::max_element(y,y+nv)-y));
            }
            f.close();need(!f.fail(),"logits saved");row["rows"]=rows;row["greedy_on_fixed_history"]=greedy;
            if(c.contains("reference_ids")){
                auto ids=c["reference_ids"].get<std::vector<int>>();
                if(greedy==ids)row["free_text"]=c["reference_text"];
                else if(freegen){
                    clear();if(capable)hooks.begin(arm_code(arm),n);set(prefix.data(),n,0);y=compute();
                    if(capable)hooks.receipt(watch.data(),counts);
                    std::vector<int> seq;std::string text;
                    for(int i=0;i<160;i++){
                        const int next=int(std::max_element(y,y+nv)-y);seq.push_back(next);
                        if(llama_vocab_is_eog(vocab,next))break;
                        text+=common_token_to_piece(ctx,next,false);set(&seq.back(),1,n+i);y=compute();
                    }
                    row["free_text"]=text;row["new_free_ids"]=seq;row["new_free_eog"]=llama_vocab_is_eog(vocab,seq.back());
                }
            }
        } else {
            std::ofstream f(out/(name+".head"),std::ios::binary);f.write((const char*)first.data(),nv*4);f.close();need(!f.fail(),"timing prefix fingerprint");
        }
        timeline<<row.dump()<<'\n'<<std::flush;need(bool(timeline),"timing write");
        std::cout<<name<<" "<<(z-t)/1e6<<" ms\n"<<std::flush;
    }
    report["margin_trial_count"]=spec["trials"].size();save(out/"RUN.json",report);
}
