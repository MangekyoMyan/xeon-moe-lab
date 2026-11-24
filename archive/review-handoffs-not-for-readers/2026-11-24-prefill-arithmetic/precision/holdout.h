static void measure_chats(llama_context *ctx,llama_batch &batch,const fs::path &out,
                          const json &fixtures,const json &spec,json &report,int nv) {
    const bool first=spec.at("greedy_reference");
    json baseline=first?json::object():read_json(fs::path(spec.at("reference_file").get<std::string>()));
    json entries=json::object();
    auto vocab=llama_model_get_vocab(&ctx->get_model());
    for(const auto &item:fixtures.at("chats")) {
        const std::string id=item["id"];
        need(id.find_first_not_of("abcdefghijklmnopqrstuvwxyz-")==std::string::npos && !id.empty(),"chat identifier");
        const auto prefix=item["ids"].get<std::vector<llama_token>>();
        const std::vector<int> fixed=first?std::vector<int>{}:baseline.at(id).at("reference_ids").get<std::vector<int>>();
        const int max=first?96:int(fixed.size());need(max>0 && max<=96,"bounded common history");
        auto set=[&](const llama_token *p,int n,int pos) {
            batch.n_tokens=n;for(int i=0;i<n;++i){batch.token[i]=p[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i+1==n;}
        };
        llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);
        set(prefix.data(),int(prefix.size()),0);
        std::ofstream raw(out/("hold-"+id+".f32"),std::ios::binary);
        need(bool(raw),"holdout capture");
        std::vector<int> ref,top;std::string decoded;json times=json::array();
        for(int i=0;i<max;++i) {
            auto t=ns();need(llama_decode(ctx,batch)==0,"holdout decode");
            const float *row=llama_get_logits_ith(ctx,-1);auto z=ns();need(row,"holdout logits");
            need(std::all_of(row,row+nv,[](float x){return std::isfinite(x);}),"finite holdout logits");
            const int best=int(std::max_element(row,row+nv)-row);
            const int input=first?best:fixed.at(i);
            need(input>=0 && input<nv,"bounded teacher ID");
            top.push_back(best);ref.push_back(input);times.push_back({t,z});
            raw.write(reinterpret_cast<const char*>(row),nv*4);
            if(llama_vocab_is_eog(vocab,input)){need(first || i+1==max,"reference EOG only at end");break;}
            decoded+=common_token_to_piece(ctx,input,false);
            set(&ref.back(),1,int(prefix.size())+i);
        }
        raw.close();need(!raw.fail(),"holdout write");
        entries[id]={{"reference_ids",ref},{"observed_top_ids",top},{"reference_text",decoded},{"prefix_ids",prefix},
                     {"last_is_eog",llama_vocab_is_eog(vocab,ref.back())},{"clock_ns",times}};
        save(out/"GREEDY.json",entries);
    }
    report["holdout_count"]=entries.size();save(out/"RUN.json",report);
}
