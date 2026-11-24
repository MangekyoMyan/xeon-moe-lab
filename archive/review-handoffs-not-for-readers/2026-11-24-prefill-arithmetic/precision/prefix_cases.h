static void measure_prefixes(llama_context *ctx,llama_batch &batch,const fs::path &out,const json &fixtures,
                             const json &spec,json &report,int nv) {
    int index=0;
    for(const char *key:{"GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"}) {
        const char *actual=std::getenv(key);
        need(actual && std::to_string(spec.at("thresholds").at(index++).get<int>())==actual,"effective factorial settings");
    }
    auto compute=[&](const llama_token *tokens,int count,int pos) {
        need(count>0 && count<=1024,"prefix batch bounds");
        batch.n_tokens=count;
        for (int i=0;i<count;++i) {
            need(tokens[i]>=0 && tokens[i]<nv,"valid token ID");
            batch.token[i]=tokens[i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;
            batch.seq_id[i][0]=0;batch.logits[i]=i==count-1;
        }
        need(llama_decode(ctx,batch)==0,"prefix decode");
        const float *output=llama_get_logits_ith(ctx,-1);
        need(output!=nullptr,"prefix output ready");return output;
    };
    auto clear=[&](){llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);};
    auto finite=[&](const float *row){for(int i=0;i<nv;++i)need(std::isfinite(row[i]),"finite prefix logits");};
    const auto warm=fixtures.at("warmup_ja64").get<std::vector<llama_token>>();
    need(warm.size()==64,"fixed 64-token warmup");finite(compute(warm.data(),64,0));clear();
    for(const auto &item:fixtures["cases"]) {
        const std::string label=item["id"];
        need(label.find_first_not_of("JAPY0123456789")==std::string::npos && !label.empty(),"fixture filename");
        auto prefix=item["prefix"].get<std::vector<llama_token>>();
        auto forced=item["forced"].get<std::vector<llama_token>>();
        need(!prefix.empty() && (forced.empty() || forced.size()==64),"quality sample sizes");
        clear();
        std::vector<float> output(size_t(1+forced.size())*nv);
        auto a=ns();auto row=compute(prefix.data(),int(prefix.size()),0);auto z=ns();
        finite(row);std::memcpy(output.data(),row,nv*sizeof(float));
        json times=json::array();
        for(size_t i=0;i<forced.size();++i) {
            auto t0=ns();row=compute(&forced[i],1,int(prefix.size()+i));auto t1=ns();
            finite(row);std::memcpy(output.data()+size_t(nv)*(i+1),row,nv*sizeof(float));
            times.push_back({t0,t1});
        }
        llama_synchronize(ctx);
        std::ofstream f(out/(label+".f32"),std::ios::binary);need(bool(f),"prefix F32 output");
        f.write(reinterpret_cast<const char*>(output.data()),output.size()*4);f.close();need(!f.fail(),"prefix F32 write");
        report["blocks"].push_back({{"id",label},{"prefix_ids",prefix},{"forced_inputs",forced},{"rows",1+forced.size()},
             {"prefill_begin_ns",a},{"prefill_ready_ns",z},{"T1_times",times},{"all_finite",true}});
        save(out/"RUN.json",report);
    }
}
