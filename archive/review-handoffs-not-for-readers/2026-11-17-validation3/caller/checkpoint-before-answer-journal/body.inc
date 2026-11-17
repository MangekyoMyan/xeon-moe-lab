        auto input=read_json(cases_path);need(input["review_revision"]==2&&input["case_count"]==3,"frozen revision2");
        std::vector<std::vector<llama_token>>prefixes,continuations;std::map<std::string,int>index;
        const std::array<std::string,3>names={"text_plan_constraints_ja","text_plan_constraints_en","code_bounded_queue_python"};
        const std::array<int,3>lengths={236,187,244};
        for(int i=0;i<3;++i){const auto &c=input["cases"][i];const auto &e=c["execution_input"];
            need(c["case_id"]==names[i]&&c["split"]=="validation"&&e["prefix_count"]==lengths[i],"fixed cases");
            std::vector<llama_token>pr=e["prefix_ids"],co=e["teacher_forced_next_targets65"],in=e["teacher_forced_input64"],joined=e["joined_expected_ids"];
            need(pr.size()==size_t(lengths[i])&&co.size()==65&&in.size()==64&&joined.size()>=pr.size()+co.size(),"input geometry");
            need(std::equal(pr.begin(),pr.end(),joined.begin())&&std::equal(co.begin(),co.end(),joined.begin()+pr.size())&&std::equal(in.begin(),in.end(),co.begin()),"prefix/continuation/target boundary");
            for(auto t:joined)need(t>=0&&t<nv,"token ID range");
            index[names[i]]=i;prefixes.push_back(pr);continuations.push_back(co);
            report["cases"].push_back({{"case_id",names[i]},{"family_id",c["family_id"]},{"split",c["split"]},{"prefix_ids",pr},{"prefix_count",pr.size()},
                {"teacher_forced_input64",in},{"next_targets65",co},{"prefix_i32_sha256",c["frozen_token_manifest_case_exact"]["prefix_ids_i32_sha256"]},{"joined_i32_sha256",e["joined_expected_ids_i32_sha256"]}});
        }
        std::vector<int>order;std::set<int>seen;for(const auto &name:spec["cases"]){need(index.count(name.get<std::string>())==1,"selected case");int i=index.at(name.get<std::string>());need(seen.insert(i).second,"duplicate case");order.push_back(i);}need(!order.empty(),"cases required");
        report["input_policy"]="Frozen prefix/continuation IDs only; no template application and no tokenization. Free answer receives prefix only.";
        resources.batch=llama_batch_init(1024,0,1);resources.has_batch=true;auto &batch=resources.batch;api=std::make_unique<API>();
        auto closed=[&](){need(snapshot(*api)["stats"]["enabled"]==false,"collector closed; internal-origin separation not implemented");};
        auto set_batch=[&](const std::vector<llama_token>&tok,int position){need(!tok.empty()&&tok.size()<=1024,"batch bounds");batch.n_tokens=tok.size();for(int i=0;i<batch.n_tokens;++i){batch.token[i]=tok[i];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==batch.n_tokens-1);}};
        auto set_one=[&](llama_token token,int position){batch.n_tokens=1;batch.token[0]=token;batch.pos[0]=position;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=true;};
        auto finite=[&](const float *p){if(!p)return false;for(int i=0;i<nv;++i)if(!std::isfinite(p[i]))return false;return true;};
        auto stats=[&](){xeon_hot_trial_stats x{};json j;if(mixed){need(xeon_hot_trial_snapshot(outer_cpu,&x)==1&&x.ready,"bank state");j={{"mode",std::string(1,x.mode)},{"custom_nodes_built",x.builds},{"native_bytes",x.payload},{"bank_allocation_bytes",x.allocation},{"calls",json::array()},{"executed_H",json::array()}};for(int l=0;l<40;++l){j["calls"].push_back(x.calls[l]);j["executed_H"].push_back(std::vector<uint64_t>(x.hist[l],x.hist[l]+7));}}return j;};
        auto validate_calls=[&](const json&a,const json&b,int calls){if(!mixed)return;uint64_t total=0;for(int l=0;l<40;++l){uint64_t n=b["calls"][l].get<uint64_t>()-a["calls"][l].get<uint64_t>();need(n==0||n==uint64_t(calls),"per-layer bounded calls");total+=n;}need(total==uint64_t(23)*calls,"23 selected layers; prefill/empty/GPU bypass");};
        auto select=[&](char mode){llama_synchronize(ctx);closed();if(mixed)need((mode=='S'||mode=='P')&&xeon_hot_trial_set_mode(outer_cpu,mode)==1,"bank mode switch");else need(mode=='F',"baseline mode");};
        auto begin_prefix=[&](int ci){llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);set_batch(prefixes[ci],0);};
        auto compute=[&](){int rc=llama_decode(ctx,batch);need(rc==0,"decode failed");const float *p=llama_get_logits_ith(ctx,-1);need(p,"normal output ready");return p;};
        auto write_f32=[&](const std::string &name,const std::vector<float>&v){need(!std::filesystem::exists(out/name),"no output overwrite");std::ofstream f(out/name,std::ios::binary);need(bool(f),"F32 open");f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));f.close();need(!f.fail(),"F32 write");};
        auto load_bytes=[&](const std::string &name){std::ifstream f(out/name,std::ios::binary);need(bool(f),"output read");return std::vector<char>((std::istreambuf_iterator<char>(f)),{});};
        // The same preparation in every process: JA prefix64, then fullJA plus4 reference inputs.
        closed();select(mixed?'P':'F');std::vector<llama_token>warm(prefixes[0].begin(),prefixes[0].begin()+64);set_batch(warm,0);need(finite(compute()),"text64 finite");memory("M2_after_text64_warmup");
        json prep={{"case_id",names[0]},{"prefix_ids",prefixes[0]},{"T1_inputs",std::vector<llama_token>(continuations[0].begin(),continuations[0].begin()+4)},{"count",4},{"timed",false},{"before",stats()}};
        begin_prefix(0);need(finite(compute()),"preparation prefix finite");for(int i=0;i<4;++i){set_one(continuations[0][i],lengths[0]+i);need(finite(compute()),"preparation T1 finite");}llama_synchronize(ctx);prep["after"]=stats();validate_calls(prep["before"],prep["after"],4);report["preparation"]=prep;
        llama_memory_clear(llama_get_memory(ctx),true);llama_synchronize(ctx);report["blocks"]=json::array();report["answers"]=json::array();save_json(out/"RUN.json",report);
        auto block=[&](int ci,char mode,bool quality){select(mode);auto before=stats();begin_prefix(ci);auto pre0=ns();const float *ptr=compute();auto pre1=ns();need(finite(ptr),"prefix finite");auto after_prefix=stats();validate_calls(before,after_prefix,0);
            std::vector<float>values;if(quality){values.resize(size_t(65)*nv);std::memcpy(values.data(),ptr,size_t(nv)*sizeof(float));}
            std::array<int64_t,64>start{},ready{};std::array<int,64>code{};std::array<uint64_t,64>builds{};
            for(int i=0;i<64;++i){set_one(continuations[ci][i],lengths[ci]+i);start[i]=ns();code[i]=llama_decode(ctx,batch);ptr=llama_get_logits_ith(ctx,-1);ready[i]=ns();need(code[i]==0&&ptr,"bounded T1 status");
                if(quality){need(finite(ptr),"quality finite");std::memcpy(values.data()+size_t(i+1)*nv,ptr,size_t(nv)*sizeof(float));if(mixed)builds[i]=stats()["custom_nodes_built"].get<uint64_t>()-after_prefix["custom_nodes_built"].get<uint64_t>();}}
            llama_synchronize(ctx);auto after=stats();validate_calls(after_prefix,after,64);
            json b={{"case_id",names[ci]},{"family_id",input["cases"][ci]["family_id"]},{"split","validation"},{"mode",std::string(1,mode)},{"quality",quality},{"prefix_ids",prefixes[ci]},
                {"prefix_count",lengths[ci]},{"input_tokens",std::vector<llama_token>(continuations[ci].begin(),continuations[ci].begin()+64)},{"next_targets",continuations[ci]},
                {"positions",input["cases"][ci]["execution_input"]["input_absolute_positions64"]},{"start_ns",start},{"output_ready_ns",ready},{"status",code},{"prefill_begin_ns",pre0},{"prefill_ready_ns",pre1},
                {"before_prefix",before},{"before_T1",after_prefix},{"after",after},{"custom_builds_after_each_T1",quality&&mixed?json(builds):json(nullptr)}};
            if(quality){std::string name="quality-"+names[ci]+"-"+mode+".f32";write_f32(name,values);b["file"]=name;b["shape"]={65,nv};b["all_finite"]=true;}
            report["blocks"].push_back(b);save_json(out/"RUN.json",report);std::cout<<stage<<' '<<names[ci]<<' '<<mode<<" complete\n"<<std::flush;
        };
        if(stage=="quality"){
            report["S_P_byte_gates"]=json::array();
            for(int ci:order){if(arm=="F")block(ci,'F',true);else{block(ci,'S',true);block(ci,'P',true);auto s=load_bytes("quality-"+names[ci]+"-S.f32"),p=load_bytes("quality-"+names[ci]+"-P.f32");bool ok=s==p&&s.size()==size_t(65)*nv*sizeof(float);report["S_P_byte_gates"].push_back({{"case_id",names[ci]},{"equal",ok}});save_json(out/"RUN.json",report);need(ok,"S/P mismatch; stop before timing or free answers");}}
        }else if(stage=="timing"){
            for(int ci:order)block(ci,arm[0],false);
        }else{
            // Greedy sampler uses the same pinned libllama. No repeat penalties or grammar.
            report["generation_contract"]={{"policy","greedy"},{"temperature",0},{"seed",42},{"seed_used",false},{"ignore_eos",false},{"max_generated_ids",1024},{"reference_supplied",false},{"all_generated_ids_including_control_counted",true}};
            report["answer_function_providers"]=json::array();std::string expected;
            for(const auto &v:pin["providers"])if(v["symbol"]=="llama_decode")expected=v["path"];
            for(const auto &entry:std::vector<std::pair<const char*,void*>>{{"llama_sampler_init_greedy",reinterpret_cast<void*>(llama_sampler_init_greedy)},{"llama_sampler_apply",reinterpret_cast<void*>(llama_sampler_apply)},{"llama_vocab_is_eog",reinterpret_cast<void*>(llama_vocab_is_eog)},{"llama_token_to_piece",reinterpret_cast<void*>(llama_token_to_piece)}}){Dl_info d{};need(dladdr(entry.second,&d)&&d.dli_fname,"answer function provider");std::string path=std::filesystem::canonical(d.dli_fname);need(path==expected,"answer same libllama");report["answer_function_providers"].push_back({{"symbol",entry.first},{"path",path}});}
            auto hex=[](const std::string &x){const char *digits="0123456789abcdef";std::string s;s.reserve(x.size()*2);for(unsigned char c:x){s+=digits[c>>4];s+=digits[c&15];}return s;};
            auto piece=[&](llama_token id,bool special){std::vector<char>b(256);int n=llama_token_to_piece(vocab,id,b.data(),b.size(),0,special);if(n<0){b.resize(-n);n=llama_token_to_piece(vocab,id,b.data(),b.size(),0,special);}need(n>=0&&n<=int(b.size()),"token piece");return std::string(b.data(),n);};
            std::vector<llama_token_data>candidates(nv);std::vector<llama_token>eog_ids;for(int i=0;i<nv;++i)if(llama_vocab_is_eog(vocab,i))eog_ids.push_back(i);report["model_eog_ids"]=eog_ids;save_json(out/"RUN.json",report);
            for(int ci:order){select(arm[0]);auto before=stats();begin_prefix(ci);auto a0=ns();const float *ptr=compute();auto pre_end=ns();need(finite(ptr),"answer prefix finite");auto after_prefix=stats();validate_calls(before,after_prefix,0);
                std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>sampler(llama_sampler_init_greedy(),llama_sampler_free);need(bool(sampler),"greedy sampler");
                std::string stem="answer-"+names[ci]+"-"+arm;need(!std::filesystem::exists(out/(stem+".json")),"answer fresh");std::ofstream events(out/(stem+".jsonl"));need(bool(events),"answer event log");
                std::vector<llama_token>generated;std::string all,visible,stop;int decoded=0;int64_t last_ready=pre_end;
                save_json(out/(stem+"-started.json"),{{"case_id",names[ci]},{"mode",arm},{"prefix_ids",prefixes[ci]},{"max_generated_ids",1024},{"reference_supplied",false},{"prefill_begin_ns",a0},{"prefill_ready_ns",pre_end}});
                for(int n=0;n<1024;++n){need(finite(ptr),"answer finite logits");for(int t=0;t<nv;++t)candidates[t]={t,ptr[t],0.0f};llama_token_data_array data{candidates.data(),candidates.size(),-1,false};
                    auto sample0=ns();llama_sampler_apply(sampler.get(),&data);auto sample1=ns();need(data.selected>=0&&data.selected<int64_t(data.size),"greedy selection");llama_token token=data.data[data.selected].id;llama_sampler_accept(sampler.get(),token);generated.push_back(token);
                    bool eog=llama_vocab_is_eog(vocab,token);auto pa=piece(token,true),pv=piece(token,false);all+=pa;visible+=pv;
                    json event={{"ordinal",n},{"token_id",token},{"is_eog",eog},{"logits_ready_ns",last_ready},{"sample_begin_ns",sample0},{"sample_end_ns",sample1},{"piece_all_hex",hex(pa)},{"piece_visible_hex",hex(pv)}};
                    if(eog)stop="eog";else if(n==1023)stop="length";
                    if(stop.empty()){set_one(token,lengths[ci]+n);auto d0=ns();ptr=compute();last_ready=ns();++decoded;event["decode_begin_ns"]=d0;event["decode_ready_ns"]=last_ready;if(mixed)event["custom_nodes_built"]=stats()["custom_nodes_built"];}
                    events<<event.dump()<<'\n';events.flush();need(bool(events),"answer event write");if(!stop.empty())break;
                }
                llama_synchronize(ctx);auto finish=ns();auto after=stats();validate_calls(after_prefix,after,decoded);need(!stop.empty()&&decoded==int(generated.size())-1,"answer stop accounting");
                for(const auto &entry:std::array<std::pair<std::string,std::string>,2>{{{stem+"-all.utf8",all},{stem+"-visible.utf8",visible}}}){std::ofstream f(out/entry.first,std::ios::binary);f.write(entry.second.data(),entry.second.size());f.close();need(!f.fail(),"answer text write");}
                json answer={{"case_id",names[ci]},{"mode",arm},{"prefix_ids",prefixes[ci]},{"generated_ids",generated},{"generated_count",generated.size()},{"decoded_T1_calls",decoded},{"stop_reason",stop},{"truncated",stop=="length"},{"terminal_token_id",stop=="eog"?json(generated.back()):json(nullptr)},
                    {"all_text_file",stem+"-all.utf8"},{"visible_text_file",stem+"-visible.utf8"},{"events_file",stem+".jsonl"},{"prefill_begin_ns",a0},{"prefill_ready_ns",pre_end},{"finish_ns",finish},{"before_prefix",before},{"before_T1",after_prefix},{"after",after},{"reference_supplied",false}};
                save_json(out/(stem+".json"),answer);report["answers"].push_back(answer);save_json(out/"RUN.json",report);std::cout<<"answer "<<names[ci]<<' '<<arm<<" generated="<<generated.size()<<" stop="<<stop<<'\n'<<std::flush;
            }
        }
        memory("M3_after_stage");closed();report["passed"]=true;save_json(out/"RUN.json",report);
        if(resources.vision){mtmd_free(resources.vision);resources.vision=nullptr;}if(resources.has_batch){llama_batch_free(batch);resources.has_batch=false;}resources.init.reset();llama_backend_free();std::cout<<"validation stage passed\n";return 0;
    }catch(const std::exception&e){report["error"]=e.what();try{if(ctx&&!resources.abandon)llama_synchronize(ctx);}catch(...){resources.abandon=true;report["cleanup_incomplete"]=true;}report["unsafe_cleanup_avoided"]=resources.abandon;if(!out.empty())try{save_json(out/"RUN.json",report);}catch(...){}std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
