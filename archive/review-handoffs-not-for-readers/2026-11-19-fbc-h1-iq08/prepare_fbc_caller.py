from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;LAB=R.parent;F=R/'fbc';OLD=LAB/'validation3-20261117/caller'
def main():
    s=(OLD/'driver.cpp').read_text();assert hashlib.sha256(s.encode()).hexdigest()==json.loads((OLD/'BUILD.json').read_text())['source_sha256']['driver.cpp']
    s='#include "fbc-trace.h"\n#include "log.h"\n'+s
    a=s.index('        need((stage=="quality"');b=s.index('        common_params params;',a)
    s=s[:a]+'''        need((stage=="quality"||stage=="timing")&&(arm=="F"||arm=="B"||arm=="C"),"FBC declaration");
        const bool has_bank=arm!="F",uses_custom=arm=="C";
        report["spec"]=spec;report["arm"]=arm;report["stage"]=stage;report["has_bank"]=has_bank;report["uses_custom"]=uses_custom;
        const char *flag=std::getenv("GGML_XEON_HOT_TRIAL_KEEP_ORIGINAL_GRAPH");
        need(flag&&std::string(flag)==(arm=="B"?"1":"0"),"context-fixed B flag");
'''+s[b:]
    s=s.replace('mixed','has_bank')
    s=s.replace('report["scope"]="Fixed original35 and unchanged engine. Three previously unused validation variants; direct frozen IDs; separate quality/timing/greedy-answer stages."','report["scope"]="Same new FBC engine; fixed original35, diagnosticCPU602a, trainingJA/Python. B retains bank but original main graph. No adoption."')
    s=s.replace('        llama_backend_init();','        common_log_set_verbosity_thold(stage=="quality"?LOG_LEVEL_DEBUG:LOG_LEVEL_INFO);llama_log_set(common_log_default_callback,nullptr);\n        llama_backend_init();',1)
    s=s.replace('report["bank_used_by_model"]=has_bank;','report["bank_used_by_model"]=uses_custom;')
    a=s.index('        auto input=read_json(cases_path);');b=s.index('        std::vector<int>order;',a)
    s=s[:a]+'''        auto input=read_json(cases_path);need(input["cases"].size()==2,"two training cases");
        std::vector<std::vector<llama_token>>prefixes,continuations;std::map<std::string,int>index;
        const std::array<std::string,2>names={"text_cache_budget_ja","code_stable_grouping_python"};
        const std::array<int,2>lengths={272,240};
        for(int i=0;i<2;++i){const auto &c=input["cases"][i];
            need(c["case_id"]==names[i]&&c["split"]=="train"&&c["prefix_count"]==lengths[i],"fixed training cases");
            std::vector<llama_token>pr=c["prefix_ids"],co=c["next_targets_prefix_and64T1"],in=c["used_T1_ids"];
            need(pr.size()==size_t(lengths[i])&&co.size()==65&&in.size()==64&&std::equal(in.begin(),in.end(),co.begin()),"input boundaries");
            for(auto t:pr){need(t>=0&&t<nv,"prefix token");}
            for(auto t:co){need(t>=0&&t<nv,"target token");}
            index[names[i]]=i;prefixes.push_back(pr);continuations.push_back(co);
            report["cases"].push_back({{"case_id",names[i]},{"family_id",c["family_id"]},{"split","train"},{"prefix_ids",pr},{"prefix_count",pr.size()},{"teacher_forced_input64",in},{"next_targets65",co},{"prefix_i32_sha256",c["prefix_ids_i32_sha256"]},{"input_i32_sha256",c["used_T1_ids_i32_sha256"]}});
        }
        using cache_fn=size_t(*)();auto cache_total=reinterpret_cast<cache_fn>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_total_v1"));need(cache_total,"actual cache count");
        Dl_info cache_info{};need(dladdr(reinterpret_cast<void*>(cache_total),&cache_info),"cache provider");
        report["cache_provider"]=std::filesystem::canonical(cache_info.dli_fname).string();
        std::string cache_expected;for(const auto &v:pin["providers"])if(v["symbol"]=="ggml_backend_cpu_init")cache_expected=v["path"].get<std::string>();need(report["cache_provider"]==cache_expected,"cache same CPU provider");report["cache_allocated_bytes"]=cache_total();need(cache_total()==127401984000ULL,"fixed actual compact allocation");
        const std::set<int>selected_layers={1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,19,20,22,28,29,33,35,36};
'''+s[b:]
    a=s.index('        auto stats=[&]()');b=s.index('        auto begin_prefix=',a)
    s=s[:a]+'''        auto stats=[&](){xeon_hot_trial_stats x{};json j;
            const int present=xeon_hot_trial_snapshot(outer_cpu,&x);need(bool(present)==has_bank,"bank state presence");
            if(has_bank){need(x.ready&&x.payload==475453440&&x.allocation>=x.payload,"bank state/payload");
                j={{"ready",x.ready},{"mode",std::string(1,x.mode)},{"custom_nodes_built",x.builds},{"native_bytes",x.payload},{"bank_allocation_bytes",x.allocation},{"calls",json::array()},{"executed_H",json::array()}};
                for(int l=0;l<40;++l){j["calls"].push_back(x.calls[l]);j["executed_H"].push_back(std::vector<uint64_t>(x.hist[l],x.hist[l]+7));if(!uses_custom){need(x.builds==0&&x.calls[l]==0,"B has no custom execution");for(auto h:x.hist[l])need(h==0,"B has no histogram calls");}}
            }return j;
        };
        auto validate_calls=[&](const json&a,const json&b,int calls){if(!has_bank)return;
            for(int l=0;l<40;++l){const uint64_t want=uses_custom&&selected_layers.count(l)?calls:0;
                need(b["calls"][l].get<uint64_t>()-a["calls"][l].get<uint64_t>()==want,"exact selected layer calls");
                for(int h=0;h<7;++h)need(b["executed_H"][l][h].get<uint64_t>()-a["executed_H"][l][h].get<uint64_t>()==(h==0?want:0),"C executed H0 only");
            }
        };
        auto select=[&](char mode){llama_synchronize(ctx);closed();need(std::string(1,mode)==arm,"single arm per context");if(uses_custom)need(xeon_hot_trial_set_mode(outer_cpu,'C')==1,"C mode");};
        report["initial_bank"]=stats();
'''+s[b:]
    s=s.replace("select(has_bank?'P':'F')","select(arm[0])")
    s=s.replace('{"split","validation"}','{"split","train"}')
    s=s.replace('{"positions",input["cases"][ci]["execution_input"]["input_absolute_positions64"]}','{"positions",[&](){std::vector<int>p;for(int i=0;i<64;++i)p.push_back(lengths[ci]+i);return p;}()}')
    needle='for(int i=0;i<64;++i){set_one(continuations[ci][i],lengths[ci]+i);start[i]=ns();'
    assert needle in s
    s=s.replace(needle,'''for(int i=0;i<64;++i){set_one(continuations[ci][i],lengths[ci]+i);
                if(quality&&i==0){common_log_flush(common_log_main());LOG_INF("FBC_TRACE_BEGIN case=%s arm=%s\\n",names[ci].c_str(),arm.c_str());common_log_flush(common_log_main());xeon_fbc_trace_arm((out/(names[ci]+"-graph")).c_str());}
                start[i]=ns();''')
    needle='need(code[i]==0&&ptr,"bounded T1 status");'
    s=s.replace(needle,needle+'''if(quality&&i==0){need(!xeon_fbc_trace_pending(),"trace produced without forcing reuse");common_log_flush(common_log_main());LOG_INF("FBC_TRACE_END case=%s arm=%s\\n",names[ci].c_str(),arm.c_str());common_log_flush(common_log_main());}''')
    needle='report["blocks"].push_back(b);save_json(out/"RUN.json",report);'
    s=s.replace(needle,'''if(!quality){std::vector<float>last(ptr,ptr+nv);write_f32("last-"+names[ci]+".f32",last);b["last_file"]="last-"+names[ci]+".f32";}
            '''+needle,1)
    a=s.index('        if(stage=="quality"){');b=s.index('        memory("M3_after_stage")',a)
    s=s[:a]+'''        for(int ci:order)block(ci,arm[0],stage=="quality");
        report["final_bank"]=stats();
'''+s[b:]
    s=s.replace('need(!order.empty(),"cases required");','need(order.size()==2,"both fixed cases required");')
    s=s.replace('including hot initialization for CSP','including hot initialization for B/C').replace('Bank is used in CSP T1 only','Bank custom path is used only for C T1, executed GPU H0').replace('Free answer receives prefix only.','No free answers in this diagnostic.').replace('validation stage passed','FBC stage passed')
    s='\n'.join(line for line in s.split('\n') if 'auto load_bytes=' not in line)
    (F/'driver.cpp').write_text(s)
    # Guard selectedlayers against the exact original35 plan before compile.
    layers={int(x.split()[0]) for x in (F/'capacity.plan').read_text().splitlines()[1:] if int(x.split()[4])}
    assert layers=={1,2,3,4,5,6,7,8,10,11,13,14,15,16,17,19,20,22,28,29,33,35,36}
    stages=[dict(name='quality-'+arm,mode=arm,stage='quality',cases=['text_cache_budget_ja','code_stable_grouping_python']) for arm in 'FBC']
    for i,arm in enumerate('FBCCBF'):stages.append(dict(name=f'timing-{i}-{arm}',mode=arm,stage='timing',cases=['code_stable_grouping_python','text_cache_budget_ja'] if i>=3 else ['text_cache_budget_ja','code_stable_grouping_python']))
    (F/'DESIGN.json').write_text(json.dumps(dict(stages=stages,provider='diagnosticCPU602a/CUDA45/base2e13; same newFBC llama allarms',quality_trace='preallocation originalDAG and postallocation metadata plusGGML_SCHED_DEBUG2, allqualityarms; disabledtiming',clock='decode start to normal logits ready',timed_calls=768,prepare='sameJA prefix64 warm then full272prefix+4referenceT1; synchronize,clearKV,targetprefix; countersafterprep',selected_layers=sorted(layers)),indent=2)+'\n')
if __name__=='__main__':main()
