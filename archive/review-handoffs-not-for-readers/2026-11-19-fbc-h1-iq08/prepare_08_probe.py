from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;D=R/'cpu08';OLD=R.parent/'fd810-followup-20261118/q8-avx1'
def main():
    D.mkdir(exist_ok=True)
    for n in ['iq08_quality_counts.cpp','iq08_quality_counts.h']:shutil.copy2(R/'astra62-review'/n,D/n)
    s=(OLD/'real_ffn.cpp').read_text();s='#include "iq08_quality_counts.h"\n#define GGML_COMMON_DECL_CPP\n#include "ggml-common.h"\n#include <cfenv>\n'+s
    s=s.replace('const json &cpuplan):cpu(b)', 'const json &cpuplan,bool direct_check):cpu(b)')
    needle='                record["weights"].push_back('
    direct='''                if(direct_check&&!compact.empty()){
                    using prep_fn=void(*)(const void*,void*,size_t);using select_fn=ggml_vec_dot_t(*)(ggml_type);
                    auto prep=reinterpret_cast<prep_fn>(dlsym(RTLD_DEFAULT,"xeon_iq_input_from_q8"));auto select=reinterpret_cast<select_fn>(dlsym(RTLD_DEFAULT,"xeon_iq_input_dot"));need(prep&&select,"08 direct functions");
                    ggml_vec_dot_t legacy=type==GGML_TYPE_IQ2_S?xeon_iq2s_compact_dot:type==GGML_TYPE_IQ3_XXS?xeon_iq3_compact_dot:xeon_iq3s_compact_dot;
                    need(provider(reinterpret_cast<void*>(legacy))==provider(reinterpret_cast<void*>(xeon_iq3_cache_add_selected_v2)),"legacy same CPU provider");
                    block_q8_K input[2]{};const int vals[8]={-128,-127,-1,0,1,2,126,127};
                    for(int b=0;b<2;++b){input[b].d=1.0f;for(int k=0;k<256;++k)input[b].qs[k]=(b==1&&k%8==0)?-127:vals[k%8];for(int g=0;g<16;++g){int sum=0;for(int k=0;k<16;++k)sum+=input[b].qs[g*16+k];input[b].bsums[g]=sum;}}
                    alignas(16) uint8_t folded[800]{};const int old_round=std::fegetround();json checks=json::array();
                    for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){need(std::fesetround(mode)==0,"direct rounding");prep(input,folded,2);float a=0,b=0;legacy(512,&a,0,table[compact[0]],0,input,0,1);select(type)(512,&b,0,table[compact[0]],0,folded,0,1);need(std::isfinite(a)&&std::memcmp(&a,&b,4)==0,"08 -128 direct legacy match");uint32_t bits;std::memcpy(&bits,&a,4);checks.push_back({{"rounding",mode},{"equal",true},{"output_bits",bits}});}
                    need(std::fesetround(old_round)==0,"restore direct rounding");record["direct_min128_checks"].push_back({{"type",ggml_type_name(type)},{"expert",compact[0]},{"blocks",2},{"second_block_has_min",false},{"legacy_provider",provider(reinterpret_cast<void*>(legacy))},{"comparison","candidate DSO legacy compact versus folded, same real packed weights"},{"modes",checks}});
                }
'''
    assert s.count(needle)==1;s=s.replace(needle,direct+needle)
    needle='    std::vector<std::unique_ptr<Shard>> shards;'
    hook='''    using snapshot_fn=int(*)(iq08_quality_counts_v1*,size_t);auto snapshot=reinterpret_cast<snapshot_fn>(dlsym(RTLD_DEFAULT,"xeon_iq08_quality_snapshot_v1"));
    const bool quality=spec["stage"]=="quality",candidate=spec["mode"]=="candidate";need(bool(snapshot)==quality,"quality-only counter presence");
    if(quality)need(provider(reinterpret_cast<void*>(snapshot))==fs::canonical(spec["quality_counter"].get<std::string>()).string(),"counter provider");
    auto snapshot_now=[&](){iq08_quality_counts_v1 a{};need(snapshot&&snapshot(&a,sizeof(a))==1&&a.abi_version==1&&a.struct_bytes==sizeof(a),"counter snapshot");return a;};
    auto binding=[&](const iq08_quality_binding_v1 &a){json j=nullptr;if(a.dso_path){std::string path=fs::canonical(a.dso_path);need(path==fs::canonical(spec["CPU"].get<std::string>()).string(),"underlying counter target CPU");j={{"path",path},{"offset",a.offset_from_dso_base}};}return j;};
'''
    assert s.count(needle)==1;s=s.replace(needle,hook+needle)
    s=s.replace('spec["cpu_plan"]);','spec["cpu_plan"],quality&&candidate);')
    a=s.index('            for(int i=0;i<3;++i) {fixture.compute();}');b=s.index('            const int loops=',a)
    s=s[:a]+'''            need(threads==12,"fixed12physical");const auto plan=ggml_graph_plan(fixture.graph,threads,pool);
            fixture.compute();fixture.compute();iq08_quality_counts_v1 before{},after{};if(quality)before=snapshot_now();fixture.compute();if(quality)after=snapshot_now();fixture.output();
            json block=fixture.record;block["threads"]=threads;block["planner_work_size"]=plan.work_size;block["output_matches_capture"]=true;block["samples"]=json::array();
            if(quality){json path;path["scope"]="one existing third warmup graph, quality-only interposer";path["pack_calls"]=after.pack_calls-before.pack_calls;path["pack_blocks"]=after.pack_blocks-before.pack_blocks;path["empty_pack_calls"]=after.empty_pack_calls-before.empty_pack_calls;path["underlying_pack"]=binding(after.underlying_pack);path["underlying_selector"]=binding(after.underlying_selector);path["types"]=json::array();need(after.other_selector_calls==before.other_selector_calls,"known selector types");
                const std::array<uint64_t,3>expected=(!candidate||ci==6)?std::array<uint64_t,3>{0,0,0}:ci==1?std::array<uint64_t,3>{23040,25600,0}:std::array<uint64_t,3>{0,27648,30720};
                need(after.pack_blocks-before.pack_blocks==(candidate&&ci!=6?94:0),"actual folded block count");
                for(int t=0;t<3;++t){auto &a=before.type[t];auto &b=after.type[t];need(b.other_nrc_calls==a.other_nrc_calls&&b.dot_rows_nrc1-a.dot_rows_nrc1==expected[t],"actual folded row count");path["types"].push_back({{"type_index",t},{"selector_calls",b.selector_calls-a.selector_calls},{"dot_rows_nrc1",b.dot_rows_nrc1-a.dot_rows_nrc1},{"underlying_dot",binding(b.underlying_dot)}});}block["path_check"]=path;
            }
'''+s[b:]
    (D/'real_ffn.cpp').write_text(s)
    design=json.loads((OLD/'FFN_DESIGN.json').read_text());specs=[]
    for mode in ['control','candidate']:specs.append(dict(name='quality-'+mode,mode=mode,stage='quality',cells=[1,6,5],threads=[12],samples=1,loops=1))
    for i,mode in enumerate(['control','candidate','candidate','control','candidate','control','control','candidate']):specs.append(dict(name=f'timing-{i}-{mode}',mode=mode,stage='timing',cells=[1,6,5] if i//2 in [0,2] else [5,6,1],threads=[12],samples=5,loops=8))
    design.update(stages=specs,timed_graph_calls=960,scope='08 only,3 real saved cells,12physical; independent current33a vs08; no04/09/03',process_design='Four pairs AB/BA/BA/AB; cellorders1,6,5 /5,6,1 /1,6,5 /5,6,1. Exact oldgraph clock includesplanning; no interposer in timing.',quality_path='Third existing warmup counters plus perprojection two-block -128/ordinary legacy-vs-folded direct check in four roundingmodes beforegraph.',NUMA='without_thp, numactl0..11 interleave0,1,ownerpacking12,pool12,rowlocal0')
    for k in ['CPU_build_sha256','FFN_build_sha256']:design.pop(k,None)
    (D/'DESIGN-prebuild.json').write_text(json.dumps(design,indent=2)+'\n');print('08 probe source/design prepared; not compiled')
if __name__=='__main__':main()
