"""Saved-evidence summaries only. No inference, timers, or runtime changes."""
from pathlib import Path
import hashlib,json,statistics,sys
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):assert not p.exists();p.write_text(json.dumps(j,indent=2)+'\n')
def saved_run(st):
    p=Path(st['path']);assert st['status']=='passed' and sha(p/'RESULTS_SHA256.json')==st['manifest_sha256']
    for n,h in read(p/'RESULTS_SHA256.json').items():assert sha(p/n)==h
    return p
def fbc():
    from verify_fbc import verify_run
    root=R/'fbc';state=read(root/'STATE.json');assert state['completed'];runs=[];boots=[]
    for i,a in enumerate('FBCCBF'):
        p=saved_run(state['runs'][f'timing-{i}-{a}']);j=verify_run(p);boots.append(read(p/'health.json')['boot_id']);runs.append((p,j))
    assert len(set(boots))==1
    rows=[]
    for case in ['text_cache_budget_ja','code_stable_grouping_python']:
        blocks=[]
        for i,(p,j) in enumerate(runs):
            b=next(x for x in j['blocks'] if x['case_id']==case);ms=[(e-s)/1e6 for s,e in zip(b['start_ns'],b['output_ready_ns'])];assert len(ms)==64
            blocks.append(dict(process=i,arm=j['arm'],raw_ms=ms,mean_ms=statistics.mean(ms),T1_s=64000/sum(ms),first_ms=ms[0],prefill_ms=(b['prefill_ready_ns']-b['prefill_begin_ns'])/1e6))
        arms={a:dict(mean_ms=statistics.mean(t for b in blocks if b['arm']==a for t in b['raw_ms']),T1_s=128000/sum(t for b in blocks if b['arm']==a for t in b['raw_ms']),block_mean_ms=[b['mean_ms'] for b in blocks if b['arm']==a]) for a in 'FBC'}
        contrasts={}
        for name,pairs in [('B_minus_F',[(1,0),(4,5)]),('C_minus_B',[(2,1),(3,4)]),('C_minus_F',[(2,0),(3,5)])]:
            values=[dict(lhs_process=l,rhs_process=r,difference_ms=blocks[l]['mean_ms']-blocks[r]['mean_ms'],time_ratio=blocks[l]['mean_ms']/blocks[r]['mean_ms']) for l,r in pairs]
            contrasts[name]=dict(pairs=values,average_ms=statistics.mean(v['difference_ms'] for v in values))
        rows.append(dict(case_id=case,blocks=blocks,arms=arms,contrasts=contrasts,F_last_minus_first_ms=blocks[5]['mean_ms']-blocks[0]['mean_ms']))
    out=dict(passed=True,timed_calls=768,rows=rows,quality_gate=read(root/'QUALITY_GATE.json'),scope='2trainingcases,2freshprocesses perarm/case; all64calls includingfirst. B-F/C-B compound differences, not purecopy or VRAM cost.')
    save(root/'ANALYSIS.json',out);print(json.dumps([dict(case=x['case_id'],arms=x['arms'],differences={k:v['average_ms'] for k,v in x['contrasts'].items()}) for x in rows],indent=2))
def h1():
    from verify_h1 import verify
    root=R/'h1';state=read(root/'STATE.json');assert state['completed'];out=[];boots=[]
    for s in read(root/'DESIGN.json')['stages']:
        p=saved_run(state['runs'][s['name']]);j=verify(p);boots.append(read(p/'health.json')['boot_id']);cells=[]
        for c in j['cells']:
            steady=[x for x in c['samples'] if not x['first_use']];pairs=[];times={a:[] for a in 'OCSP'};joins={a:[] for a in 'SP'}
            for b in range(12):
                values={x['mode']:x for x in steady if x['block']==b};ms={a:(x['output_ready_ns']-x['input_ready_ns'])/1e6 for a,x in values.items()}
                pairs.append(dict(block=b,C_minus_P_ms=ms['C']-ms['P'],S_minus_P_ms=ms['S']-ms['P'],C_minus_O_ms=ms['C']-ms['O'],O_over_P=ms['O']/ms['P']))
                for a in times:times[a].append(ms[a])
                for a in 'SP':
                    x=values[a]
                    if x['executed_gpu_H']:
                        if a=='S':assert x['join_end_ns']<=x['cpu_begin_ns']
                        else:assert x['cpu_end_ns']<=x['join_begin_ns']
                        joins[a].append((x['join_end_ns']-x['join_begin_ns'])/1e6)
            cells.append(dict(cell=c['cell'],layer=c['layer'],H=c['matched_H'],GPU_ids=c['gpu_global_ids'],primary=c['cell'] in [1,4],times_ms=times,mean_ms={a:statistics.mean(v) for a,v in times.items()},pairs=pairs,median_C_minus_P_ms=statistics.median(x['C_minus_P_ms'] for x in pairs),median_S_minus_P_ms=statistics.median(x['S_minus_P_ms'] for x in pairs),joins_ms=joins,join_scope='S preCPU join; P residual afterCPU. Neither is GPU totalduration.',correctness=c['correctness'],compact=c['compact']))
        out.append(dict(name=s['name'],plan=s['plan'],native_bytes=j['bank_native_bytes'],allocated_bytes=j['bank_allocated_bytes'],cells=cells))
    assert len(set(boots))==1
    summary=[]
    for ci in [1,4]:
        plans={}
        for plan in ['H1_COMPACT','H1_NONCOMPACT']:
            cc=[run['cells'][ci] for run in out if run['plan']==plan];plans[plan]=dict(process_C_minus_P_ms=[c['median_C_minus_P_ms'] for c in cc],process_S_minus_P_ms=[c['median_S_minus_P_ms'] for c in cc],process_P_mean_ms=[c['mean_ms']['P'] for c in cc])
        summary.append(dict(cell=ci,plans=plans))
    result=dict(passed=True,processes=out,primary_summary=summary,steady_observations=1536,first_use_observations=128,scope='Existing2primarycells, eachplan2freshprocesses;12withinprocessblocks not12independentprocesses. All8retained, cell3H3/H2 noallcellaverage. DifferentGPUassignment permitsdifferentPbytes; no immediatebankadoption.')
    save(root/'ANALYSIS.json',result);print(json.dumps(summary,indent=2))
def cpu08():
    from verify_08 import verify
    root=R/'cpu08';state=read(root/'STATE.json');assert state['completed'];order=['control','candidate','candidate','control','candidate','control','control','candidate'];runs=[];boots=[]
    for i,m in enumerate(order):
        p=saved_run(state['runs'][f'timing-{i}-{m}']);j=verify(p);boots.append(read(p/'health.json')['boot_id']);runs.append(j)
    assert len(set(boots))==1;rows=[]
    for ci in [1,5,6]:
        blocks=[]
        for i,j in enumerate(runs):
            b=next(x for x in j['blocks'] if x['cell']==ci);v=[(x['end_ns']-x['begin_ns'])/x['calls']/1e6 for x in b['samples']];blocks.append(dict(process=i,mode=order[i],position=j['spec']['cells'].index(ci),raw_ms=v,median_ms=statistics.median(v),planner_work_size=b['planner_work_size']))
        pairs=[]
        for a,b in [(0,1),(3,2),(5,4),(6,7)]:
            control,candidate=blocks[a],blocks[b];assert control['position']==candidate['position'];pairs.append(dict(control_process=a,candidate_process=b,position=control['position'],control_first=a<b,ratio=candidate['median_ms']/control['median_ms'],saved_ms=control['median_ms']-candidate['median_ms']))
        ratios=[x['ratio'] for x in pairs];rows.append(dict(cell=ci,blocks=blocks,pairs=pairs,median_ratio=statistics.median(ratios),median_saved_ms=statistics.median(x['saved_ms'] for x in pairs),faster_pairs=sum(x<1 for x in ratios),range=[min(ratios),max(ratios)]))
    result=dict(passed=True,timed_graph_calls=960,rows=rows,scope='08onlycurrent33a control, fixed12physical,2IQprimarycells andMXshared-pathcontrol; no09/04/03. Fourpairs, allrawretained; noMXsubtraction orposthoc24T.')
    save(root/'ANALYSIS.json',result);print(json.dumps([dict(cell=x['cell'],median_ratio=x['median_ratio'],median_saved_ms=x['median_saved_ms'],faster_pairs=x['faster_pairs'],range=x['range'],pairs=x['pairs']) for x in rows],indent=2))
if __name__=='__main__':{'fbc':fbc,'h1':h1,'08':cpu08}[sys.argv[1]]()
