"""Analyze complete frozen operator runs; no engine or API execution."""
from pathlib import Path
import array,hashlib,json,math,statistics
R=Path(__file__).resolve().parent;Q=R/'q8-avx1';RUNS=Q/'runs-v2'
CAPTURE_OVERRIDE=None  # Offline public replay may supply the previous frozen archive.
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    state=read(Q/'STATE-v2.json');design=read(Q/'FFN_DESIGN.json');runs={};copies=0
    capfile=Path(CAPTURE_OVERRIDE) if CAPTURE_OVERRIDE else Path(design['common']['capture_json']);capture=read(capfile);caproot=capfile.parent
    for stage in design['stages']:
        st=state['runs'][stage['name']];assert st['status']=='passed';run=Path(st['path']);assert sha(run/'RESULTS_SHA256.json')==st['manifest_sha256']
        for n,h in read(run/'RESULTS_SHA256.json').items():assert sha(run/n)==h,(run,n)
        health=read(run/'health.json');assert health['status']=='exited' and health['child_returncode']==0
        j=read(run/'RUN.json');assert j['passed'];assert read(run/'LAUNCH.json')['design_sha256']==sha(Q/'FFN_DESIGN.json')
        blocks={}
        for b in j['blocks']:
            key=(b['cell'],b['threads']);assert key not in blocks;blocks[key]=b
            raw=(run/b['output_file']).read_bytes();old=(caproot/capture['cells'][b['cell']]['down_file']).read_bytes();assert raw==old and len(raw)==5120*6*4
            values=array.array('f');values.frombytes(raw);assert all(math.isfinite(v) for v in values);copies+=1
            assert sorted(b['worker_receipt_after_timing']['CPU'])==list(range(b['threads']))
            assert len(b['samples'])==stage['samples'] and all(s['calls']==stage['loops'] and s['end_ns']>s['begin_ns'] for s in b['samples'])
        runs[stage['name']]=dict(path=str(run),blocks=blocks)
    output=[];total=0
    order=['control','candidate','candidate','control','candidate','control','control','candidate']
    for cell in [1,5,6]:
        for threads in [12,24]:
            processes=[]
            for i,mode in enumerate(order):
                name=f'timing-{i}-{mode}';b=runs[name]['blocks'][cell,threads]
                assert b['weights']==runs['quality-control']['blocks'][cell,threads]['weights']
                v=[(s['end_ns']-s['begin_ns'])/s['calls']/1e6 for s in b['samples']];total+=sum(s['calls'] for s in b['samples'])
                processes.append(dict(process=i,name=name,mode=mode,sample_ms_per_expert_group=v,median_ms=statistics.median(v),mean_ms=statistics.mean(v)))
            pairs=[]
            for c,p in [(0,1),(3,2),(5,4),(6,7)]:
                a=processes[c]['median_ms'];b=processes[p]['median_ms'];pairs.append(dict(control_process=c,candidate_process=p,control_ms=a,candidate_ms=b,candidate_over_control=b/a,saved_ms=a-b))
            ratios=[p['candidate_over_control'] for p in pairs];ratio=statistics.median(ratios)
            output.append(dict(cell=cell,layer=capture['cells'][cell]['layer'],types=[p['type'] for p in capture['cells'][cell]['projections']],threads=threads,processes=processes,pairs=pairs,median_time_ratio=ratio,time_reduction_percent=100*(1-ratio),pairs_faster=sum(x<1 for x in ratios),pairs_total=4,min_ratio=min(ratios),max_ratio=max(ratios),control_median_ms=statistics.median(p['median_ms'] for p in processes if p['mode']=='control'),candidate_median_ms=statistics.median(p['median_ms'] for p in processes if p['mode']=='candidate')))
    assert total==1920
    result=dict(passed=True,complete_processes=len(runs),timed_graph_calls=total,saved_output_files_checked=copies,F32_values_in_saved_files=copies*5120*6,all_saved_outputs_byte_equal_capture=True,rows=output,
        scope='CPU3projections+clampedactivation producing6expertoutputs. Excludesroutingweight/merge, model/GPU/API. Fourindependent pairedprocesses percell/thread;5samples of8calls withinprocess are not independentprocesses. Outputcheckedafterwarmupseries/lastcallofsample, noteachcall.',
        comparison='Existing current CPU byte-reproducedcontrol versus onlyQ8_K wrappercandidate; samebase/realGGUF/savedinput/selectedcompact/NUMAstrategy. 24thread T1geometry is notprefill.')
    p=Q/'FFN_ANALYSIS.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ['processes','pairs']} for x in output],indent=2))
if __name__=='__main__':main()
