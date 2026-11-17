"""Analyze the one exploratory repeat separately from the initial screen."""
from pathlib import Path
import json,statistics
from verify_evidence import load_run,common,write,sha
R=Path(__file__).resolve().parent
def main():
    assert json.loads((R/'COMPLETE.json').read_text())['passed']
    rr=[load_run(f'timing-{i}-{m}') for i,m in enumerate(['F','P','P','F'])];common(rr);blocks=[]
    for path,j,manifest,launch in rr:
        assert len(j['blocks'])==1
        b=j['blocks'][0];assert b['case_id']=='code_bounded_queue_python'
        ms=[(e-s)/1e6 for s,e in zip(b['start_ns'],b['output_ready_ns'])];assert len(ms)==64 and min(ms)>0
        blocks.append(dict(name=j['spec']['name'],mode=b['mode'],ms=ms,mean_ms=statistics.mean(ms),T1_s=64000/sum(ms),first_T1_ms=ms[0],prefill_ms=(b['prefill_ready_ns']-b['prefill_begin_ns'])/1e6,before_T1=b['before_T1'],after=b['after']))
    sums={}
    for m in ['F','P']:
        bb=[b for b in blocks if b['mode']==m];v=[t for b in bb for t in b['ms']]
        sums[m]=dict(T1_s=128000/sum(v),mean_ms=statistics.mean(v),block_mean_ms=[b['mean_ms'] for b in bb],min_ms=min(v),max_ms=max(v))
    pairs=[dict(F_mean_ms=blocks[fi]['mean_ms'],P_mean_ms=blocks[pi]['mean_ms'],speed_ratio=blocks[fi]['mean_ms']/blocks[pi]['mean_ms']) for fi,pi in [(0,1),(3,2)]]
    result=dict(scope='Exploratory Python-only extra ABBA, selected after initial weak gain. Same binary/IDs/DSOs, common prep; different case-only sequence. Report separately; all256calls retained; no further repeat.',decision_sha256=sha(R/'DECISION.json'),blocks=blocks,summary=sums,paired=pairs,speed_ratio=sums['P']['T1_s']/sums['F']['T1_s'],saved_ms=sums['F']['mean_ms']-sums['P']['mean_ms'],F_before_after_difference_ms=blocks[3]['mean_ms']-blocks[0]['mean_ms'])
    write(R/'ANALYSIS.json',result);print(json.dumps({k:v for k,v in result.items() if k!='blocks'}))
if __name__=='__main__':main()
