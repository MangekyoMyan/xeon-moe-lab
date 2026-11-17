"""Offline numeric/timing reports for the frozen three-case screen."""
from pathlib import Path
import importlib.util,json,statistics,sys
from verify_evidence import R,read,load_run,common,sha,write

OLD=R.parent/'parallel-pilot-20261117'
sys.path.insert(0,str(OLD))
s=importlib.util.spec_from_file_location('prior_quality_math',OLD/'analyze_integrated.py');mathmod=importlib.util.module_from_spec(s);s.loader.exec_module(mathmod)
def quant(v):
    x=sorted(v)
    def q(t):
        at=(len(x)-1)*t;i=int(at);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(at-i)
    return dict(count=len(x),mean=statistics.mean(x),median=statistics.median(x),min=x[0],p95=q(.95),p99=q(.99),max=x[-1])
def quality():
    assert read(R/'QUALITY_GATE.json')['passed']
    f=load_run('quality-F');p=load_run('quality-SP');common([f,p]);result=dict(passed=True,scope='Fixed reference trajectory; not natural-answer correctness',cases=[])
    for fb in f[1]['blocks']:
        ci=fb['case_id'];out={};diag={};case=dict(case_id=ci,prefix_count=fb['prefix_count'],positions=fb['positions'],quality={},comparison={})
        for mode,run,block in [('F',f,fb)]+[(m,p,next(b for b in p[1]['blocks'] if b['case_id']==ci and b['mode']==m)) for m in ['S','P']]:
            raw,values=mathmod.logits(run[0]/block['file']);out[mode]=(raw,values);diag[mode]=mathmod.row_diagnostics(raw,values,fb['next_targets']);case['quality'][mode]=diag[mode]
            if mode=='P':case['custom_node_builds_after_T1']=block['custom_builds_after_each_T1'];case['before_T1']=block['before_T1'];case['after']=block['after']
        for left,right in [('S','P'),('P','F')]:case['comparison'][left+'_vs_'+right]=mathmod.comparison(*out[left],*out[right],diag[left],diag[right])
        assert case['comparison']['S_vs_P']['byte_equal']
        # Keep each row's error, not only an aggregate across different logits magnitudes.
        a=out['P'][1];b=out['F'][1];n=mathmod.NV;per=[]
        import math
        for i in range(65):
            av=a[i*n:(i+1)*n];bv=b[i*n:(i+1)*n];err=math.fsum((float(x)-y)**2 for x,y in zip(av,bv));norm=math.fsum(float(y)*y for y in bv)
            delta=math.fsum(float(x)-y for x,y in zip(av,bv))/n
            target=fb['next_targets'][i];zf=diag['F'][i]['NLL']+bv[target];zp=diag['P'][i]['NLL']+av[target]
            pf=[math.exp(x-zf) for x in bv];pp=[math.exp(x-zp) for x in av]
            kl=math.fsum(prob*(float(x)-y+zp-zf) for prob,x,y in zip(pf,bv,av))
            tv=.5*math.fsum(abs(x-y) for x,y in zip(pf,pp))
            per.append(dict(row=i,NMSE=err/norm if norm else None,max_abs=max(abs(float(x)-y) for x,y in zip(av,bv)),mean_logit_shift=delta,
                centered_max_abs=max(abs(float(x)-y-delta) for x,y in zip(av,bv)),KL_F_to_P=kl,total_variation=tv,probability_sums=[math.fsum(pf),math.fsum(pp)]))
        case['P_vs_F_rows']=per;result['cases'].append(case)
    write(R/'QUALITY_ANALYSIS.json',result)
    print(json.dumps([dict(case_id=c['case_id'],top1_changes=len(c['comparison']['P_vs_F']['top1_changed_rows']),NLL_delta=c['comparison']['P_vs_F']['mean_reference_token_NLL_delta'],custom_build_end=c['custom_node_builds_after_T1'][-1]) for c in result['cases']]))
def timing():
    assert read(R/'TIMING_COMPLETE.json')['passed']
    runs=[load_run(n) for n in ['timing-0-F','timing-1-P','timing-2-P','timing-3-F']];common(runs)
    plan=read(R/'caller/FROZEN_PLAN.json');result=dict(passed=True,scope='3variants/2families;2independentprocesses per mode; teacher-forced64 includes first call; no sampling/API/UI',cases=[])
    for ci in plan['selected_case_ids_in_fixed_order']:
        blocks=[]
        for x in runs:
            b=next(b for b in x[1]['blocks'] if b['case_id']==ci);wall=[(e-s)/1e6 for s,e in zip(b['start_ns'],b['output_ready_ns'])];gaps=[(b['start_ns'][i+1]-b['output_ready_ns'][i])/1e6 for i in range(63)]
            assert len(wall)==64 and min(wall)>0 and min(gaps)>=0
            blocks.append(dict(process=x[1]['spec']['name'],mode=b['mode'],ms=wall,total_ms=sum(wall),mean_ms=statistics.mean(wall),T1_s=64000/sum(wall),first_T1_ms=wall[0],prefill_ms=(b['prefill_ready_ns']-b['prefill_begin_ns'])/1e6,intercall_gap_ms=gaps,
                before_T1=b['before_T1'],after=b['after']))
        sums={}
        for m in ['F','P']:
            bb=[b for b in blocks if b['mode']==m];v=[t for b in bb for t in b['ms']];assert len(v)==128
            sums[m]=dict(T1_s=128000/sum(v),mean_ms=statistics.mean(v),token_ms=quant(v),block_mean_ms=[b['mean_ms'] for b in bb],prefill_ms=[b['prefill_ms'] for b in bb])
        pair=[dict(F_mean_ms=blocks[fi]['mean_ms'],P_mean_ms=blocks[pi]['mean_ms'],saved_ms=blocks[fi]['mean_ms']-blocks[pi]['mean_ms'],speed_ratio=blocks[fi]['mean_ms']/blocks[pi]['mean_ms']) for fi,pi in [(0,1),(3,2)]]
        c=dict(case_id=ci,prefix_count=next(c['execution_input']['prefix_count'] for c in plan['cases'] if c['case_id']==ci),blocks=blocks,summary=sums,paired=pair,speed_ratio=sums['P']['T1_s']/sums['F']['T1_s'],saved_ms=sums['F']['mean_ms']-sums['P']['mean_ms'],F_before_after_difference_ms=blocks[3]['mean_ms']-blocks[0]['mean_ms'])
        result['cases'].append(c)
    write(R/'TIMING_ANALYSIS.json',result);print(json.dumps([dict(case_id=c['case_id'],F=c['summary']['F']['T1_s'],P=c['summary']['P']['T1_s'],ratio=c['speed_ratio'],pair_ratios=[p['speed_ratio'] for p in c['paired']]) for c in result['cases']]))
if __name__=='__main__':
    if sys.argv[1]=='quality':quality()
    elif sys.argv[1]=='timing':timing()
    else:raise SystemExit('quality or timing')
