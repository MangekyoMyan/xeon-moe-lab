#!/usr/bin/env python3
"""Independent offline descriptions from raw timing/65-row F32 files.

No supplied Python/ELF, inference, model, GPU, network or service is executed.
Repeated calls on ONE teacher-forced trajectory do not become independent tasks.
No sampling-distribution confidence interval or general-quality threshold is made.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

PIN = 'd4d50b651ff7754ee2189228762468fc02971bfa'
REL = 'archive/review-handoffs-not-for-readers/2026-11-17-parallel-pilot'


def summary(x):
    x = np.asarray(x,dtype=np.float64)
    return {'count':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'min':float(x.min()),'max':float(x.max()),
            'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo',type=Path)
    ap.add_argument('--out',type=Path,default=Path(__file__).parent)
    args=ap.parse_args();n=args.repo/REL;sha={}
    def read(p):
        raw=p.read_bytes();sha[str(p.relative_to(args.repo))]=hashlib.sha256(raw).hexdigest();return raw
    runs={}
    for p in sorted((n/'integrated/runs').glob('*/RUN.json')):
        j=json.loads(read(p));assert j['passed']
        runs[j['arm']]=(p.parent,j)
    assert set(runs)=={'F-before','CSP','F-after'}
    sample_groups={};time_blocks=[];qualities={};targets=None
    for arm,(directory,run) in runs.items():
        for index,b in enumerate(run['blocks']):
            start=np.asarray(b['start_ns'],dtype=np.int64)
            ready=np.asarray(b['output_ready_ns'],dtype=np.int64)
            assert start.shape==ready.shape==(64,)
            assert np.all(ready>start) and np.all(start[1:]>ready[:-1])
            assert b['status']==[0]*64 and b['prefix_tokens']==272
            label=arm if arm.startswith('F-') else b['mode']
            if targets is None:targets=np.asarray(b['next_targets'],dtype=np.int64)
            assert np.array_equal(targets,np.asarray(b['next_targets']))
            if b['quality']:
                raw=read(directory/b['file']);f=np.frombuffer(raw,dtype='<f4').reshape(65,129280)
                assert np.isfinite(f).all();qualities[label]=f.copy()
                continue
            duration=(ready-start)/1e6
            sample_groups.setdefault(label,[]).append(duration)
            time_blocks.append({'arm':label,'process_arm':arm,'index_in_process':index,
                                'repetition':b['repetition'],'start_ns':int(start[0]),
                                'mean_ms':float(duration.mean()),'total_ms':float(duration.sum()),
                                'prefill_ms':(b['prefill_ready_ns']-b['prefill_begin_ns'])/1e6,
                                'gap_us':summary((start[1:]-ready[:-1])/1000),
                                'raw_ms':duration.tolist()})
    sample_groups['F-bracket']=sample_groups['F-before']+sample_groups['F-after']
    stats={k:{'token_ms':summary(np.concatenate(v)),
              'block_mean_ms':summary([np.mean(b) for b in v]),
              't1_per_second':1000/float(np.concatenate(v).mean())} for k,v in sample_groups.items()}
    expected=json.loads(read(n/'INTEGRATED_ANALYSIS.json'))
    for k,v in stats.items():
        if k!='F-bracket':assert np.isclose(v['t1_per_second'],expected['timing_summary'][k]['tokens_per_second'],rtol=0,atol=1e-12)
    F=stats['F-bracket']['token_ms']['mean'];C=stats['C']['token_ms']['mean']
    S=stats['S']['token_ms']['mean'];P=stats['P']['token_ms']['mean']
    paired=[]
    for rep in range(6):
        bb={b['arm']:b for b in time_blocks if b['process_arm']=='CSP' and b['repetition']==rep}
        assert set(bb)=={'C','S','P'}
        ordered=sorted(bb,key=lambda x:bb[x]['start_ns'])
        paired.append({'repetition':rep,'order':''.join(ordered),
                       'C_minus_P_ms':bb['C']['mean_ms']-bb['P']['mean_ms'],
                       'S_minus_P_ms':bb['S']['mean_ms']-bb['P']['mean_ms'],
                       'S_over_P_speed':bb['S']['mean_ms']/bb['P']['mean_ms']})
    f=qualities['F-before'];p=qualities['P']
    assert f.tobytes()==qualities['F-after'].tobytes()==qualities['C'].tobytes()
    assert p.tobytes()==qualities['S'].tobytes()
    row_details=[]
    for r,(ff,pp) in enumerate(zip(f,p)):
        ff=ff.astype(np.float64);pp=pp.astype(np.float64)
        lf=ff-(ff.max()+np.log(np.exp(ff-ff.max()).sum()))
        lp=pp-(pp.max()+np.log(np.exp(pp-pp.max()).sum()))
        pf=np.exp(lf);qp=np.exp(lp)
        df=pp-ff
        row_details.append({'row':r,'input_position':271+r,'target_id':int(targets[r]),
                            'F_top1':int(np.argmax(ff)),'P_top1':int(np.argmax(pp)),
                            'forced_target_is_F_top1':int(targets[r])==int(np.argmax(ff)),
                            'F_NLL':float(-lf[targets[r]]),'P_NLL':float(-lp[targets[r]]),
                            'NLL_delta':float(lf[targets[r]]-lp[targets[r]]),
                            'KL_F_to_P_nats':float(np.sum(pf*(lf-lp))),
                            'KL_P_to_F_nats':float(np.sum(qp*(lp-lf))),
                            'total_variation':float(.5*np.sum(np.abs(pf-qp))),
                            'top1_probability_F':float(pf.max()),'top1_probability_P':float(qp.max()),
                            'mean_logit_shift':float(df.mean()),'rms_centered_logit_difference':float(np.sqrt(np.mean((df-df.mean())**2)))})
    nlld=np.asarray([r['NLL_delta'] for r in row_details])
    assert np.isclose(nlld.mean(),expected['quality_comparisons']['P_vs_F-before']['mean_reference_token_NLL_delta'],atol=1e-12,rtol=0)
    # Per-parity summaries address only this caller, not HTTP/SSE/UI cadence.
    parity={}
    for arm,v in sample_groups.items():
        m=np.stack(v)
        even=m[:,::2].mean();odd=m[:,1::2].mean()
        parity[arm]={'even_call_mean_ms':float(even),'odd_call_mean_ms':float(odd),
                     'even_minus_odd_ms':float(even-odd),
                     'median_ms_by_token_position':np.median(m,axis=0).tolist(),
                     'scope':'64 teacher-forced token indices, no SSE/UI observation; not a periodicity test'}
    result={'source_commit':PIN,'sources_sha256':sha,'trajectory_count':1,
            'process_count':3,'timed_calls':sum(len(v)*64 for k,v in sample_groups.items() if k!='F-bracket'),
            'quality_rows_per_arm':65,'quality_arms':list(qualities),
            'stats':stats,'blocks':time_blocks,'paired_CSP_blocks':paired,
            'descriptive_accounting':{'C_minus_F_ms':C-F,'C_minus_S_ms':C-S,'S_minus_P_ms':S-P,
                                      'F_minus_P_ms':F-P,'P_speed_ratio_vs_F':F/P,'P_speed_ratio_vs_S':S/P,
                                      'P_speed_ratio_vs_F_before':stats['F-before']['token_ms']['mean']/P,
                                      'P_speed_ratio_vs_F_after':stats['F-after']['token_ms']['mean']/P,
                                      'C_F_gap_per_custom_layer_us':(C-F)*1000/23,
                                      'warning':'Differences between measured modes; not attribution to pure copy, dispatch or math. F and CSP are different processes.'},
            'hypothetical_9_8_target_on_THIS_workload':{'target_ms':1000/9.8,
                 'extra_ms_to_remove':P-1000/9.8,'time_reduction_fraction':1-1000/(9.8*P),
                 'speed_increase_fraction':9.8/(1000/P)-1,
                 'if_entire_C_F_gap_removed_ms':P-(C-F),
                 'if_entire_C_F_gap_removed_t1_per_s':1000/(P-(C-F)),
                 'assumption':'Pure arithmetic scenario, not a prediction; old9.8 measurement is not comparable without matched workload.'},
            'quality_extra':{'F_C_byte_equal':True,'S_P_byte_equal':True,
                'F_top1_equals_P_top1_all_rows':all(r['F_top1']==r['P_top1'] for r in row_details),
                'forced_target_equals_F_top1_rows':sum(r['forced_target_is_F_top1'] for r in row_details),
                'prefix_forced_target_equals_top1':row_details[0]['forced_target_is_F_top1'],
                'NLL_delta':summary(nlld),'KL_F_to_P_nats':summary([r['KL_F_to_P_nats'] for r in row_details]),
                'total_variation':summary([r['total_variation'] for r in row_details]),
                'largest_absolute_NLL_deltas':sorted(row_details,key=lambda r:-abs(r['NLL_delta']))[:6],
                'all_rows':row_details,
                'warning':'These distributions live on a forced reference trajectory. They do not establish free generated answer quality or greedy conversation identity.'},
            'caller_parity_description':parity,
            'limits':['No independent prompts or hardware replication added.',
                      'No inference sampling confidence interval; repeated trajectory calls are dependent.',
                      'Performance arm did not export all logits; byte comparisons apply to quality arm.',
                      'No engine/API/Open WebUI measurements by this review.']}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'TIMING_AND_DISTRIBUTION_RECOMPUTED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stats':{k:{'mean_ms':v['token_ms']['mean'],'t1_per_s':v['t1_per_second']} for k,v in stats.items()},
                      'paired_CSP':paired,'accounting':result['descriptive_accounting'],
                      'hypothetical_target':result['hypothetical_9_8_target_on_THIS_workload'],
                      'quality':{k:result['quality_extra'][k] for k in ['forced_target_equals_F_top1_rows','prefix_forced_target_equals_top1','NLL_delta','KL_F_to_P_nats','total_variation']}},indent=2))


if __name__=='__main__':main()
