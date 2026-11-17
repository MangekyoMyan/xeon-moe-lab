"""Offline analysis; stored logits are teacher-forced diagnostics, not answer scores."""
from pathlib import Path
import argparse,array,hashlib,json,math,statistics,sys
from analyze import check_run,quantiles

NV=129280
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def logits(path):
    b=path.read_bytes();assert len(b)==65*NV*4
    v=array.array('f');v.frombytes(b)
    if sys.byteorder!='little':v.byteswap()
    assert all(math.isfinite(x) for x in v)
    return b,v
def row_diagnostics(raw,values,targets):
    rows=[]
    for i,target in enumerate(targets):
        v=values[i*NV:(i+1)*NV];best=-math.inf;second=-math.inf;top=-1
        for k,x in enumerate(v):
            if x>best:second=best;best=x;top=k
            elif x>second:second=x
        nll=math.log(math.fsum(math.exp(x-best) for x in v))+best-v[target]
        rows.append(dict(row=i,next_target=target,sha256=sha(raw[i*NV*4:(i+1)*NV*4]),top1=top,margin=best-second,target_logit=v[target],NLL=nll))
    return rows
def comparison(ar,av,br,bv,adiag,bdiag):
    exact=ar==br;norm=math.fsum(float(x)*x for x in bv)
    err=0 if exact else math.fsum((float(x)-y)**2 for x,y in zip(av,bv))
    maxabs=0 if exact else max(abs(float(x)-y) for x,y in zip(av,bv))
    changed=[i for i,(x,y) in enumerate(zip(adiag,bdiag)) if x['sha256']!=y['sha256']]
    top_changes=[i for i,(x,y) in enumerate(zip(adiag,bdiag)) if x['top1']!=y['top1']]
    return dict(byte_equal=exact,rows_changed=changed,first_changed_row=changed[0] if changed else None,
        top1_changed_rows=top_changes,NMSE=err/norm if norm else None,max_abs=maxabs,
        mean_reference_token_NLL_delta=statistics.mean(x['NLL']-y['NLL'] for x,y in zip(adiag,bdiag)),
        row_NLL_delta=[x['NLL']-y['NLL'] for x,y in zip(adiag,bdiag)])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('before',type=Path);ap.add_argument('mixed',type=Path);ap.add_argument('after',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    runs=[a.before,a.mixed,a.after];docs=[]
    manifests=[];launches=[]
    for p,arm in zip(runs,['F-before','CSP','F-after']):
        manifests.append(check_run(p));launches.append(read(p/'LAUNCH_MANIFEST.json'));j=read(p/'RUN.json');assert j['arm']==arm;docs.append(j)
        for b in j['blocks']:
            assert b['prefix_tokens']==272
            assert all(len(b[k])==64 for k in ['start_ns','output_ready_ns','status','input_tokens'])
            assert len(b['next_targets'])==65 and all(0<=x<NV for x in b['next_targets'])
            assert all(z==0 for z in b['status'])
            assert all(e>s for s,e in zip(b['start_ns'],b['output_ready_ns']))
            assert all(b['start_ns'][i+1]>=b['output_ready_ns'][i] for i in range(63))
            tok=j['cases'][2]['full_tokens'];assert b['input_tokens']==tok[272:336] and b['next_targets']==tok[272:337]
        if arm=='CSP':
            assert [(b['mode'],b['quality']) for b in j['blocks'][:3]]==[(x,True) for x in 'CSP']
            expected=[(m,k) for k,order in enumerate(['CSP','PSC','SPC','CPS','PCS','SCP']) for m in order]
            assert [(b['mode'],b['repetition']) for b in j['blocks'][3:]]==expected and all(not b['quality'] for b in j['blocks'][3:])
        else:assert [(b['mode'],b['quality'],b['repetition']) for b in j['blocks']]==[('F',True,0)]+[('F',False,k) for k in range(3)]
    for k in ['cases','cparams','placement','native_expert_inventory','providers','requested_ubatch','vision_weights_loaded']:
        assert docs[0][k]==docs[1][k]==docs[2][k],k
    assert docs[0]['active_backend_providers']==docs[2]['active_backend_providers']
    assert [v for v in docs[1]['active_backend_providers'] if v['name']!='XEON_HOT_FFN']==docs[0]['active_backend_providers']
    assert [v['name'] for v in docs[1]['active_backend_providers']]==['CUDA0','XEON_HOT_FFN','CPU']
    for k in ['build_sha256','baseline_config_sha256','model_manifest_sha256','guard_source_sha256','bank_plan_sha256','engine_build_sha256','reserve_GiB','budget_seconds','runner_sha256','ubatch']:
        assert launches[0][k]==launches[1][k]==launches[2][k],k
    assert docs[0]['F_prefix_first4_equal_old'] and docs[2]['F_prefix_first4_equal_old'] and docs[1]['quality_S_P_byte_equal']
    libs=[read(p/'BUILD.json')['engine_libraries'] for p in runs];assert libs[0]==libs[1]==libs[2]
    outputs={};targets=None;quality={};timing={};hist={}
    for p,j,manifest in zip(runs,docs,manifests):
        for b in j['blocks']:
            label=j['arm'] if b['mode']=='F' else b['mode']
            if targets is None:targets=b['next_targets']
            assert b['next_targets']==targets
            if b['quality']:
                assert Path(b['file']).name==b['file'] and b['file'] in manifest
                raw,v=logits(p/b['file']);diag=row_diagnostics(raw,v,targets);outputs[label]=(raw,v);quality[label]=diag
            else:
                wall=[(e-s)/1e6 for s,e in zip(b['start_ns'],b['output_ready_ns'])];assert all(x>0 for x in wall)
                timing.setdefault(label,[]).append(dict(repetition=b['repetition'],ms=wall,total_ms=sum(wall),mean_ms=statistics.mean(wall),tokens_per_second=len(wall)*1000/sum(wall)))
            if b['mode']!='F':
                delta=[[bb-aa for aa,bb in zip(x,y)] for x,y in zip(b['before']['H'],b['after']['H'])]
                calls=[bb-aa for aa,bb in zip(b['before']['calls'],b['after']['calls'])]
                assert all(sum(d)==n for d,n in zip(delta,calls)) and sum(calls)==23*64
                assert all(n in (0,64) for n in calls)
                if b['mode']=='C':assert all(all(v==0 for v in d[1:]) for d in delta)
                hist.setdefault(b['mode'],[]).append(dict(quality=b['quality'],repetition=b['repetition'],executed_H=delta,calls=calls,custom_nodes_built=b['after']['builds']-b['before']['builds']))
    old=read(Path(__file__).resolve().parent/'integrated/BASELINE_PIN.json');assert len(outputs['F-before'][0][:5*NV*4])==2585600
    assert sha(outputs['F-before'][0][:5*NV*4])==old['files']['text_cache_budget_ja.f32']
    assert sha(outputs['F-after'][0][:5*NV*4])==old['files']['text_cache_budget_ja.f32']
    assert len(timing['F-before'])==len(timing['F-after'])==3
    assert all(len(timing[m])==6 for m in 'CSP')
    summaries={}
    for m,blocks in timing.items():
        all_ms=[v for b in blocks for v in b['ms']]
        summaries[m]=dict(total_tokens=len(all_ms),total_ms=sum(all_ms),tokens_per_second=1000*len(all_ms)/sum(all_ms),token_ms=quantiles(all_ms),block_mean_ms=quantiles([b['mean_ms'] for b in blocks]))
    ftotal=summaries['F-before']['total_ms']+summaries['F-after']['total_ms'];ftokens=summaries['F-before']['total_tokens']+summaries['F-after']['total_tokens'];fmean=ftotal/ftokens
    cmp={}
    for left,right in [('F-after','F-before'),('C','F-before'),('S','F-before'),('P','F-before'),('S','P')]:
        cmp[left+'_vs_'+right]=comparison(*outputs[left],*outputs[right],quality[left],quality[right])
    assert cmp['S_vs_P']['byte_equal'],'S/P exact output gate'
    result=dict(passed=True,runs=list(map(str,runs)),quality=quality,quality_comparisons=cmp,timing_raw=timing,timing_summary=summaries,
        F_bracket=dict(mean_ms=fmean,tokens_per_second=1000/fmean,total_tokens=ftokens),
        speed_vs_F_bracket={m:fmean/(summaries[m]['total_ms']/summaries[m]['total_tokens']) for m in 'CSP'},
        execution_histograms=hist,all_S_P_aggregate_H_equal=all(x['executed_H']==y['executed_H'] for x,y in zip(hist['S'],hist['P'])),
        scope='Single-sequence teacher-forced T1/s, excludes sampling/stream/API. One fixed JA teacher-forced trajectory. Operator and full-T1 metrics are separate. H is executed GPU count; no logical route trace. No general answer quality/adoption claim.',
        routing_origin_fix_implemented=False,callback=None,route_collector='closed',memory_phases={j['arm']:j['memory_phases'] for j in docs})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),tokens_per_second={k:v['tokens_per_second'] for k,v in summaries.items()},F_bracket=result['F_bracket'],speed=result['speed_vs_F_bracket'],quality={k:{q:v[q] for q in ['byte_equal','first_changed_row','top1_changed_rows','mean_reference_token_NLL_delta']} for k,v in cmp.items()})))
if __name__=='__main__':main()
