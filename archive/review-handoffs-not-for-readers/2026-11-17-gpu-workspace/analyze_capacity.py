"""Offline capacity/ledger/logit checks; no new acceptance tolerance is invented."""
from pathlib import Path
import argparse,array,csv,hashlib,heapq,json,math,re
GPU='GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_case(b,nv):
    p=b['prefix_tokens'];pre=b['prefill_trace'];tg=b['T1_trace']
    for t in [pre,tg]:
        s=t['stats'];assert not s['enabled'] and s['flags']==s['dropped_events']==0
        assert s['calls']==len(t['calls']) and s['attempts']==len(t['attempts']) and s['records']==len(t['records'])
        for c in t['calls']:assert c['flags']==c['public_or_driver_status']==0 and c['synchronized'] and c['completed']
        for a in t['attempts']:assert a['flags']==a['graph_status']==0 and a['graph_returned'] and a['caller_completed']
    assert pre['stats']['calls']==1 and pre['stats']['records']==pre['stats']['complete_decode_attempts']==0 and not pre['stats']['observing']
    c=pre['calls'][0];assert c['attempts']==len(pre['attempts']) and c['first_attempt']==0
    pos=0
    for i,a in enumerate(pre['attempts']):
        assert a['position']==pos and a['id']==i+1 and a['ubatch']==i and not a['observing'] and a['records']==a['layers']==0
        assert a['label']==c['label']==dict(request=0,sequence=0,decode_call=0,phase=1,logical_input_token=0)
        assert a['n_tokens']>0;pos+=a['n_tokens']
    assert pos==p and [a['n_tokens'] for a in pre['attempts']]==b['actual_ubatches']==b['expected_ubatches']
    assert tg['stats']['observing'] and tg['stats']['calls']==tg['stats']['attempts']==tg['stats']['complete_decode_attempts']==4 and tg['stats']['records']==148
    for i,(a,c,step) in enumerate(zip(tg['attempts'],tg['calls'],b['T1'])):
        label=dict(request=0,sequence=0,decode_call=i,phase=2,logical_input_token=i)
        assert a['label']==c['label']==label and a['position']==p+i and a['id']==i+1 and a['ubatch']==0 and a['n_tokens']==1
        assert a['records']==37 and a['first_record']==37*i and a['layers']==(1<<37)-1 and a['observing'] and a['scope_t1']
        assert c['first_attempt']==i and c['attempts']==1
        rows=tg['records'][37*i:37*(i+1)];assert sorted(r['layer'] for r in rows)==list(range(37))
        for j,r in enumerate(rows,37*i):
            for k,v in dict(ordinal=j,attempt_id=i+1,request=0,sequence=0,decode_call=i,ubatch=0,logical_input_token=i,position=p+i,generation=a['generation'],context=1,phase=2,projection=1,n_tokens=1,flags=0).items():assert r[k]==v,(k,r[k],v)
            assert len(r['ids'])==len(set(r['ids']))==6 and all(0<=x<384 for x in r['ids'])
        assert step['position']==p+i and step['synchronized'] and step['full_logits_finite'] and step['decode_status']==0
        assert step['boundaries_ns']==sorted(step['boundaries_ns'])
    assert b['logical_prefill_output_positions']==[p-1] and b['logits_shape']==[5,nv]

def one(root):
    d=json.loads((root/'RUN.json').read_text());h=json.loads((root/'health.json').read_text());launch=json.loads((root/'LAUNCH_MANIFEST.json').read_text());pin=json.loads((root/'PROVIDER_EXPECTED.json').read_text())
    assert d['passed'] and h['status']=='exited' and h['child_returncode']==0 and not (root/'STOP_REQUESTED').exists()
    assert d['session_closed_before_owner_destruction'] and d['providers']==pin['providers'] and d['active_backend_providers']==pin['active_backend_providers']
    assert launch['environment']['CUDA_VISIBLE_DEVICES']==GPU
    for name,digest in json.loads((root/'RESULTS_SHA256.json').read_text()).items():assert sha(root/name)==digest,name
    phases=[];previous=-1
    expected=['M0_after_context_before_vision','M1_after_vision_warmup','M2_after_text64_warmup','M3_after_long1024_and4T1']
    bank=bool(launch.get('bank'));bank_checks=None
    if bank:
        assert d['capacity_bank_initialized'] and d['bank_used_by_model'] is False and launch['bank_used_by_model'] is False
        expected.insert(1,'Gpost_bank_initialized_before_vision')
        raw=(root/'server.stderr.log').read_text(errors='replace')
        pre=re.findall(r'capacity-bank-Gpre free=(\d+) total=(\d+) payload=(\d+) margin=(\d+)',raw)
        post=re.findall(r'capacity-bank-Gpost free=(\d+) total=(\d+) margin=(\d+)',raw)
        ready=re.findall(r'xeon-hot-ready experts=(\d+) native_payload=(\d+) GPU_bank_allocation=(\d+) GPU_free_before_vision=(\d+)',raw)
        warm=re.findall(r'capacity-bank-warm layer=(\d+) H=(\d+) repeats=3 zero_output=1',raw)
        assert len(pre)==len(post)==len(ready)==1 and len(warm)==24
        assert set((int(l),int(h)) for l,h in warm)=={(l,h) for l in [0,4,23,24] for h in range(1,7)}
        free0,total0,payload,margin0=map(int,pre[0]);free1,total1,margin1=map(int,post[0]);count,payload1,allocation,free_ready=map(int,ready[0])
        assert margin0==3072*1048576 and margin1==2560*1048576 and free0>payload+margin0 and free1>margin1
        assert count==32 and payload1==payload==int(458.4375*1048576) and free_ready==free1
        bank_checks=dict(actual_Gpre_free_MiB=free0/1048576,actual_Gpost_free_MiB=free1/1048576,native_payload_MiB=payload/1048576,bank_buffer_allocation_MiB=allocation/1048576,extra_guard_delta_minus_bank_buffer_MiB=(free0-free1-allocation)/1048576,private_warmup_cells=24,private_warmup_graph_calls=72,all_zero_outputs_verified=True,model_uses_bank=False)
    assert [x['phase'] for x in d['memory_phases']]==expected
    for x in d['memory_phases']:
        cu=x['cuda'];nv=x['nvml_cli'];assert cu['begin_ns']>=previous and cu['end_ns']>=cu['begin_ns'] and nv['begin_ns']>=cu['end_ns'] and nv['end_ns']>=nv['begin_ns'];previous=nv['end_ns']
        assert nv['status']==0 and x['bank_allocated']==(bank and x['phase']!='M0_after_context_before_vision') and 0<=cu['free_bytes']<=cu['total_bytes']
        rows=list(csv.reader(nv['raw_csv'].strip().splitlines()));assert len(rows)==1 and len(rows[0])==6;row=[v.strip() for v in rows[0]]
        assert row[1]==GPU and row[2].upper()=='00000000:84:00.0'
        total,used,free=map(float,row[3:]);assert 0<=free<=total and 0<=used<=total
        phases.append(dict(phase=x['phase'],cuda_free_MiB=cu['free_bytes']/1048576,cuda_total_MiB=cu['total_bytes']/1048576,nvml_total_MiB=total,nvml_used_MiB=used,nvml_free_MiB=free,query_span_ms=(nv['end_ns']-cu['begin_ns'])/1e6))
    cases=[]
    for name in d['completed_cases']:
        b=json.loads((root/(name+'.json')).read_text());check_case(b,d['vocab_size']);vals=array.array('f');vals.frombytes((root/b['logits_file']).read_bytes());assert len(vals)==5*d['vocab_size'] and all(math.isfinite(v) for v in vals)
        item=dict(name=name,prefix_tokens=b['prefix_tokens'],actual_ubatches=b['actual_ubatches'],logits_sha256=sha(root/b['logits_file']),prefill_ms=(b['prefill_output_ready_ns']-b['prefill_decode_begin_ns'])/1e6,T1_engine_ms=[(s['boundaries_ns'][3]-s['boundaries_ns'][1])/1e6 for s in b['T1']],route_records=len(b['T1_trace']['records']),logits_scalars=len(vals))
        if name=='short64':assert b['T1_byte_equal_previous_F_before'] is True
        cases.append(item)
    minimum=min(x['native_triplet_bytes'] for x in d['native_expert_inventory'] if x['host'])
    m0=d['memory_phases'][0]['cuda']['free_bytes']
    return dict(run=root.name,ubatch=d['cparams']['n_ubatch'],cparams=d['cparams'],memory=phases,cases=cases,native_smallest_triplet_MiB=minimum/1048576,
                M0_payload_upper_bound_bytes_inference=max(0,m0-3072*1048576-1),actual_Gpre_pass=True if bank else None,actual_Gpost_pass=True if bank else None,bank_checks=bank_checks,
                scope='M0 is a bank-free corresponding point; strict payload bound is an inference, not admission. M1 is dummy vision reserve; snapshots are retained allocations, not peaks.'),d

def metrics(a,b,target):
    n=len(a);diff=[float(y)-float(x) for x,y in zip(a,b)];sq=sum(x*x for x in diff);norm=sum(float(x)*float(x) for x in a)
    topa=heapq.nlargest(2,range(n),key=a.__getitem__);topb=heapq.nlargest(2,range(n),key=b.__getitem__)
    def lp(v):
        m=max(v);return v[target]-m-math.log(sum(math.exp(x-m) for x in v))
    return dict(max_abs=max(map(abs,diff)),RMSE=math.sqrt(sq/n),relative_L2=math.sqrt(sq/norm) if norm else (0 if sq==0 else None),top2_baseline=topa,top2_candidate=topb,top1_margin_baseline=float(a[topa[0]]-a[topa[1]]),top1_margin_candidate=float(b[topb[0]]-b[topb[1]]),target_token=target,target_logprob_baseline=lp(a),target_logprob_candidate=lp(b))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path,nargs='+');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    all_results=[one(r) for r in a.runs];out=dict(schema=1,source_sha256=sha(Path(__file__)),runs=[x[0] for x in all_results])
    if len(all_results)==2:
        base,cand=all_results[0][1],all_results[1][1];assert base['cases']==cand['cases'] and base['providers']==cand['providers'] and base['placement']==cand['placement']
        assert {k:v for k,v in base['cparams'].items() if k!='n_ubatch'}=={k:v for k,v in cand['cparams'].items() if k!='n_ubatch'}
        comparisons=[]
        for name in base['completed_cases']:
            b=json.loads((a.runs[0]/(name+'.json')).read_text());c=json.loads((a.runs[1]/(name+'.json')).read_text());assert b['next_token_targets_for_rows']==c['next_token_targets_for_rows'] and b['continuation_inputs']==c['continuation_inputs']
            raw0=(a.runs[0]/b['logits_file']).read_bytes();raw1=(a.runs[1]/c['logits_file']).read_bytes();v0=array.array('f');v1=array.array('f');v0.frombytes(raw0);v1.frombytes(raw1);nv=base['vocab_size']
            comparisons.append(dict(name=name,byte_equal=raw0==raw1,rows=[metrics(v0[i*nv:(i+1)*nv],v1[i*nv:(i+1)*nv],b['next_token_targets_for_rows'][i]) for i in range(5)],quality_acceptance=None))
        out['comparisons']=comparisons
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out if len(all_results)==1 else {'runs':out['runs'],'comparison_file':str(a.output)},indent=2))
if __name__=='__main__':main()
