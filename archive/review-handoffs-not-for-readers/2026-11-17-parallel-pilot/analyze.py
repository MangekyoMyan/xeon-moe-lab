"""Read-only numerical checks and paired summaries of the bounded operator pilot."""
from pathlib import Path
import argparse, hashlib, json, math, statistics, struct

R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def check_run(p):
    h=read(p/'health.json');assert h['status']=='exited' and h['child_returncode']==0
    assert read(p/'RUN.json')['passed']
    m=read(p/'RESULTS_SHA256.json')
    for n,d in m.items():assert sha(p/n)==d,n
    return m
def f32(p):
    b=p.read_bytes();return struct.unpack('<'+'f'*(len(b)//4),b)
def capture_check(p,old_ja=None):
    m=check_run(p);cap=read(p/'CAPTURE.json');planned=read(R/'CAPTURE_CELLS.json')['cells']
    j=read(p/'text_cache_budget_ja.json');assert j['prefix_first4_byte_equal_old_callback_free']
    assert j['prefix_tokens']==272 and len(j['T1'])==9
    assert len(cap['cells'])==len(planned)==8
    baseline=read(R/'capture/BASELINE_PIN.json');old=read(old_ja if old_ja else Path(baseline['directory'])/'text_cache_budget_ja.json')
    actual=j['T1_trace']['records'];prior=old['T1_trace']['records'][:333];assert len(actual)==len(prior)==333
    for a,b in zip(actual,prior):
        for k in ['ids','layer','decode_call','position','generation','context','flags']:assert a[k]==b[k],(k,a,b)
    for actual,want in zip(cap['cells'],planned):
        assert actual['cell']==want['cell'] and actual['layer']==want['layer']
        assert actual['ids']==want['ordered_ids'] and actual['call']==want['decode_call']
        tr=actual['route_record'];assert tr['position']==want['position'] and tr['generation']==want['observed_generation'] and tr['context']==want['context_label_in_source_session']
        assert actual['seen']==7 and actual['flags']==0
        assert [x['type'] for x in actual['projections']]==want['projection_types']
        for k,n in [('x_file',5120),('down_file',5120*6),('routing_weights_file',6)]:
            a=f32(p/actual[k]);assert len(a)==n and all(math.isfinite(v) for v in a)
    return dict(passed=True,run=str(p),cells=8,outputs_prefix_first4_byte_equal=True,
        all_planned_route_labels_match=True,all_first9_333_routes_equal=True,callback_GPU_weight_readbacks=sum(c['routing_weight_readback'] for c in cap['cells']),
        callback_weight_readback_bytes=sum(c['routing_weight_readback'] for c in cap['cells'])*24,
        source_call5_8_intermediate_callback_free_identity='not proven',files=m)
def quantiles(v):
    v=sorted(v)
    def q(p):
        at=(len(v)-1)*p;i=int(at);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(at-i)
    return dict(count=len(v),mean=statistics.mean(v),median=statistics.median(v),min=v[0],p10=q(.1),p90=q(.9),max=v[-1])
def replay_check(p):
    check_run(p);j=read(p/'REPLAY.json');assert j['passed'] and len(j['cells'])==8
    assert len(j['private_initialized_layers'])==23 and len(j['empty_layers_bypassed'])==14
    assert j['bank_native_bytes']==j['bank_native_bytes_verified']==475453440
    result=dict(passed=True,run=str(p),cells=[],whole_model_tok_s=None,GPU_ready=None,
        scope='8 fixed actual activations from one JA case. Repetitions are not independent inputs. Instrumented operator, not main scheduler.',
        memory_phases=read(p/'RUN.json')['memory_phases'])
    expected_orders=['OCPS','CSOP','SPCO','POSC']*3
    for c in j['cells']:
        ci=c['cell'];v={a:f32(p/f'cell{ci}-{a}.f32') for a in 'OCSP'}
        assert (p/f'cell{ci}-O.f32').read_bytes()==(p/f'cell{ci}-C.f32').read_bytes()
        assert (p/f'cell{ci}-S.f32').read_bytes()==(p/f'cell{ci}-P.f32').read_bytes()
        assert all(len(x)==5120*6 and all(math.isfinite(z) for z in x) for x in v.values())
        samples=[s for s in c['samples'] if not s['first_use']];assert len(samples)==48
        values={a:[] for a in 'OCSP'};paired={k:[] for k in ['C_minus_O_ms','S_minus_P_ms','O_minus_P_ms','O_over_P']}
        for b in range(12):
            ss=[s for s in samples if s['block']==b];assert ''.join(x['mode'] for x in ss)==expected_orders[b]
            d={s['mode']:(s['output_ready_ns']-s['input_ready_ns'])/1e6 for s in ss}
            assert all(x>0 for x in d.values())
            for a in values:values[a].append(d[a])
            paired['C_minus_O_ms'].append(d['C']-d['O']);paired['S_minus_P_ms'].append(d['S']-d['P']);paired['O_minus_P_ms'].append(d['O']-d['P']);paired['O_over_P'].append(d['O']/d['P'])
            for s in ss:
                if s['executed_gpu_H']==0:assert s['gpu_operations']==0 and s['join_begin_ns']==s['join_end_ns']==0
                else:assert s['gpu_operations']==5 and s['join_end_ns']>=s['join_begin_ns']>0
        result['cells'].append(dict(cell=ci,layer=c['layer'],H=c['matched_H'],ids=c['ids'],gpu_slots=c['gpu_slots'],
            ms={k:quantiles(v) for k,v in values.items()},paired={k:dict(summary=quantiles(v),raw=v) for k,v in paired.items()},
            correctness=c['correctness'],reference=c.get('reference'),compact=c['compact']))
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capture','replay']);ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--old-ja',type=Path);a=ap.parse_args()
    assert not a.output.exists(),'preserve previous analysis'
    x=capture_check(a.run,a.old_ja) if a.mode=='capture' else replay_check(a.run)
    a.output.write_text(json.dumps(x,indent=2)+'\n')
    print('Verified '+str(a.output))
    if a.mode=='replay':
        for c in x['cells']:print(c['cell'],c['layer'],c['H'],{k:round(z['median'],4) for k,z in c['ms'].items()},'O/P',round(c['paired']['O_over_P']['summary']['median'],4))
if __name__=='__main__':main()
