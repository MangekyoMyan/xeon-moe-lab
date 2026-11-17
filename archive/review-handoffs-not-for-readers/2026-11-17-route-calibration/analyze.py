"""Analyze completed study runs offline, keeping every block and first T1."""
from pathlib import Path
from collections import Counter
import argparse
import array
import hashlib
import json
import math
import statistics

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(xs):
    xs=sorted(xs)
    k=(len(xs)-1)*.95;i=int(k)
    return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),
                p95=xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(k-i),minimum=xs[0],maximum=xs[-1])

def trace_check(t,n,observing):
    s=t['stats'];assert s['calls']==s['attempts']==n and s['flags']==s['dropped_events']==0 and not s['enabled']
    assert s['observing']==observing and s['records']==(37*n if observing else 0)
    assert s['complete_decode_attempts']==(n if observing else 0)
    assert len(t['records'])==(37*n if observing else 0)
    assert len(t['attempts'])==len(t['calls'])==n
    for i,(a,c) in enumerate(zip(t['attempts'],t['calls'])):
        label=dict(request=0,sequence=0,decode_call=i,phase=2,logical_input_token=i)
        assert a['label']==c['label']==label
        assert a['id']==i+1 and a['position']==64+i and a['context']==1 and a['ubatch']==0 and a['n_tokens']==1
        assert a['observing']==observing and a['scope_t1'] and a['graph_returned'] and a['caller_completed']
        assert a['flags']==a['graph_status']==c['flags']==c['public_or_driver_status']==0
        assert a['records']==(37 if observing else 0) and a['first_record']==(37*i if observing else 0)
        assert a['layers']==((1<<37)-1 if observing else 0)
        assert c['first_attempt']==i and c['attempts']==1 and c['completed'] and c['synchronized']
        rows=t['records'][37*i:37*(i+1)] if observing else []
        if observing:assert sorted(r['layer'] for r in rows)==list(range(37))
        for j,r in enumerate(rows,37*i):
            expect=dict(ordinal=j,attempt_id=i+1,request=0,sequence=0,decode_call=i,ubatch=0,
                        logical_input_token=i,position=64+i,generation=a['generation'],context=1,
                        phase=2,projection=1,n_tokens=1,flags=0)
            for k,v in expect.items():assert r[k]==v,(i,k)
            assert len(r['ids'])==len(set(r['ids']))==6 and all(type(x)is int and 0<=x<384 for x in r['ids'])

def one(root):
    d=json.loads((root/'RUN.json').read_text());h=json.loads((root/'health.json').read_text())
    assert d['passed'] and h['status']=='exited' and h['child_returncode']==0
    assert d['session_closed_before_owner_destruction'] and not (root/'STOP_REQUESTED').exists()
    for name,digest in json.loads((root/'RESULTS_SHA256.json').read_text()).items():assert sha(root/name)==digest,name
    pin=json.loads((root/'PROVIDER_EXPECTED.json').read_text());assert pin['providers']==d['providers'] and pin['active_backend_providers']==d['active_backend_providers']
    out=dict(run=root.name,arm=d['arm'],cparams=d['cparams'],memory_after_warmup=d['memory_after_warmup'],blocks=[],cases=[],limits='Raw text first128-token screen, not equal semantic units or sampled answers; timings include first T1.')
    for c in d['cases']:
        token_bytes=array.array('i',c['full_tokens']).tobytes()
        out['cases'].append(dict(case_id=c['case_id'],text_sha256=c['text_sha256'],full_token_count=c['full_token_count'],token_ids_native_i32_sha256=hashlib.sha256(token_bytes).hexdigest()))
    keys=['step_begin_ns','decode_begin_ns','decode_return_ns','output_ready_ns','validation_end_ns','finalize_end_ns','step_end_ns']
    for meta in d['blocks']:
        b=json.loads((root/meta['result_file']).read_text());steps=b['steps'];n=b['calls'];assert len(steps)==n
        spans=dict(engine_us=[],label_prepare_us=[],decode_return_us=[],output_sync_us=[],validation_us=[],finalize_us=[],post_us=[],step_us=[],next_step_interval_us=[])
        for i,s in enumerate(steps):
            assert s['ok'] and s['synchronized'] and s['decode_status']==0 and s['index']==i and s['position']==64+i
            times=[s[k] for k in keys];assert times==sorted(times)
            spans['engine_us'].append((times[3]-times[1])/1000)
            for key,a,z in [('label_prepare_us',0,1),('decode_return_us',1,2),('output_sync_us',2,3),('validation_us',3,4),('finalize_us',4,5),('post_us',5,6),('step_us',0,6)]:spans[key].append((times[z]-times[a])/1000)
            if i+1<n:
                assert steps[i+1]['step_begin_ns']>=times[-1]
                spans['next_step_interval_us'].append((steps[i+1]['step_begin_ns']-times[0])/1000)
        item=dict(name=b['name'],mode=b['mode'],case_id=b['case_id'],numerical=meta['numerical'],calls=n,
                  interval_statistics={k:summary(v) for k,v in spans.items() if v},
                  engine_each_us=spans['engine_us'],first_engine_us=spans['engine_us'][0],
                  engine_after_first_us=summary(spans['engine_us'][1:]),
                  block_wall_us=(b['block_end_ns']-steps[0]['step_begin_ns'])/1000,
                  prefill_us=(b['prefill_ready_ns']-b['prefill_begin_ns'])/1000,
                  setup_us=(b['session_setup_end_ns']-b['session_setup_begin_ns'])/1000,
                  output_validation=b['output_validation'])
        if b['collector_session']:
            trace_check(b['trace'],n,b['observing'])
            item['records']=len(b['trace']['records'])
            if b['observing']:
                item['layer_unique_experts']=[len({x for r in b['trace']['records'] if r['layer']==layer for x in r['ids']}) for layer in range(37)]
        if meta['numerical']:
            p=root/b['logits_file'];vals=array.array('f');vals.frombytes(p.read_bytes());assert len(vals)==n*d['vocab_size'] and all(math.isfinite(x) for x in vals)
            item['logits_sha256']=sha(p);item['logits_scalars']=len(vals);item['frozen_reference_byte_equal']=b.get('frozen_reference_byte_equal')
        out['blocks'].append(item)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path,nargs='+');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'Preserve prior analysis'
    result=dict(schema=1,analysis_source_sha256=sha(Path(__file__)),runs=[one(r) for r in a.runs])
    numeric=[b for r in result['runs'] for b in r['blocks'] if b['numerical']]
    result['numeric_all_byte_equal']=len({b['logits_sha256'] for b in numeric})==1
    timings=[b for r in result['runs'] for b in r['blocks'] if b['name'].startswith('timing_')]
    result['timing_block_means_us']={mode:[b['interval_statistics']['engine_us']['mean'] for b in timings if b['mode']==mode] for mode in ['F','D0','D1','D2']}
    result['limits']=['F/D are sequential processes, D modes reverse order in one process; no isolated single-instruction causality.',
                      'No after-the-fact performance acceptance threshold; compare process/block variability.',
                      'No CPU/GPU hot bank, quality or long-context workload measured.']
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs']},indent=2))
if __name__=='__main__':main()
