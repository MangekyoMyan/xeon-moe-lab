"""Strict saved-evidence checks; no model or API calls."""
from pathlib import Path
import array,hashlib,json,sys
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    assert not p.exists(),'preserve prior receipt'
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load_run(name):
    state=read(R/'STATE.json');e=state['runs'][name];assert e['status']=='passed'
    p=Path(e['path']);assert sha(p/'RESULTS_SHA256.json')==e['manifest_sha256']
    m=read(p/'RESULTS_SHA256.json')
    for n,h in m.items():
        assert Path(n).name==n and sha(p/n)==h,n
    h=read(p/'health.json');assert h['status']=='exited' and h['child_returncode']==0
    j=read(p/'RUN.json');assert j['passed']
    launch=read(p/'LAUNCH_MANIFEST.json');build=read(p/'BUILD.json')
    assert launch['plan_sha256']==sha(R/'caller/FROZEN_PLAN.json') and launch['bank_plan_sha256']==sha(R/'caller/capacity.plan')
    assert launch['build_sha256']==sha(R/'caller/BUILD.json') and build==read(R/'caller/BUILD.json')
    assert launch['spec']==j['spec']==read(p/'RUN_SPEC.json') and launch['run_spec_sha256']==sha(p/'RUN_SPEC.json')
    spec=next(x for x in read(R/'caller/DESIGN.json')['stages'] if x['name']==name);assert spec==j['spec']
    assert j['stage']==spec['stage'] and j['arm']==spec['mode']
    assert j['providers']==read(p/'PROVIDER_EXPECTED.json')['providers'] and j['active_backend_providers']==read(p/'PROVIDER_EXPECTED.json')['active_backend_providers']
    assert j['cparams']['n_ctx']==131072 and j['cparams']['n_ubatch']==1024 and j['cparams']['n_outputs_max']==1
    assert j['cparams']['threads']==12 and j['cparams']['threads_batch']==24
    assert j['placement']==[dict(layer=i,host=i<37) for i in range(40)]
    plan=read(R/'caller/FROZEN_PLAN.json');cases={c['case_id']:c for c in plan['cases']}
    assert [c['case_id'] for c in j['cases']]==list(cases)
    for c in j['cases']:
        frozen=cases[c['case_id']]['execution_input'];assert c['prefix_ids']==frozen['prefix_ids'] and c['next_targets65']==frozen['teacher_forced_next_targets65']
    pr=j['preparation'];ja=cases['text_plan_constraints_ja']['execution_input'];assert pr['count']==4 and pr['timed'] is False and pr['prefix_ids']==ja['prefix_ids'] and pr['T1_inputs']==ja['teacher_forced_input64'][:4]
    expected_layers={int(line.split()[0]) for line in (R/'caller/capacity.plan').read_text().splitlines()[1:] if int(line.split()[4])>0};assert len(expected_layers)==23
    def counters(a,b,n):
        if spec['mode']=='F':assert a is None and b is None;return
        assert a['native_bytes']==b['native_bytes']==475453440
        for l in range(40):
            d=b['calls'][l]-a['calls'][l];assert d==(n if l in expected_layers else 0),(l,d,n)
            hist=[v-u for u,v in zip(a['executed_H'][l],b['executed_H'][l])];assert len(hist)==7 and min(hist)>=0 and sum(hist)==d
    counters(pr['before'],pr['after'],4)
    for b in j['blocks']:
        c=cases[b['case_id']]['execution_input'];assert b['prefix_ids']==c['prefix_ids'] and b['prefix_count']==c['prefix_count']
        assert b['input_tokens']==c['teacher_forced_input64'] and b['next_targets']==c['teacher_forced_next_targets65'] and b['positions']==c['input_absolute_positions64']
        assert len(b['next_targets'])==65 and all(type(t)==int and 0<=t<129280 for t in b['next_targets'])
        assert all(len(b[k])==64 for k in ['start_ns','output_ready_ns','status','input_tokens','positions'])
        assert all(v==0 for v in b['status']) and all(t>s for s,t in zip(b['start_ns'],b['output_ready_ns']))
        assert all(b['start_ns'][i+1]>=b['output_ready_ns'][i] for i in range(63))
        counters(b['before_prefix'],b['before_T1'],0);counters(b['before_T1'],b['after'],64)
        if b['quality']:
            assert b['file'] in m and b['file']==f"quality-{b['case_id']}-{b['mode']}.f32"
            assert (p/b['file']).stat().st_size==65*129280*4 and b['shape']==[65,129280] and b['all_finite']
        else:assert b['custom_builds_after_each_T1'] is None
    if spec['stage']=='quality':
        expected=[(ci,mode,True) for ci in spec['cases'] for mode in (['F'] if spec['mode']=='F' else ['S','P'])]
        assert [(b['case_id'],b['mode'],b['quality']) for b in j['blocks']]==expected and not j['answers']
    elif spec['stage']=='timing':assert [(b['case_id'],b['mode'],b['quality']) for b in j['blocks']]==[(ci,spec['mode'],False) for ci in spec['cases']] and not j['answers']
    else:
        assert not j['blocks'] and [a['case_id'] for a in j['answers']]==spec['cases']
        for a in j['answers']:counters(a['before_prefix'],a['before_T1'],0);counters(a['before_T1'],a['after'],a['decoded_T1_calls'])
    return p,j,m,launch
def common(runs):
    for k in ['cases','cparams','placement','native_expert_inventory','providers','requested_ubatch','vision_weights_loaded']:
        assert all(x[1][k]==runs[0][1][k] for x in runs),k
    for k in ['build_sha256','baseline_config_sha256','model_manifest_sha256','guard_source_sha256','bank_plan_sha256','runner_sha256','reserve_GiB','budget_seconds']:
        assert all(x[3][k]==runs[0][3][k] for x in runs),k
def run_receipt(x):return dict(path=str(x[0]),manifest_sha256=sha(x[0]/'RESULTS_SHA256.json'))
def quality_gate():
    import math
    f=load_run('quality-F');p=load_run('quality-SP');common([f,p]);checked=[]
    fd={b['case_id']:b for b in f[1]['blocks']};pd={(b['case_id'],b['mode']):b for b in p[1]['blocks']}
    for ci in f[1]['spec']['cases']:
        hashes={}
        for mode,x,b in [('F',f,fd[ci]),('S',p,pd[ci,'S']),('P',p,pd[ci,'P'])]:
            data=(x[0]/b['file']).read_bytes();a=array.array('f');a.frombytes(data)
            if sys.byteorder!='little':a.byteswap()
            assert len(a)==65*129280 and all(math.isfinite(v) for v in a)
            hashes[mode]=hashlib.sha256(data).hexdigest()
        assert hashes['S']==hashes['P'],ci
        checked.append(dict(case_id=ci,sha256=hashes,S_P_equal=True,finite=True,shape=[65,129280]))
    result=dict(passed=True,plan_sha256=sha(R/'caller/FROZEN_PLAN.json'),runs={'F':run_receipt(f),'SP':run_receipt(p)},cases=checked)
    write(R/'QUALITY_GATE.json',result);print('Quality gate passed for all3cases',flush=True)
def timing_gate():
    names=['timing-0-F','timing-1-P','timing-2-P','timing-3-F'];runs=[load_run(n) for n in names];common(runs)
    assert sum(len(x[1]['blocks'])*64 for x in runs)==768
    write(R/'TIMING_COMPLETE.json',dict(passed=True,plan_sha256=sha(R/'caller/FROZEN_PLAN.json'),runs=[run_receipt(x) for x in runs],timed_calls=768));print('All768timedT1 saved',flush=True)
if __name__=='__main__':
    if sys.argv[1]=='quality':quality_gate()
    elif sys.argv[1]=='timing':timing_gate()
    else:raise SystemExit('quality or timing')
