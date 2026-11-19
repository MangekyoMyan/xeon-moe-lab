"""Serialized original-provider diagnostic. Never changes normal API configuration."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,resource,subprocess,sys,time
R=Path(__file__).resolve().parent;M=R/'trial';LAB=R.parent;E=LAB/'deepseek41-bringup-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def prepare(stage):
    b=read(M/'BUILD.json');assert b['returncode']==0 and sha(M/'driver')==b['driver_sha256']
    reference=read(R/'BASELINE_REVERIFIED.json');assert reference['passed']
    for name,digest in reference['outputs'].items():assert sha(R/'main/runs/baseline'/name)==digest
    for n,h in b['source_sha256'].items():assert sha(M/n)==h,n
    for v in b['libraries'].values():assert sha(Path(v['path']))==v['sha256']
    b=json.loads(json.dumps(b))
    if stage['arm']=='B':
        engine=read(R/'prefix-engine/BUILD.json');arm=engine['arms']['candidate'];assert arm['result']==[0,0] and engine['arms']['control']['bit_identical_deployed']
        for p,h in engine['source_sha256'].items():assert sha(Path(p))==h,p
        for p,h in engine['original_link_inputs'].items():assert sha(Path(p))==h,p
        b['libraries']['libggml-base.so.0.28.0']=dict(path=arm['library'],sha256=arm['library_sha256'])
        assert sha(Path(arm['library']))==arm['library_sha256']

    p=module('original_prepare',LAB/'route37-integration-20261117/run_capture.py').prepare(False)
    env=p['environment'].copy();env['LD_LIBRARY_PATH']=str(M/'runtime')+':'+str(E/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib');env['LD_BIND_NOW']='1';env.pop('LD_PRELOAD',None)
    env['XEON_169_SHARED_PREFIX']='1' if stage['arm']=='B' else '0'
    if stage['arm']=='B':env['LD_LIBRARY_PATH']=str(M/'runtime-B')+':'+env['LD_LIBRARY_PATH']
    if stage['stage']=='observe':
        observer=read(R/'astra62/observer/BUILD.json');assert observer['returncode']==0
        lib=Path(observer['library']);assert sha(lib)==observer['library_sha256']
        for n,h in observer['source_sha256'].items():assert sha(Path(n))==h,n
        assert sha(Path(observer['cupti_path']))==observer['cupti_sha256']
        env['LD_PRELOAD']=str(lib)
    return b,p,env
def providers(b,env,out):
    clean={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))};clean.update(env)
    subprocess.run([str(M/'driver'),'--check-providers',str(out)],env=clean,check=True,timeout=30)
    j=read(out);assert len(j['providers'])==5 and j['model_loaded'] is False
    for x in j['providers']:
        n='libllama.so.0.8.0' if x['symbol'].startswith('llama_') else 'libggml-base.so.0.28.0' if x['symbol'].startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0'
        p=Path(b['libraries'][n]['path']);s=p.stat();assert x['path']==str(p) and (x['device'],x['inode'],x['bytes'])==(s.st_dev,s.st_ino,s.st_size)
    j['active_backend_providers']=[dict(name='CUDA0',graph_compute_provider=b['libraries']['libggml-cuda.so.0.28.0']['path']),dict(name='CPU',graph_compute_provider=b['libraries']['libggml-cpu.so.0.28.0']['path'])];write(out,j)
def verify(run):
    import array,math
    assert sys.byteorder=='little'
    j=read(run/'RUN.json');s=read(run/'SPEC.json');b=read(run/'BUILD.json');assert j['passed'] and j['spec']==s
    pin=read(run/'PROVIDER_EXPECTED.json');assert j['providers']==pin['providers'] and j['active_backend_providers']==pin['active_backend_providers']
    assert j['compact_bytes']==127401984000 and j['vision_loaded'] and len(j['placement'])==40 and len(j['shared_inventory'])==37
    for l,row in enumerate(j['shared_inventory']):
        assert row['layer']==l and [w['type'] for w in row['weights']]==['q6_K','q8_0','q6_K']
        assert [w['ne'] for w in row['weights']]==[[5120,2304,1,1],[5120,2304,1,1],[2304,5120,1,1]] and all(w['buffer']=='CUDA0' for w in row['weights'])
    corpus=read(M/'CASES.json')['cases'];assert len(j['blocks'])==2
    for block,ci in zip(j['blocks'],s['cases']):
        c=corpus[ci];assert block['case_index']==ci and block['case_id']==c['case_id'] and block['prefix_ids']==c['prefix_ids'] and block['input_tokens']==c['used_T1_ids'] and block['positions_begin']==c['prefix_count']
        assert len(block['start_ns'])==64 and len(block['output_ready_ns'])==64 and block['status']==[0]*64
        assert all(y>x for x,y in zip(block['start_ns'],block['output_ready_ns'])) and all(x>y for x,y in zip(block['start_ns'][1:],block['output_ready_ns'][:-1]))
        expected_rows=1 if s['stage']=='timing' else 65
        assert block['rows']==expected_rows and block['traced_calls']==(4 if s['stage']=='observe' else 0)
        assert block['file']==c['case_id']+'.f32' and (run/block['file']).stat().st_size==expected_rows*129280*4
        arr=array.array('f');arr.frombytes((run/block['file']).read_bytes());assert len(arr)==expected_rows*129280 and all(math.isfinite(v) for v in arr)

    for block in j['blocks']:
        a=block['prefix_stats_before_prefill'];b0=block['prefix_stats_before_T1'];b1=block['prefix_stats_after_T1']
        if s['arm']=='B':
            assert j['prefix_provider']==b['libraries']['libggml-base.so.0.28.0']['path']
            assert a[0]==b0[0]==b1[0]==1 and a[4]==b0[4]==b1[4]==0
            assert b0[2]-a[2]==b0[3]-a[3]==0 and b1[2]-b0[2]==b1[3]-b0[3]==37*64
            assert 27648<=b1[5]<=65536 and b1[6]==37
            if s['stage']!='timing':
                prev=b0;assert len(block['per_call_prefix_stats'])==64
                for cur in block['per_call_prefix_stats']:
                    assert cur[2]-prev[2]==cur[3]-prev[3]==37 and cur[4]==0 and cur[6]==37;prev=cur
        else:
            assert a==b0==b1==[0]*7 and 'prefix_provider' not in j
        reference=R/'main/runs/baseline'/block['file'];data=(run/block['file']).read_bytes();original=reference.read_bytes()
        assert data==(original[-129280*4:] if s['stage']=='timing' else original),'candidate output mismatch'
    if s['stage']=='observe':
        trace=read(run/'trace169-summary.json');assert trace['calls']==8 and trace['splits']==904
        assert all(trace[k]==0 for k in ['unknown_kernel_correlation','foreign_thread_calls','dropped_inside','unknown_records','errors'])
    health=read(run/'health.json');assert health['child_returncode']==0 and health['status']=='exited'
    return j
def one(stage):
    from api_window import clear_monitor,active
    state=read(M/'STATE.json');assert stage['name'] not in state['runs'],'explicit revision required to retry'
    b,base,env=prepare(stage);clear_monitor(R)
    if stage['stage']!='quality':assert read(M/'QUALITY_GATE.json')['passed']
    for unit in ['llm-deepseek41-api.service','llm-openai-api.service']:assert not active(unit),unit
    mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Mlocked:'))};assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    run=M/'runs'/stage['name'];run.mkdir(parents=True);owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid);providers(b,env,run/'PROVIDER_EXPECTED.json');write(run/'SPEC.json',stage);write(run/'BUILD.json',b)
    command=[*base['prefix'],str(M/'driver'),'--study-output',str(run),'--study-cases',str(M/'CASES.json'),'--run-spec',str(run/'SPEC.json'),*base['args']]
    write(run/'server-config.json',dict(command=command,environment=env));write(run/'LAUNCH.json',dict(command=command,environment=env,spec=stage,driver_sha256=sha(M/'driver'),build_sha256=sha(M/'BUILD.json'),cases_sha256=sha(M/'CASES.json'),design_sha256=sha(M/'DESIGN.json'),engine_build_sha256=sha(R/'prefix-engine/BUILD.json'),baseline_config_sha256=base['baseline_config_sha256'],model_manifest_sha256=base['model_manifest_sha256'],guard_source_sha256=base['guard_source_sha256'],runner_sha256=sha(Path(__file__)),reserve_GiB=64,budget_seconds=7200))
    state['phase']='Running '+stage['name'];state['runs'][stage['name']]=dict(path=str(run),status='running');write(M/'STATE.json',state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(E));guard=module('shared_guard',E/'supervise.py');guard.ROOT=run;original=guard.check_host;deadline=time.monotonic()+7200
    def bounded(initial):
        cpu,available=original(initial)
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError('Shared study reserve/time guard')
        return cpu,available
    guard.check_host=bounded;print('Starting '+stage['name'],flush=True);code=guard.main();state=read(M/'STATE.json');state['runs'][stage['name']].update(status='child-exited',returncode=code);write(M/'STATE.json',state);assert code==0
    verify(run);write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=read(M/'STATE.json');state['runs'][stage['name']].update(status='passed',manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(M/'STATE.json',state);print('Passed '+stage['name'],flush=True)
def main():
    from api_window import window
    assert os.geteuid()==0;lock=(R/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(M/'STATE.json').get('completed'):print('Diagnostic complete; no API action');return
    design=read(M/'DESIGN.json')
    for s in design['stages']:prepare(s)
    with window(R,'SHARED_PREFIX_TRIAL'):
        for stage in design['stages']:
            state=read(M/'STATE.json')
            if stage['name'] in state['runs']:
                assert state['runs'][stage['name']]['status']=='passed';verify(Path(state['runs'][stage['name']]['path']))
                if stage['stage']=='quality' and not (M/'QUALITY_GATE.json').exists():write(M/'QUALITY_GATE.json',dict(passed=True,both_cases_rows=65,logits_byte_identical=True,each_T1_exact_37_early_and_tail=True,baseline='main/runs/baseline',observer_absent=True,reconstructed_from_verified_run=True))
                continue
            one(stage)
            if stage['stage']=='quality':write(M/'QUALITY_GATE.json',dict(passed=True,both_cases_rows=65,logits_byte_identical=True,each_T1_exact_37_early_and_tail=True,baseline='main/runs/baseline',observer_absent=True))
    state=read(M/'STATE.json');state.update(completed=True,phase='One fixed candidate trial complete and API window closed; analyze ABBA');write(M/'STATE.json',state)
if __name__=='__main__':main()
