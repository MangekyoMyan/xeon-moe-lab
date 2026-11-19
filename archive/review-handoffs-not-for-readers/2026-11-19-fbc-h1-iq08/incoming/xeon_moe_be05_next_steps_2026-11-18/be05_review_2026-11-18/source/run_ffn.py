"""One frozen CPU-only operator screen under the existing hardware supervisor."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,urllib.request
R=Path(__file__).resolve().parent;Q=R/'q8-avx1';LAB=R.parent;E=LAB/'deepseek41-bringup-20261101'
STATE=Q/'STATE-v2.json';RUNS=Q/'runs-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def active(unit):return subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0
def capture_pin(design):
    root=Path(design['common']['capture_json']).parent;m=root/'RESULTS_SHA256.json'
    assert sha(m)==design['common']['capture_manifest_sha256']
    for name,digest in read(m).items():assert Path(name).name==name and sha(root/name)==digest,name
def clear_monitor():
    w=read(Path('/var/lib/xeon-watch/latest.json'));assert 0<=time.time()-w['unix']<=45 and not w['errors'] and not w['active_alerts'],'monitor condition'
    for unit in ['xeon-watch.service','xeon-watch-notify.service']:assert active(unit)
    for p in [E/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED',*Q.glob('runs*/**/STOP_REQUESTED')]:assert not p.exists(),'protection stop requires review'
def verify(run):
    h=read(run/'health.json');assert h['status']=='exited' and h['child_returncode']==0
    j=read(run/'RUN.json');s=read(run/'SPEC.json');assert j['passed'] and j['spec']==s
    assert [(b['cell'],b['threads']) for b in j['blocks']]==[(c,t) for c in s['cells'] for t in s['threads']]
    for b in j['blocks']:
        assert b['output_matches_capture'] and len(b['samples'])==s['samples']
        assert all(x['end_ns']>x['begin_ns'] and x['calls']==s['loops'] for x in b['samples'])
        assert sorted(b['worker_receipt_after_timing']['CPU'])==list(range(b['threads']))
    return j
def run_one(stage,design):
    state=read(STATE);assert stage['name'] not in state['runs'],'never retry a stage silently'
    if stage['stage']=='timing':assert read(Q/'FFN_NUMERIC_GATE.json')['passed']
    clear_monitor();capture_pin(design);assert not active('llm-deepseek41-api.service') and not active('llm-openai-api.service')
    cpu=Q/stage['mode']/'runtime/libggml-cpu.so.0.28.0';build=read(Q/'BUILD.json')
    assert sha(cpu)==build['variants'][stage['mode']]['CPU_sha256']
    common=design['common'];base=Path(common['base']);assert sha(base)=='2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558'
    run=RUNS/stage['name'];run.mkdir(parents=True);owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid);spec={**common,**stage,'CPU':str(cpu)};write(run/'SPEC.json',spec)
    env=dict(LD_LIBRARY_PATH=str(cpu.parent)+':'+str(base.parent),LD_PRELOAD='',OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',GGML_CPU_IQ_ROW_LOCAL='0')
    command=['/usr/bin/python3',str(LAB/'moe-implementation-20261026/release/without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',str(Q/'real_ffn'),str(run/'SPEC.json'),str(run)]
    write(run/'server-config.json',dict(command=command,environment=env))
    write(run/'LAUNCH.json',dict(command=command,environment=env,design_sha256=sha(Q/'FFN_DESIGN.json'),CPU_sha256=sha(cpu),base_sha256=sha(base),binary_sha256=sha(Q/'real_ffn'),spec_sha256=sha(run/'SPEC.json'),guard_sha256=sha(E/'supervise.py'),runner_sha256=sha(Path(__file__)),reserve_GiB=64,budget_seconds=600))
    state['phase']='Running '+stage['name'];state['runs'][stage['name']]=dict(path=str(run),status='running');write(STATE,state)
    sys.path.insert(0,str(E));modspec=importlib.util.spec_from_file_location('ffn_guard',E/'supervise.py');guard=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(guard);guard.ROOT=run
    original=guard.check_host;deadline=time.monotonic()+600
    def check(initial):
        cpu_temp,available=original(initial)
        if available<64*1024**2:raise RuntimeError('CPU operator RAM reserve64GiB')
        if time.monotonic()>deadline:raise TimeoutError('CPU operator10min limit')
        return cpu_temp,available
    guard.check_host=check;print('Starting '+stage['name'],flush=True);code=guard.main()
    state=read(STATE);state['runs'][stage['name']].update(status='child-exited',returncode=code);write(STATE,state)
    assert code==0,(stage['name'],code);j=verify(run)
    write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=read(STATE);state['runs'][stage['name']].update(status='passed',manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(STATE,state);print('Passed '+stage['name'],flush=True)
    return j
def main():
    assert os.geteuid()==0
    lock=(Q/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(STATE);assert not state['runs'],'fresh controller only; inspect prior state before resuming'
    design=read(Q/'FFN_DESIGN.json');assert read(Q/'NUMERIC.json')['passed']
    assert sha(Q/'BUILD.json')==design['CPU_build_sha256'] and sha(Q/'FFN_BUILD.json')==design['FFN_build_sha256']
    assert sha(Q/'real_ffn')==read(Q/'FFN_BUILD.json')['binary_sha256'] and sha(E/'supervise.py')==design['guard_sha256'] and sha(E/'server-config.json')==design['baseline_config_sha256']
    for f in design['common']['shards']:
        p=Path(f);s=design['common']['shard_stats'][p.name];assert p.stat().st_size==s['bytes'] and p.stat().st_mtime_ns==s['mtime_ns']
    clear_monitor();capture_pin(design);was_active=active('llm-deepseek41-api.service');inherited_restore=bool(state.get('restore_required'));state.update(API_initially_active=was_active or inherited_restore,restore_required=inherited_restore);write(STATE,state)
    if was_active:
        cfg=read(E/'server-config.json');key=Path(cfg['api_key_file']).read_text().strip();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request=urllib.request.Request(f"http://{cfg['host']}:{cfg['port']}/slots",headers={'Authorization':'Bearer '+key})
        with opener.open(request,timeout=10) as f:slots=json.load(f)
        assert isinstance(slots,list) and slots and all(s.get('is_processing') is False for s in slots),'wait for idle API before testing'
    stop_attempted=inherited_restore
    try:
        if was_active:
            state=read(STATE);state['restore_required']=True;write(STATE,state)
            stop_attempted=True
            subprocess.run(['systemctl','stop','llm-deepseek41-api.service'],check=True,timeout=90)
        for stage in design['stages']:
            j=run_one(stage,design)
            if stage['name']=='quality-candidate':
                a=verify(RUNS/'quality-control');b=j
                for aa,bb in zip(a['blocks'],b['blocks']):
                    assert aa['weights']==bb['weights'] and (RUNS/'quality-control'/aa['output_file']).read_bytes()==(RUNS/'quality-candidate'/bb['output_file']).read_bytes()
                write(Q/'FFN_NUMERIC_GATE.json',dict(passed=True,cases=3,thread_conditions=[12,24],all_outputs_match_capture=True,control_candidate_byte_equal=True))
        state=read(STATE);state.update(phase='All realFFN comparisons completed',completed=True);write(STATE,state)
    finally:
        if stop_attempted:
            try:
                # A hardware protection stop deliberately prevents automatic restart.
                clear_monitor()
                assert sha(E/'server-config.json')==design['baseline_config_sha256']
                if not active('llm-deepseek41-api.service'):
                    # start.py uses the persistent sensor snapshot. Wait for that
                    # snapshot to observe released GPU memory; do not relax its guard.
                    deadline=time.monotonic()+60
                    while True:
                        clear_monitor();snapshot=read(Path('/var/lib/xeon-watch/latest.json'))
                        if snapshot['gpu']['used_MiB']<=128:break
                        if time.monotonic()>deadline:raise TimeoutError('GPU release not reflected by monitor')
                        time.sleep(2)
                    subprocess.run(['python3',str(E/'start.py')],check=True,timeout=60)
                subprocess.run(['python3',str(R/'verify_restore.py')],check=True,timeout=660)
                state=read(STATE);state.update(restore_required=False,normal_API_restored=True);write(STATE,state)
            except Exception as error:
                state=read(STATE);state.update(restore_required=True,restoration_error=str(error));write(STATE,state)
                raise
    print('Operator study ended; original API state restored',flush=True)
if __name__=='__main__':main()
