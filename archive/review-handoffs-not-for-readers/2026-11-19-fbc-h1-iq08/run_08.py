from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from api_window import window,clear_monitor,active
from verify_08 import verify,gate
R=Path(__file__).resolve().parent;D=R/'cpu08';LAB=R.parent;E=LAB/'deepseek41-bringup-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def pins(design):
    assert sha(D/'BUILD.json')==design['CPU_build_sha256'] and sha(D/'PROBE_BUILD.json')==design['probe_build_sha256']
    build=read(D/'BUILD.json');probe=read(D/'PROBE_BUILD.json');assert sha(D/'real_ffn')==probe['binary_sha256'] and sha(D/'quality-counts.so')==probe['counter_sha256']
    for n,h in probe['source_sha256'].items():assert sha(D/n)==h
    for mode in ['control','candidate']:assert sha(D/mode/'runtime/libggml-cpu.so.0.28.0')==build['variants'][mode]['CPU_sha256']
    cap=Path(design['common']['capture_json']).parent;assert sha(cap/'RESULTS_SHA256.json')==design['common']['capture_manifest_sha256']
    for n,h in read(cap/'RESULTS_SHA256.json').items():assert sha(cap/n)==h
    for name in design['common']['shards']:
        p=Path(name);expected=design['common']['shard_stats'][p.name];st=p.stat();assert st.st_size==expected['bytes'] and st.st_mtime_ns==expected['mtime_ns']
    plan=LAB/'exploration-20261112/plans/frequency19200.plan';assert sha(plan)==design['common']['cpu_plan_sha256'];actual={}
    for line in plan.read_text().splitlines()[1:]:
        x=line.split();ids=list(map(int,x[6:]));assert len(ids)==int(x[5]);actual[x[0]]=ids
    assert actual==design['common']['cpu_plan']
    assert sha(E/'supervise.py')==design['guard_sha256'] and sha(E/'server-config.json')==design['baseline_config_sha256']
def one(stage,design):
    pins(design);clear_monitor(R);assert not active('llm-deepseek41-api.service') and not active('llm-openai-api.service')
    if stage['stage']=='timing':assert read(D/'QUALITY_GATE.json')['passed']
    run=D/'runs'/stage['name'];run.mkdir(parents=True);o=LAB.stat();os.chown(run,o.st_uid,o.st_gid)
    cpu=D/stage['mode']/'runtime/libggml-cpu.so.0.28.0';base=Path(design['common']['base']);assert sha(base)=='2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558'
    spec={**design['common'],**stage,'CPU':str(cpu),'quality_counter':str(D/'quality-counts.so')};write(run/'SPEC.json',spec)
    env=dict(LD_LIBRARY_PATH=str(cpu.parent)+':'+str(base.parent),LD_PRELOAD=str(D/'quality-counts.so') if stage['stage']=='quality' else '',OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='12',OMP_MAX_ACTIVE_LEVELS='1',GGML_CPU_IQ_ROW_LOCAL='0')
    cmd=['/usr/bin/python3',str(LAB/'moe-implementation-20261026/release/without_thp.py'),'numactl','--physcpubind=0-11','--interleave=0,1',str(D/'real_ffn'),str(run/'SPEC.json'),str(run)]
    write(run/'server-config.json',dict(command=cmd,environment=env));write(run/'LAUNCH.json',dict(command=cmd,environment=env,spec=stage,design_sha256=sha(D/'DESIGN.json'),CPU_sha256=sha(cpu),base_sha256=sha(base),runner_sha256=sha(Path(__file__)),build_sha256=sha(D/'BUILD.json'),probe_sha256=sha(D/'PROBE_BUILD.json'),guard_sha256=sha(E/'supervise.py'),reserve_GiB=64,budget_seconds=600))
    state=read(D/'STATE.json');assert stage['name'] not in state['runs'];state['runs'][stage['name']]=dict(path=str(run),status='running');state['phase']='Running '+stage['name'];write(D/'STATE.json',state)
    sys.path.insert(0,str(E));ms=importlib.util.spec_from_file_location('iq08_guard',E/'supervise.py');guard=importlib.util.module_from_spec(ms);ms.loader.exec_module(guard);guard.ROOT=run;old=guard.check_host;deadline=time.monotonic()+600
    def check(initial):
        c,a=old(initial)
        if a<64*1024**2 or time.monotonic()>deadline:raise RuntimeError('08 reserve/time guard')
        return c,a
    guard.check_host=check;print('Starting08 '+stage['name'],flush=True);code=guard.main();state=read(D/'STATE.json');state['runs'][stage['name']].update(status='child-exited',returncode=code);write(D/'STATE.json',state)
    assert code==0;verify(run);write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=read(D/'STATE.json');state['runs'][stage['name']].update(status='passed',manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(D/'STATE.json',state);print('Passed08 '+stage['name'],flush=True)
def main():
    assert os.geteuid()==0;lock=(R/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(D/'STATE.json').get('completed'):return
    assert read(R/'h1/STATE.json')['completed'],'H1 first'
    for p in R.glob('API_WINDOW_*.json'):assert not read(p)['restore_required']
    design=read(D/'DESIGN.json');pins(design)
    with window(R,'CPU08'):
        for s in design['stages']:
            state=read(D/'STATE.json')
            if s['name'] in state['runs']:assert state['runs'][s['name']]['status']=='passed';verify(Path(state['runs'][s['name']]['path']))
            else:one(s,design)
            if s['name']=='quality-candidate' and not (D/'QUALITY_GATE.json').exists():gate()
        state=read(D/'STATE.json');state.update(completed=True,phase='Completed 08 fixed12T screen',timed_graph_calls=960);write(D/'STATE.json',state)
if __name__=='__main__':main()
