"""Fresh FBC diagnostic/measurement group; API restoration managed by window helper."""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,subprocess,sys,time,fcntl
from verify_fbc import verify_run,gate
R=Path(__file__).resolve().parent;F=R/'fbc';LAB=R.parent;E=LAB/'deepseek41-bringup-20261101';OLD=LAB/'route37-integration-20261117'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(stage):
    build=read(F/'BUILD.json');assert build['returncode']==0 and sha(F/'driver')==build['driver_sha256']
    for n,h in build['source_sha256'].items():assert sha(F/n)==h,n
    for v in build['engine_libraries']['diagnostic'].values():assert sha(Path(v['path']))==v['sha256']
    eng=read(R/'fbc-engine/ENGINE_BUILD.json');assert eng['libllama_sha256']==build['engine_libraries']['diagnostic']['libllama.so.0.8.0']['sha256']
    for n,h in eng['source_sha256'].items():assert sha(R/'fbc-engine'/n)==h,n
    old=module('old_profile',OLD/'run_capture.py');base=old.prepare(False);env=base['environment'].copy()
    env.update(LD_LIBRARY_PATH=str(F/'runtime')+':'+str(E/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),LD_BIND_NOW='1',GGML_XEON_HOT_FFN='0' if stage['mode']=='F' else '1',GGML_XEON_HOT_PLAN=str(F/'capacity.plan'),GGML_XEON_HOT_TRIAL_KEEP_ORIGINAL_GRAPH='1' if stage['mode']=='B' else '0',GGML_SCHED_DEBUG='2' if stage['stage']=='quality' else '0')
    return build,base,env
def providers(build,env,out):
    clean={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))};clean.update(env)
    subprocess.run([str(F/'driver'),'--check-providers',str(out)],env=clean,check=True,timeout=30)
    j=read(out);libs=build['engine_libraries']['diagnostic'];assert len(j['providers'])==16 and j['model_loaded'] is False
    for x in j['providers']:
        n='libllama.so.0.8.0' if x['symbol'].startswith('llama_') else 'libggml-base.so.0.28.0' if x['symbol'].startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0';p=Path(libs[n]['path']);st=p.stat();assert x['path']==str(p) and (x['device'],x['inode'],x['bytes'])==(st.st_dev,st.st_ino,st.st_size)
    j['active_backend_providers']=[dict(name='CUDA0',graph_compute_provider=libs['libggml-cuda.so.0.28.0']['path'])]
    if env['GGML_XEON_HOT_FFN']=='1':j['active_backend_providers'].append(dict(name='XEON_HOT_FFN',graph_compute_provider=libs['libllama.so.0.8.0']['path']))
    j['active_backend_providers'].append(dict(name='CPU',graph_compute_provider=libs['libggml-cpu.so.0.28.0']['path']));write(out,j)
def one(stage):
    from api_window import clear_monitor,active
    state=read(F/'STATE.json');assert stage['name'] not in state['runs'],'explicit revision required for retry'
    build,base,env=prepare(stage);clear_monitor(R);assert not active('llm-deepseek41-api.service')
    if stage['stage']=='timing':assert read(F/'QUALITY_GATE.json')['passed']
    mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Mlocked:'))};assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    run=F/'runs'/stage['name'];run.mkdir(parents=True);o=LAB.stat();os.chown(run,o.st_uid,o.st_gid);providers(build,env,run/'PROVIDER_EXPECTED.json');write(run/'SPEC.json',stage)
    command=[*base['prefix'],str(F/'driver'),'--study-output',str(run),'--study-cases',str(F/'CASES.json'),'--run-spec',str(run/'SPEC.json'),*base['args']]
    write(run/'server-config.json',dict(command=command,environment=env));write(run/'BUILD.json',build)
    write(run/'LAUNCH.json',dict(command=command,environment=env,spec=stage,driver_sha256=sha(F/'driver'),build_sha256=sha(F/'BUILD.json'),engine_build_sha256=sha(R/'fbc-engine/ENGINE_BUILD.json'),cases_sha256=sha(F/'CASES.json'),design_sha256=sha(F/'DESIGN.json'),bank_plan_sha256=sha(F/'capacity.plan'),baseline_config_sha256=base['baseline_config_sha256'],model_manifest_sha256=base['model_manifest_sha256'],guard_source_sha256=base['guard_source_sha256'],runner_sha256=sha(Path(__file__)),reserve_GiB=64,budget_seconds=7200))
    state['phase']='Running '+stage['name'];state['runs'][stage['name']]=dict(path=str(run),status='running');write(F/'STATE.json',state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(E));guard=module('fbc_guard',E/'supervise.py');guard.ROOT=run;original=guard.check_host;deadline=time.monotonic()+7200
    def bounded(initial):
        cpu,available=original(initial)
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError('FBC reserve/time guard')
        return cpu,available
    guard.check_host=bounded;print('Starting '+stage['name'],flush=True);code=guard.main()
    state=read(F/'STATE.json');state['runs'][stage['name']].update(status='child-exited',returncode=code);write(F/'STATE.json',state)
    assert code==0;verify_run(run)
    write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=read(F/'STATE.json');state['runs'][stage['name']].update(status='passed',manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(F/'STATE.json',state);print('Passed '+stage['name'],flush=True)
def main():
    from api_window import window
    assert os.geteuid()==0
    lock=(R/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(F/'STATE.json').get('completed'):print('FBC already complete; no API action');return
    prepare(read(F/'DESIGN.json')['stages'][0])
    with window(R,'FBC'):
        for stage in read(F/'DESIGN.json')['stages']:
            state=read(F/'STATE.json')
            if stage['name'] in state['runs']:
                assert state['runs'][stage['name']]['status']=='passed';verify_run(Path(state['runs'][stage['name']]['path']))
                if stage['name']=='quality-C' and not (F/'QUALITY_GATE.json').exists():gate()
                continue
            one(stage)
            if stage['name']=='quality-C':gate()
        state=read(F/'STATE.json');state.update(phase='Completed9processes',completed=True,timed_calls=768);write(F/'STATE.json',state)
if __name__=='__main__':main()
