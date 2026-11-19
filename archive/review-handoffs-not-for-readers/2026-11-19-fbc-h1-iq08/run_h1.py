from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,resource,subprocess,sys,time,csv
from api_window import window,clear_monitor,active
from verify_h1 import verify
R=Path(__file__).resolve().parent;H=R/'h1';LAB=R.parent;E=LAB/'deepseek41-bringup-20261101';OLD=LAB/'parallel-pilot-20261117/replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(stage):
    old=module('old_replay',OLD/'run_capacity.py');oldbuild,base,args,env=old.prepare(1024)
    build=read(H/'BUILD.json');assert build['returncode']==0 and sha(H/'driver')==build['driver_sha256'] and build['engine_libraries']==oldbuild['engine_libraries']
    for n,h in build['source_sha256'].items():assert sha(H/n)==h,n
    for v in build['engine_libraries']['diagnostic'].values():assert sha(Path(v['path']))==v['sha256']
    plan=H/(stage['plan']+'.plan');assert sha(plan)==read(H/'H1_EXPECTED.json')['plans'][stage['plan']]['sha256']
    env.update(LD_LIBRARY_PATH=str(H/'runtime')+':'+str(E/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),GGML_XEON_HOT_PLAN=str(plan),GGML_XEON_H1_LABEL=stage['plan'],GGML_XEON_H1_EXPECTED=str(H/'H1_EXPECTED.json'),GGML_SCHED_DEBUG='0')
    return build,base,args,env
def providers(build,env,out):
    clean={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))};clean.update(env)
    subprocess.run([str(H/'driver'),'--check-providers',str(out)],env=clean,check=True,timeout=30)
    j=read(out);libs=build['engine_libraries']['diagnostic'];assert j['model_loaded'] is False and len(j['providers'])==16
    for x in j['providers']:
        n='libllama.so.0.8.0' if x['symbol'].startswith('llama_') else 'libggml-base.so.0.28.0' if x['symbol'].startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0';p=Path(libs[n]['path']);st=p.stat();assert x['path']==str(p) and (x['device'],x['inode'],x['bytes'])==(st.st_dev,st.st_ino,st.st_size)
    j['active_backend_providers']=[dict(name='CUDA0',graph_compute_provider=libs['libggml-cuda.so.0.28.0']['path']),dict(name='CPU',graph_compute_provider=libs['libggml-cpu.so.0.28.0']['path'])];write(out,j)
def one(stage):
    clear_monitor(R);assert not active('llm-deepseek41-api.service') and not active('llm-openai-api.service');build,base,args,env=prepare(stage)
    mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Mlocked:'))};assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    gpu='GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3';raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=10);rows=list(csv.reader(raw.strip().splitlines()));assert len(rows)==1 and rows[0][1].strip()==gpu and rows[0][2].strip().upper()=='00000000:84:00.0'
    run=H/'runs'/stage['name'];run.mkdir(parents=True);o=LAB.stat();os.chown(run,o.st_uid,o.st_gid);providers(build,env,run/'PROVIDER_EXPECTED.json');write(run/'SPEC.json',stage);write(run/'BUILD.json',build)
    command=[*base['prefix'],str(H/'driver'),'--study-output',str(run),'--study-cases',str(H/'CASES.json'),'--wiring-reference',str(LAB/'route37-calibration-20261117/runs/20261117T012730Z-F-before-a30e72f7'),'--fixture',read(H/'CAPTURE_PIN.json')['directory'],*args]
    write(run/'server-config.json',dict(command=command,environment=env));write(run/'LAUNCH.json',dict(command=command,environment=env,spec=stage,GPU_identity_raw=raw,build_sha256=sha(H/'BUILD.json'),design_sha256=sha(H/'DESIGN.json'),bank_plan_sha256=sha(H/(stage['plan']+'.plan')),capture_pin_sha256=sha(H/'CAPTURE_PIN.json'),baseline_config_sha256=base['baseline_config_sha256'],guard_source_sha256=base['guard_source_sha256'],runner_sha256=sha(Path(__file__)),reserve_GiB=64,budget_seconds=7200))
    state=read(H/'STATE.json');assert stage['name'] not in state['runs'];state['phase']='Running '+stage['name'];state['runs'][stage['name']]=dict(path=str(run),status='running');write(H/'STATE.json',state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(E));guard=module('h1_guard',E/'supervise.py');guard.ROOT=run;original=guard.check_host;deadline=time.monotonic()+7200
    def bounded(initial):
        cpu,available=original(initial)
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError('H1 reserve/time guard')
        return cpu,available
    guard.check_host=bounded;print('Starting '+stage['name'],flush=True);code=guard.main()
    state=read(H/'STATE.json');state['runs'][stage['name']].update(status='child-exited',returncode=code);write(H/'STATE.json',state)
    assert code==0;verify(run);write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=read(H/'STATE.json');state['runs'][stage['name']].update(status='passed',manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(H/'STATE.json',state);print('Passed '+stage['name'],flush=True)
def main():
    assert os.geteuid()==0
    lock=(R/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(H/'STATE.json').get('completed'):return
    for p in R.glob('API_WINDOW_FBC*.json'):assert not read(p)['restore_required'],'resolve FBC restoration before H1'
    assert read(R/'fbc/STATE.json')['completed'],'FBC first';prepare(read(H/'DESIGN.json')['stages'][0])
    with window(R,'H1'):
        for s in read(H/'DESIGN.json')['stages']:
            st=read(H/'STATE.json')['runs']
            if s['name'] in st:assert st[s['name']]['status']=='passed';verify(Path(st[s['name']]['path']))
            else:one(s)
        state=read(H/'STATE.json');state.update(completed=True,phase='Completed H1 fourprocesses');write(H/'STATE.json',state)
if __name__=='__main__':main()
