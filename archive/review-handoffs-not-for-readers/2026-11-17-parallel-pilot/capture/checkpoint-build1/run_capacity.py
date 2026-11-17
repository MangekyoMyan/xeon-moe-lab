"""Run one explicitly selected capacity point under the unchanged host guard.

No automatic API stop/restore. --prepare checks pins without loading a model.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, resource, subprocess, sys, time, uuid
R=Path(__file__).resolve().parent;LAB=R.parent.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
OLD=LAB/'route37-integration-20261117'
REFERENCE=LAB/'route37-calibration-20261117/runs/20261117T012730Z-F-before-a30e72f7'
GPU='GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(ubatch):
    build=json.loads((R/'BUILD.json').read_text());assert build['returncode']==0 and sha(R/'driver')==build['driver_sha256']
    for name,digest in build['source_sha256'].items():assert sha(R/name)==digest,name
    for item in build['engine_libraries']['diagnostic'].values():assert sha(Path(item['path']))==item['sha256']
    ref=json.loads((REFERENCE/'RUN.json').read_text());health=json.loads((REFERENCE/'health.json').read_text())
    assert ref['passed'] and ref['arm']=='F-before' and health['status']=='exited' and health['child_returncode']==0
    for name,digest in json.loads((REFERENCE/'RESULTS_SHA256.json').read_text()).items():assert sha(REFERENCE/name)==digest,name
    assert sha(REFERENCE/'correct_F.f32')=='95430744558e7414b336f6548a6c03e064e12e1adc03cb8256fc45dcfc3187b0'
    old=module('original_capture_prepare',OLD/'run_capture.py');base=old.prepare(False)
    args=base['args'].copy();args[args.index('-ub')+1]=str(ubatch)
    env=base['environment'].copy();env['LD_LIBRARY_PATH']=str(R/'runtime')+':'+str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib')
    env['LD_BIND_NOW']='1';env['CUDA_VISIBLE_DEVICES']=GPU
    return build,base,args,env
def providers(build,env,out):
    full={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))};full.update(env)
    subprocess.run([str(R/'driver'),'--check-providers',str(out)],env=full,check=True,timeout=30)
    data=json.loads(out.read_text());libs=build['engine_libraries']['diagnostic'];assert data['model_loaded'] is False and len(data['providers'])==16
    for item in data['providers']:
        sym=item['symbol'];name='libllama.so.0.8.0' if sym.startswith('llama_') else 'libggml-base.so.0.28.0' if sym.startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0'
        p=Path(libs[name]['path']);st=p.stat();assert item['path']==str(p) and (item['device'],item['inode'],item['bytes'])==(st.st_dev,st.st_ino,st.st_size)
    data['active_backend_providers']=[dict(name='CUDA0',graph_compute_provider=libs['libggml-cuda.so.0.28.0']['path']),dict(name='CPU',graph_compute_provider=libs['libggml-cpu.so.0.28.0']['path'])]
    data['CUDA_VISIBLE_DEVICES']=GPU;write(out,data)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true');ap.add_argument('--ubatch',type=int,choices=[1024,512,256],default=1024);a=ap.parse_args()
    build,base,args,env=prepare(a.ubatch)
    if a.prepare:
        out=R/('PREFLIGHT-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'.json');providers(build,env,out);print('Provider pins verified; model not loaded: '+str(out));return 0
    if os.geteuid()!=0:raise RuntimeError('Use sudo for the existing supervisor')
    for name in ['llm-deepseek41-api.service','llm-openai-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',name]).returncode==0:raise RuntimeError('Ordinary API still active; not stopped automatically')
    for name in ['xeon-watch.service','xeon-watch-notify.service']:assert subprocess.run(['systemctl','is-active','--quiet',name]).returncode==0,'Monitor inactive'
    stops=[ENGINE/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED']+list((R/'runs').glob('*/STOP_REQUESTED'))
    assert not any(p.exists() for p in stops),'Protection stop requires investigation'
    monitor=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());assert 0<=time.time()-monitor['unix']<=45 and not monitor['errors'] and not monitor['active_alerts']
    mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    gpu_raw=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=10)
    gpu_rows=list(csv.reader(gpu_raw.strip().splitlines()));assert len(gpu_rows)==1
    assert gpu_rows[0][1].strip()==GPU and gpu_rows[0][2].strip().upper()=='00000000:84:00.0'
    run=R/'runs'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'Z-U'+str(a.ubatch)+'-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True);owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    providers(build,env,run/'PROVIDER_EXPECTED.json')
    command=[*base['prefix'],str(R/'driver'),'--study-output',str(run),'--study-cases',str(R/'CASES.json'),'--wiring-reference',str(REFERENCE),*args]
    write(run/'server-config.json',dict(command=command,environment=env));write(run/'BUILD.json',build)
    write(run/'LAUNCH_MANIFEST.json',dict(command=command,environment=env,ubatch=a.ubatch,GPU_identity_raw=gpu_raw,reference=str(REFERENCE),reference_manifest_sha256=sha(REFERENCE/'RESULTS_SHA256.json'),runner_sha256=sha(Path(__file__)),build_sha256=sha(R/'BUILD.json'),baseline_config_sha256=base['baseline_config_sha256'],model_manifest_sha256=base['model_manifest_sha256'],guard_source_sha256=base['guard_source_sha256'],reserve_GiB=64,budget_seconds=7200,bank=False,study='bounded8_activation_capture',pilot_design_sha256=sha(R/'PILOT_DESIGN.json')))
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(ENGINE));supervisor=module('existing_capacity_supervisor',ENGINE/'supervise.py');supervisor.ROOT=run
    original=supervisor.check_host;deadline=time.monotonic()+7200
    def strict(initial):
        cpu,available=original(initial)
        if available<64*1024**2:raise RuntimeError('Capacity study RAM reserve below64GiB')
        if time.monotonic()>deadline:raise TimeoutError('Capacity study2hour limit')
        return cpu,available
    supervisor.check_host=strict
    print('Capacity run: '+str(run),flush=True);code=supervisor.main()
    d=json.loads((run/'RUN.json').read_text()) if (run/'RUN.json').exists() else {};code=code or (0 if d.get('passed') else 1)
    write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    print(json.dumps(dict(run=str(run),code=code,passed=d.get('passed',False))),flush=True);return code
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:raise SystemExit(str(error))
