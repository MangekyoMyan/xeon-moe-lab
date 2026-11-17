"""Bounded isolated caller, using the existing GPU/thermal supervisor.

Requires the ordinary API already stopped. Never changes hardware settings or
automatically restarts a model. --prepare resolves/checks both arms without a model.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import uuid

R=Path(__file__).resolve().parent
LAB=R.parent
ENGINE=LAB/'deepseek41-bringup-20261101'
OLD=LAB/'route37-integration-20261117'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def pin():
    build=json.loads((R/'BUILD.json').read_text())
    assert build['returncode']==0 and sha(R/'driver')==build['driver_sha256']
    for name,digest in build['source_sha256'].items(): assert sha(R/name)==digest,name
    for files in build['engine_libraries'].values():
        for info in files.values(): assert sha(Path(info['path']))==info['sha256'],info['path']
    old=module('original_capture_prepare',OLD/'run_capture.py')
    base=old.prepare(False) # Reuse the original model/profile/source pin; not its driver.
    return build,base

def expected(arm, build, base, dest):
    kind='diagnostic' if arm=='D' else 'frozen'
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))}
    env.update(base['environment'])
    env['LD_LIBRARY_PATH']=str(R/('runtime-'+kind))+':'+str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib')
    env['LD_BIND_NOW']='1'
    subprocess.run([str(R/'driver'),'--check-providers','D' if arm=='D' else 'F',str(dest)],env=env,check=True,timeout=30)
    data=json.loads(dest.read_text()); assert data['model_loaded'] is False
    libs=build['engine_libraries'][kind]
    for p in data['providers']:
        symbol=p['symbol'];name='libllama.so.0.8.0' if symbol.startswith('llama_') else 'libggml-base.so.0.28.0' if symbol.startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0'
        want=Path(libs[name]['path']);stat=want.stat()
        assert p['path']==str(want) and (p['device'],p['inode'],p['bytes'])==(stat.st_dev,stat.st_ino,stat.st_size)
    assert len(data['providers'])==(16 if arm=='D' else 4)
    data.update(arm=arm,active_backend_providers=[
        dict(name='CUDA0',graph_compute_provider=libs['libggml-cuda.so.0.28.0']['path']),
        dict(name='CPU',graph_compute_provider=libs['libggml-cpu.so.0.28.0']['path'])])
    write(dest,data)
    return {k:v for k,v in env.items() if k.startswith(('GGML_','OMP_','GOMP_','LD_','CUDA_'))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--arm',choices=['F-before','D','F-after'],default='F-before')
    ap.add_argument('--reference',type=Path)
    args=ap.parse_args(); build,base=pin()
    if args.prepare:
        dest=R/('preflight-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()));dest.mkdir()
        for arm in ['F-before','D']:expected(arm,build,base,dest/(arm+'.json'))
        write(dest/'RECEIPT.json',dict(passed=True,model_started=False,build_sha256=sha(R/'BUILD.json')))
        print('Pinned function providers verified without model startup: '+str(dest));return 0
    if os.geteuid()!=0:raise RuntimeError('Run with sudo for the existing hardware guard')
    if args.arm!='F-before':
        assert args.reference and args.reference.parent==R/'runs'
        ref=json.loads((args.reference/'RUN.json').read_text());rh=json.loads((args.reference/'health.json').read_text())
        assert ref['passed'] and ref['arm']=='F-before' and rh['status']=='exited' and rh['child_returncode']==0
        assert sha(args.reference/'BUILD.json')==sha(R/'BUILD.json'),'reference from a different build'
        frozen_files=json.loads((args.reference/'RESULTS_SHA256.json').read_text())
        for name in ['RUN.json','PROVIDER_EXPECTED.json','correct_F.json','correct_F.f32','BUILD.json','LAUNCH_MANIFEST.json']:
            assert sha(args.reference/name)==frozen_files[name],'reference evidence changed: '+name
    for service in ['llm-deepseek41-api.service','llm-openai-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0:raise RuntimeError(service+' active; not automatically stopping it')
    for service in ['xeon-watch.service','xeon-watch-notify.service']:
        assert subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0,'monitor inactive'
    stops=[ENGINE/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED']
    stops+=list((R/'runs').glob('*/STOP_REQUESTED'))+list((OLD/'runs').glob('*/STOP_REQUESTED'))
    assert not any(p.exists() for p in stops),'Protection stop requires investigation'
    snap=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    assert 0<=time.time()-snap['unix']<=45 and not snap['errors'] and not snap['active_alerts']
    mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    run=R/'runs'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'Z-'+args.arm+'-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True)
    owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    env=expected(args.arm,build,base,run/'PROVIDER_EXPECTED.json')
    cmd=[*base['prefix'],str(R/'driver'),'--study-output',str(run),'--study-arm',args.arm,'--study-cases',str(R/'CASES.json')]
    if args.reference:cmd+=['--study-reference',str(args.reference)]
    cmd+=base['args']
    write(run/'server-config.json',dict(command=cmd,environment=env))
    write(run/'BUILD.json',build)
    write(run/'LAUNCH_MANIFEST.json',dict(command=cmd,environment=env,arm=args.arm,reference=str(args.reference) if args.reference else None,build_sha256=sha(R/'BUILD.json'),runner_sha256=sha(Path(__file__)),baseline_config_sha256=base['baseline_config_sha256'],model_manifest_sha256=base['model_manifest_sha256'],guard_source_sha256=base['guard_source_sha256'],reserve_GiB=64,budget_seconds=7200,API_automatically_stopped_or_restarted=False))
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    sys.path.insert(0,str(ENGINE));supervisor=module('existing_study_supervisor',ENGINE/'supervise.py');supervisor.ROOT=run
    original=supervisor.check_host;deadline=time.monotonic()+7200
    def strict(initial):
        cpu,available=original(initial)
        if available<64*1024**2:raise RuntimeError('Study RAM reserve below64GiB')
        if time.monotonic()>deadline:raise TimeoutError('Study2hour bound exceeded')
        return cpu,available
    supervisor.check_host=strict
    print('Study run: '+str(run),flush=True)
    code=supervisor.main()
    report=json.loads((run/'RUN.json').read_text()) if (run/'RUN.json').exists() else {}
    code=code or (0 if report.get('passed') else 1)
    write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    print(json.dumps(dict(run=str(run),code=code,passed=report.get('passed',False))),flush=True)
    return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:raise SystemExit(str(e))
