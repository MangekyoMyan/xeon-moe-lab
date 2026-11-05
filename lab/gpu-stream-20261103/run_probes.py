#!/usr/bin/env python3
"""Bounded real-weight CPU/CUDA operator tests while the production API is stopped."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parent;LAB=R.parent
ENGINE=LAB/'deepseek41-bringup-20261101'
WATCH=Path('/var/lib/xeon-watch/latest.json')

def check_watch():
    j=json.loads(WATCH.read_text())
    if time.time()-j['unix']>45 or j['errors'] or j['active_alerts']:
        raise RuntimeError('Hardware monitor stale or reports an issue')
    g=j['gpu']
    if g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5:
        raise RuntimeError('GPU operating policy mismatch')
    if max(j['cpu'].values())>=60:
        raise RuntimeError('CPU temperature >=60C')
    return {'unix':j['unix'],'cpu':j['cpu'],'gpu':g,'host_counters':j.get('host_counters')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','sweep','focus','mxfp4-large'],required=True)
    ap.add_argument('--cpu',choices=['stock','hybrid'],default='stock');a=ap.parse_args()
    R.joinpath('results').mkdir(exist_ok=True)
    with (R/'.probe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode==0:
            raise RuntimeError('API must be stopped before this study')
        initial=check_watch()
        config=json.loads((ENGINE/'server-config.json').read_text())
        env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD','CUDA_VISIBLE_DEVICES']}
        env.update(config['environment'])
        for k in list(env):
            if k.startswith('GGML_MOE_HIST_'):del env[k]
        env['LD_LIBRARY_PATH']=str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib')
        if a.cpu=='hybrid':env['LD_LIBRARY_PATH']=str(LAB/'mxfp4-hybrid-20261102/runtime/hybrid')+':'+env['LD_LIBRARY_PATH']
        env['GGML_CUDA_REGISTER_HOST']='1'
        cases=json.loads((LAB/'expert-validation-20261101/cases.json').read_text())
        if a.phase=='smoke':
            cases=[c for c in cases if c['name']=='MXFP4-up'];sizes=[(8,6),(128,384),(512,384)]
        elif a.phase=='sweep':sizes=[(128,384),(512,384),(1024,384)]
        elif a.phase=='mxfp4-large':
            cases=[c for c in cases if c['type']=='MXFP4'];sizes=[(128,384),(512,384),(1024,384)]
        else:
            cases=[c for c in cases if c['type'] in ['MXFP4','IQ3_XXS']]
            sizes=[(1,6),(8,6),(32,96),(128,96),(512,384)]
        stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
        out=R/'results'/f'{stamp}-{a.phase}-{a.cpu}';out.mkdir()
        provenance={'phase':a.phase,'cpu':a.cpu,'initial_watch':initial,'model_startup':False,
          'binary_sha256':hashlib.sha256((R/'stream_probe').read_bytes()).hexdigest(),
          'source_sha256':hashlib.sha256((R/'stream_probe.cpp').read_bytes()).hexdigest(),
          'ldd':subprocess.check_output(['ldd',str(R/'stream_probe')],env=env,text=True)}
        (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
        allrows=[]
        for case in cases:
            for tokens,experts in sizes:
                before=check_watch()
                if before['host_counters']!=initial['host_counters']:raise RuntimeError('Host ECC/throttle counter change')
                name=f'{case["name"]}-t{tokens}-e{experts}'
                command=['numactl','--physcpubind=0-23','--interleave=0,1',str(R/'stream_probe'),case['shard'],case['tensor'],str(tokens),str(experts),str(12 if tokens==1 else 24),'42']
                started=time.monotonic();child=None;samples=[]
                with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
                    try:
                        child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                        while child.poll() is None:
                            snapshot=check_watch();samples.append(snapshot)
                            if snapshot['host_counters']!=initial['host_counters']:raise RuntimeError('Host ECC/throttle counter change')
                            if time.monotonic()-started>180:raise RuntimeError('180-second case limit')
                            try:child.wait(timeout=1)
                            except subprocess.TimeoutExpired:pass
                    finally:
                        if child is not None and child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        (out/(name+'.run.json')).write_text(json.dumps({'command':command,'elapsed_s':time.monotonic()-started,
                          'exit_code':child.returncode if child else None,'before':before,'samples':samples},indent=2)+'\n')
                raw=(out/(name+'.stdout')).read_text()
                row=json.loads(raw) if raw.strip().startswith('{') else None
                if child.returncode or not row or not row['pass']:raise RuntimeError('Operator test failed: '+str(out/name))
                allrows.append(dict(case=case,result=row))
                (out/'result.json').write_text(json.dumps(allrows,indent=2)+'\n')
                keys=['cpu_ms','gpu_resident_ms','pageable_upload_compute_ms','registered','registered_upload_compute_ms','registered_all_io_compute_ms','registration_ms','cpu_gpu_nmse']
                print(json.dumps({'case':name,**{k:row[k] for k in keys}}),flush=True)
        print('Saved: '+str(out),flush=True)

if __name__=='__main__':main()
