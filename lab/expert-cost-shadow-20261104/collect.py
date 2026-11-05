#!/usr/bin/env python3
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;L=R.parent;ENGINE=L/'deepseek41-bringup-20261101'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','calibration','test'],required=True)
    ap.add_argument('--case',default='all');ap.add_argument('--workload');a=ap.parse_args()
    config=json.loads((ENGINE/'server-config.json').read_text());key=Path(config['api_key_file']).read_text().strip()
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def idle():
        req=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers={'Authorization':'Bearer '+key})
        with http.open(req,timeout=3) as f:
            if any(s.get('is_processing') for s in json.load(f)):raise RuntimeError('Production API busy; calibration stopped')
    def monitor():
        d=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());g=d['gpu']
        if not 0<=time.time()-d['unix']<45 or d['errors'] or d['active_alerts']:raise RuntimeError('Monitor stale/alarm')
        if g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5 or g['gpu_C']>=58 or g['hbm_C']>=63 or max(d['cpu'].values())>=59:raise RuntimeError('Operating boundary')
        return {k:d[k] for k in ['unix','gpu','cpu','host_counters']}
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']}
    env.update(config['environment']);env['GGML_CUDA_DISABLE_GRAPHS']='1'
    cases=json.loads((L/'expert-validation-20261101/cases.json').read_text())
    if a.case!='all':cases=[c for c in cases if c['name']==a.case]
    if not cases:raise RuntimeError('Unknown operator case')
    workloads=[w for w in json.loads((R/'WORKLOADS.json').read_text()) if w['split']==a.split and (not a.workload or w['id']==a.workload)]
    if not workloads:raise RuntimeError('No workload')
    if a.split=='test' and not (R/'MODEL.json').exists():raise RuntimeError('Freeze MODEL.json before final test collection')
    eval_lock=R/'EVALUATION_LOCK.json'
    if eval_lock.exists():
        lockinfo=json.loads(eval_lock.read_text())
        if a.split!='test':raise RuntimeError('Evaluation began; new training/calibration is forbidden in this study')
        if hashlib.sha256((R/'MODEL.json').read_bytes()).hexdigest()!=lockinfo['model_sha256']:raise RuntimeError('Frozen evaluation model changed')
    signature={'schema':2,'scope':'GPU activations and host routes ready -> GPU output ready; no queue; standalone node',
        'shared_route_fetch':'excluded equally; already available, account once per routing group outside choice',
        'native_moe_cache':False,'memory_headroom':'conservative estimate +400MiB, not a hard reservation or no-eviction guarantee',
        'cpu_threads':24,'cpu_affinity':'0-23 strict','numa':'interleave0,1','cpu_compact':'not used; T>=64',
        'THP':'process disabled and verified, matching production without_thp.py',
        'gpu':'PG500-216 sm70','power_W':150,'ECC':True,'graphs':'disabled','source':'registered_readonly_mmap',
        'gpu_stream_buffer':'COMPUTE','gpu_resident_buffer':'WEIGHTS','cpu_activation_buffer':'CPU backend allocation',
        'build':json.loads((R/'BUILD.json').read_text()),'config_sha256':hashlib.sha256((ENGINE/'server-config.json').read_bytes()).hexdigest(),
        'libraries':[]}
    for stem in ['libggml-cpu.so.0.28.0','libggml-cuda.so.0.28.0','libggml-base.so.0.28.0']:
        p=next(Path(d)/stem for d in env['LD_LIBRARY_PATH'].split(':') if (Path(d)/stem).exists()).resolve()
        signature['libraries'].append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    sig=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();root=R/'data';root.mkdir(exist_ok=True)
    with (R/'.calibration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.split=='test':
            from cost_model import verify_model
            model=json.loads((R/'MODEL.json').read_text());verify_model(model)
            if sig!=model['timing_signature_sha256']:raise RuntimeError('Frozen model timing signature mismatch BEFORE heldout collection')
            if hashlib.sha256((R/'WORKLOADS.json').read_bytes()).hexdigest()!=model['study_plan']['workloads_sha256']:raise RuntimeError('Frozen workload definitions changed')
            planned={w['id']:w for w in model['study_plan']['workloads'] if w['split']=='test'}
            if any(w['id'] not in planned or w['sha256']!=planned[w['id']]['sha256'] for w in workloads):raise RuntimeError('Unplanned test workloads')
            if any(c!=model['case_definitions'].get(c['name']) for c in cases):raise RuntimeError('Frozen operator definitions changed')
        if a.split=='test' and not eval_lock.exists():
            with eval_lock.open('x') as f:json.dump({'model_sha256':hashlib.sha256((R/'MODEL.json').read_bytes()).hexdigest(),'signature_sha256':sig},f,indent=2)
        for case in cases:
            for workload in workloads:
                if eval_lock.exists():
                    frozen=json.loads(eval_lock.read_text())
                    if a.split!='test' or frozen['model_sha256']!=hashlib.sha256((R/'MODEL.json').read_bytes()).hexdigest() or frozen['signature_sha256']!=sig:
                        raise RuntimeError('Frozen evaluation boundary/signature mismatch')
                folder=root/(case['name']+'--'+workload['id']);receipt=folder/'RESULT.json'
                if receipt.exists():
                    prior=json.loads(receipt.read_text())
                    if prior.get('ok') and prior.get('signature_sha256')==sig and prior['workload']==workload:continue
                    raise RuntimeError('Existing incomplete/different case; preserve for review: '+str(folder))
                idle();initial=monitor();folder.mkdir(exist_ok=False)
                if hashlib.sha256(Path(workload['path']).read_bytes()).hexdigest()!=workload['sha256']:raise RuntimeError('Workload changed')
                if hashlib.sha256((R/'calibration_probe').read_bytes()).hexdigest()!=signature['build']['binary_sha256']:raise RuntimeError('Calibration executable changed')
                cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(R/'calibration_probe'),case['shard'],case['tensor'],str(workload['tokens']),str(workload['experts']),workload['path'],str(workload['seed']),'4']
                record={'signature':signature,'signature_sha256':sig,'workload':workload,'case':case,'command':cmd,'initial':initial,'samples':[],'ok':False}
                if a.split=='test':record['frozen_model_sha256']=hashlib.sha256((R/'MODEL.json').read_bytes()).hexdigest()
                process=None;started=time.monotonic()
                try:
                    with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
                        process=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
                        while process.poll() is None:
                            snapshot=monitor();record['samples'].append(snapshot)
                            if snapshot['host_counters']!=initial['host_counters']:raise RuntimeError('Host counter change')
                            idle()
                            if time.monotonic()-started>120:raise RuntimeError('Case exceeded120seconds')
                            try:process.wait(timeout=1)
                            except subprocess.TimeoutExpired:pass
                    if process.returncode:raise RuntimeError('Probe failed; inspect '+str(folder/'stderr.txt'))
                    idle();result=json.loads((folder/'stdout.txt').read_text())
                    if not result['pass'] or result['histogram']!=workload['histogram']:raise RuntimeError('Result/route mismatch')
                    record['result']=result;record['ok']=True
                finally:
                    if process and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    record['elapsed_s']=time.monotonic()-started;record['exit_code']=process.returncode if process else None
                    receipt.write_text(json.dumps(record,indent=2)+'\n')
                import statistics
                print(json.dumps({'case':case['name'],'workload':workload['id'],'CPU_ms':statistics.median(result['cpu_gpu_boundary_ms']),'GPU_stream_ms':statistics.median(result['gpu_stream_ms'])}),flush=True)
if __name__=='__main__':main()
