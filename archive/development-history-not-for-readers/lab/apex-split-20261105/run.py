from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--workload',required=True);p.add_argument('--chunk',type=int,default=16);p.add_argument('--threads',type=int,default=24);p.add_argument('--repeats',type=int,default=3);p.add_argument('--label',default='pilot');p.add_argument('--gpu-experts',type=int);a=p.parse_args()
    config_raw=(E/'server-config.json').read_bytes();config=json.loads(config_raw);key=Path(config['api_key_file']).read_text().strip();http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def idle():
        request=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers={'Authorization':'Bearer '+key})
        with http.open(request,timeout=3) as f:
            if any(x.get('is_processing') for x in json.load(f)):raise RuntimeError('API busy; bounded study stopped')
    def monitor():
        d=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());g=d['gpu']
        if not 0<=time.time()-d['unix']<45 or d['errors'] or d['active_alerts']:raise RuntimeError('Monitor stale/alarm')
        if g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5 or g['gpu_C']>=58 or g['hbm_C']>=63 or max(d['cpu'].values())>=59:raise RuntimeError('Operating boundary')
        return {k:d[k] for k in ['unix','gpu','cpu','host_counters']}
    w=next(w for w in json.loads((R/'WORKLOADS.json').read_text()) if w['name']==a.workload)
    build=json.loads((R/'BUILD.json').read_text())
    for name,digest in build['sha256'].items():
        if hashlib.sha256((R/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Build/source mismatch')
    if hashlib.sha256(Path(w['path']).read_bytes()).hexdigest()!=w['sha256']:raise RuntimeError('Route fixture changed')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','XEON_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']};env.update(config['environment']);env['GGML_CUDA_DISABLE_GRAPHS']='1'
    libs=[]
    for stem in ['libggml-cpu.so.0.28.0','libggml-cuda.so.0.28.0','libggml-base.so.0.28.0']:
        path=next(Path(d)/stem for d in env['LD_LIBRARY_PATH'].split(':') if (Path(d)/stem).exists()).resolve();libs.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    folder=R/'data'/f'{a.label}-L{a.layer}-{a.workload}-C{a.chunk}-t{a.threads}'
    with (R/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle();initial=monitor();folder.mkdir(parents=True,exist_ok=False)
        cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(R/'split_probe'),str(Path(config['model']).parent),str(a.layer),w['path'],str(w['tokens']),str(w['seed']),str(a.chunk),str(a.repeats),str(a.threads)]
        if a.gpu_experts is not None:cmd.append(str(a.gpu_experts))
        record={'ok':False,'arguments':vars(a),'workload':w,'build':build,'config_sha256':hashlib.sha256(config_raw).hexdigest(),'libraries':libs,'command':cmd,'initial':initial,'samples':[],
            'scope':'real full384 host weights; synthetic routes/activations; whole gate-up-clamp-down FFN; GPU compact streamed chunks; no model/request speedup claim',
            'timing':'GPU input + host routes available -> weighted merged GPU output; gather,D2H,H2D,CPU,GPU,scatter,merge included; plan/graph creation excluded and recorded separately',
            'selection':'pilot sweep (oracle)' if a.gpu_experts is None else 'p fixed before new timing; validate fixed split, not an adaptive controller',
            'GPU_cache':'none: all assigned GPU weights reuploaded each run; no resident expert policy modeled'}
        child=None;started=time.monotonic()
        try:
            with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
                child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
                while child.poll() is None:
                    d=monitor();record['samples'].append(d);idle()
                    if d['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counter changed')
                    if time.monotonic()-started>240:raise RuntimeError('Case time limit240seconds')
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('Probe failed: '+str(folder/'stderr.log'))
            result=json.loads((folder/'stdout.json').read_text());assert result['pass'] and result['histogram']==w['histogram'];record['result']=result;record['ok']=True
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record['elapsed_s']=time.monotonic()-started;record['exit_code']=child.returncode if child else None;(folder/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
        import statistics
        rows=[{'gpu_experts':x['gpu_experts'],'wall_ms':statistics.median(x['wall_ms']),'serial_ms':statistics.median(x['serial_wall_ms']) if x['serial_wall_ms'] else None,'nmse':x['nmse_vs_native_CPU']} for x in result['measurements']]
        print(json.dumps({'folder':str(folder),'elapsed_s':record['elapsed_s'],'rows':rows},indent=2))
if __name__=='__main__':main()
