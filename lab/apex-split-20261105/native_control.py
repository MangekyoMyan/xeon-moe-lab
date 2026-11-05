from pathlib import Path
import argparse,hashlib,json,subprocess,os,urllib.request,time,signal,fcntl,statistics
R=Path(__file__).resolve().parent;E=R.parent/'deepseek41-bringup-20261101'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,default=R/'VALIDATION_PLAN.json');args=ap.parse_args()
    config=json.loads((E/'server-config.json').read_text());key=Path(config['api_key_file']).read_text().strip();http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def check():
        req=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers={'Authorization':'Bearer '+key})
        with http.open(req,timeout=3) as f:
            if any(x.get('is_processing') for x in json.load(f)):raise RuntimeError('API busy')
        d=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());g=d['gpu']
        if not 0<=time.time()-d['unix']<45 or d['errors'] or d['active_alerts'] or g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5 or g['gpu_C']>=58 or g['hbm_C']>=63 or max(d['cpu'].values())>=59:raise RuntimeError('Monitor boundary')
        return {k:d[k] for k in ['unix','gpu','cpu','host_counters']}
    build=json.loads((R/'BUILD.json').read_text());cmd=list(build['commands'][-1]);cmd[cmd.index(str(R/'split_probe.cpp'))]=str(R/'native_cpu_probe.cpp');cmd[cmd.index('-o')+1]=str(R/'native_cpu_probe')
    subprocess.run(cmd,check=True,timeout=60);nb={'command':cmd,'included_split_BUILD':build,'sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['native_cpu_probe.cpp','native_cpu_probe','split_probe.cpp','kernels.cu']}}
    (R/'NATIVE_BUILD.json').write_text(json.dumps(nb,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','XEON_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']};env.update(config['environment']);env['GGML_CUDA_DISABLE_GRAPHS']='1'
    wmap={w['name']:w for w in json.loads((R/'WORKLOADS.json').read_text())};plan=json.loads(args.plan.read_text());seen=set()
    with (R/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in plan['jobs']:
            identity=(job['layer'],job['workload'])
            if identity in seen:continue
            seen.add(identity)
            w=wmap[job['workload']];initial=check();folder=R/'native'/f"L{job['layer']}-{job['workload']}";folder.mkdir(parents=True,exist_ok=False)
            cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(R/'native_cpu_probe'),str(Path(config['model']).parent),str(job['layer']),w['path'],str(w['tokens']),str(w['seed']),'7','24']
            record={'ok':False,'job':job,'workload':w,'build':nb,'command':cmd,'initial':initial,'samples':[],'config_sha256':hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest()};child=None;start=time.monotonic()
            try:
                with (folder/'stdout.json').open('w') as out,(folder/'stderr.log').open('w') as err:
                    child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
                    while child.poll() is None:
                        d=check();record['samples'].append(d)
                        if d['host_counters']!=initial['host_counters'] or time.monotonic()-start>90:raise RuntimeError('Counter/time boundary')
                        try:child.wait(timeout=1)
                        except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError('Native CPU control failed: '+str(folder/'stderr.log'))
                result=json.loads((folder/'stdout.json').read_text());assert result['pass'];record['result']=result;record['ok']=True
            finally:
                if child and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record['elapsed_s']=time.monotonic()-start;record['exit_code']=child.returncode if child else None;(folder/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps({'case':folder.name,'native_CPU_boundary_ms':statistics.median(result['wall_ms']),'compute_ms':statistics.median(result['cpu_compute_ms'])}),flush=True)
if __name__=='__main__':main()
