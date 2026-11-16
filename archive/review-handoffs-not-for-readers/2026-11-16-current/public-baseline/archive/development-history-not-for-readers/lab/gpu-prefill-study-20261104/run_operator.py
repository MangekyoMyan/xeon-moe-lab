#!/usr/bin/env python3
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
P=L/'strata-tc-study-20261031/cupti/cuda_cupti-linux-x86_64-12.9.79-archive'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--tokens',type=int,default=256);ap.add_argument('--experts',type=int,default=96)
    ap.add_argument('--pattern',choices=['uniform','skew'],default='uniform');ap.add_argument('--mode',choices=['timing','trace','pmu'],default='timing');ap.add_argument('--candidate',type=int,choices=[0,1,2],default=0)
    ap.add_argument('--compute-buffer',action='store_true');ap.add_argument('--details',action='store_true');a=ap.parse_args()
    case=next(c for c in json.loads((L/'expert-validation-20261101/cases.json').read_text()) if c['name']==a.case)
    with (R/'.operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((E/'server-config.json').read_text());key=Path(config['api_key_file']).read_text().strip()
        op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def idle():
            req=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers={'Authorization':'Bearer '+key})
            with op.open(req,timeout=3) as f:
                if any(x.get('is_processing') for x in json.load(f)):raise RuntimeError('Production API became busy; standalone result must be discarded')
        def watch():
            s=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
            if time.time()-s['unix']>45 or s['errors'] or s['active_alerts']:raise RuntimeError('Monitor stale/alarm')
            g=s['gpu']
            if g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5 or g['gpu_C']>=58 or g['hbm_C']>=63 or max(s['cpu'].values())>=59:raise RuntimeError('Hardware operating boundary')
            return {'unix':s['unix'],'gpu':g,'cpu':s['cpu'],'host_counters':s.get('host_counters')}
        idle();initial=watch()
        folder=R/'operators'/f'{time.time_ns()}-{a.case}-T{a.tokens}-E{a.experts}-{a.pattern}-{a.mode}-{a.candidate}'
        folder.mkdir(parents=True)
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD','CUDA_VISIBLE_DEVICES']}
        env.update(config['environment']);env['GGML_CUDA_DISABLE_GRAPHS']='1'
        if a.compute_buffer:env['XEON_PROBE_COMPUTE_BUFFER']='1'
        if a.details:env['XEON_PMU_DETAILS']='1'
        env['LD_LIBRARY_PATH']=str(P/'lib')+':'+env['LD_LIBRARY_PATH']
        if (R/'runtime/libggml-cuda.so.0.28.0').exists():env['LD_LIBRARY_PATH']=str(R/'runtime')+':'+env['LD_LIBRARY_PATH']
        env['LD_PRELOAD']=str(R/'grouped_dispatch.so')
        if a.mode=='trace':env['LD_PRELOAD']+=':'+str(R/'activity_trace.so');env['XEON_ACTIVITY_FILE']=str(folder/'activity.tsv')
        command=['numactl','--physcpubind=6','--membind=1',str(R/'operator_probe'),case['shard'],case['tensor'],str(a.tokens),str(a.experts),a.pattern,a.mode,str(a.candidate)]
        rec={'command':command,'config_sha256':hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest(),'initial':initial,'samples':[],'error':None,'production_not_restarted':True,'synthetic_routing':True,
             'probe_sha256':hashlib.sha256((R/'operator_probe').read_bytes()).hexdigest(),'shim_sha256':hashlib.sha256((R/'grouped_dispatch.so').read_bytes()).hexdigest(),'ld_library_path':env['LD_LIBRARY_PATH']}
        rec['compute_buffer']=a.compute_buffer;rec['detailed_pmu']=a.details
        if (R/'CUDA_BUILD.json').exists():rec['cuda_build']=json.loads((R/'CUDA_BUILD.json').read_text())
        child=None;started=time.monotonic()
        try:
            with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
                child=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
                while child.poll() is None:
                    s=watch();rec['samples'].append(s)
                    if s['host_counters']!=initial['host_counters']:raise RuntimeError('Host error counter change')
                    idle()
                    if time.monotonic()-started>100:raise RuntimeError('100 second probe bound')
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
                idle();rec['exit_code']=child.returncode
                if child.returncode:raise RuntimeError('Probe failure; inspect stderr')
        except BaseException as ex:rec['error']=repr(ex);raise
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            rec['elapsed_s']=time.monotonic()-started
            rec['exit_code']=child.returncode if child else None
            (folder/'RESULT.json').write_text(json.dumps(rec,indent=2)+'\n')
            print('RESULT_FOLDER',folder,flush=True)
            print((folder/'stdout.txt').read_text()[-5000:],flush=True)
            if rec['error']:print((folder/'stderr.txt').read_text()[-2500:],flush=True)

if __name__=='__main__':main()
