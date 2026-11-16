"""Bound auxiliary work, using existing daemon snapshots and API CPU activity."""
from pathlib import Path
import argparse,fcntl,json,math,os,signal,subprocess,time
R=Path(__file__).resolve().parent
LAB=R.parents[1]
def snapshot():
    d=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    if time.time()-d['unix']>45 or d.get('errors') or d.get('active_alerts'):raise RuntimeError('Monitor stale/error/alert')
    cpu=list(d.get('cpu',{}).values());b=d.get('bmc',{})
    if not cpu or not all(math.isfinite(v) for v in cpu) or max(cpu)>=57:raise RuntimeError('CPU missing or >=57C test limit')
    if time.time()-b.get('observed_unix',0)>120:raise RuntimeError('BMC stale')
    for key,limit in [('pch_C',55),('peripheral_C',45)]:
        v=b.get(key)
        if v is None or not math.isfinite(v) or v>=limit:raise RuntimeError('Board sensor unavailable/limit')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    if available<64*1024**2:raise RuntimeError('MemAvailable below64GiB reserve')
    return {'unix':d['unix'],'cpu_max_C':max(cpu),'host_counters':d.get('host_counters'),'available_kib':available}
def api_ticks(pid):
    if pid is None:return 0
    p=Path('/proc')/str(pid)
    if (p/'comm').read_text().strip()!='llama-server':raise RuntimeError('API process identity changed')
    s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
    return int(v[11])+int(v[12])
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--timeout',type=int,default=300);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not a.name.replace('-','').replace('_','').isalnum():raise RuntimeError('Bad run name')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise RuntimeError('No command')
    lock=(R/'.experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=R/'execution'/a.name;folder.mkdir(parents=True,exist_ok=False)
    h=json.loads((LAB/'deepseek41-bringup-20261101/health.json').read_text());pid=h.get('pid') if h.get('status')=='running' else None
    t=time.monotonic();ticks=api_ticks(pid);time.sleep(.3)
    now=time.monotonic();new=api_ticks(pid);hz=os.sysconf('SC_CLK_TCK')
    if (new-ticks)/hz/(now-t)>.5:raise RuntimeError('API active')
    first=snapshot();record={'argv':command,'api_pid':pid,'started_unix':time.time(),'initial':first,'samples':[]}
    process=None;error=None;rc=None;started=time.monotonic();ticks=new;t=now
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update(OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1')
    try:
        with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
            process=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
            while process.poll() is None:
                s=snapshot();now=time.monotonic();new=api_ticks(pid);cores=(new-ticks)/hz/max(now-t,.001)
                if cores>.5:raise RuntimeError('API became active; auxiliary work cancelled')
                if s['host_counters']!=first['host_counters']:raise RuntimeError('Host counters changed')
                s['api_cores']=cores;record['samples'].append(s);ticks=new;t=now
                if now-started>a.timeout:raise TimeoutError('Auxiliary work timeout')
                time.sleep(.5)
            rc=process.returncode
    except BaseException as exc:error=repr(exc)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        record.update(exit_code=rc,error=error,elapsed_s=time.monotonic()-started,finished_unix=time.time())
        (folder/'STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'folder':str(folder),'exit_code':rc,'error':error,'elapsed_s':record['elapsed_s']}),flush=True)
    return 1 if error or rc!=0 else 0
if __name__=='__main__':raise SystemExit(main())
