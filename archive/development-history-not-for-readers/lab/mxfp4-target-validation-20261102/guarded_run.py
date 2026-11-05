#!/usr/bin/env python3
"""Bounded subprocess using the existing thermal daemon; no fan/service changes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
SNAPSHOT=Path('/var/lib/xeon-watch/latest.json')

def snapshot():
    j=json.loads(SNAPSHOT.read_text())
    if time.time()-j['unix']>45:raise RuntimeError('thermal snapshot stale')
    if j.get('errors') or j.get('active_alerts'):raise RuntimeError('thermal monitor reports errors/alerts')
    cpu=j.get('cpu',{})
    if not cpu or max(cpu.values())>=60:raise RuntimeError('CPU missing or >=60C')
    b=j.get('bmc',{})
    if time.time()-b.get('observed_unix',0)>120:raise RuntimeError('BMC snapshot stale')
    if b.get('peripheral_C',0)>=45 or b.get('pch_C',0)>=55:raise RuntimeError('BMC temperature limit')
    return {'unix':j['unix'],'cpu_max_C':max(cpu.values()),'peripheral_C':b.get('peripheral_C'),
            'pch_C':b.get('pch_C'),'host_counters':j.get('host_counters')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    if '/' in a.name or a.name in ['.','..']:ap.error('invalid name')
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:ap.error('command missing')
    folder=ROOT/'execution'/a.name;folder.mkdir(parents=True,exist_ok=False)
    initial=snapshot();record={'command':cmd,'started_unix':time.time(),'initial':initial,'samples':[]}
    process=None;error=None;started=time.monotonic();exitcode=1
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update(OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',GIT_CEILING_DIRECTORIES=str(ROOT))
    try:
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            while process.poll() is None:
                s=snapshot()
                if s['host_counters']!=initial['host_counters']:raise RuntimeError('host ECC/throttle counters changed')
                record['samples'].append(s)
                if time.monotonic()-started>a.timeout:raise TimeoutError('bounded test timeout')
                time.sleep(2)
            exitcode=process.returncode
    except BaseException as exc:
        error=repr(exc)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        record.update(exitcode=exitcode,error=error,finished_unix=time.time(),elapsed_seconds=time.monotonic()-started)
        (folder/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'name':a.name,'exitcode':exitcode,'error':error,'elapsed_seconds':record['elapsed_seconds'],
                      'max_cpu_C':max([initial['cpu_max_C']]+[s['cpu_max_C'] for s in record['samples']]),'logs':str(folder)}),flush=True)
    return 1 if error else exitcode

if __name__=='__main__':raise SystemExit(main())
