#!/usr/bin/env python3
"""Perf stat ROI experiment; leave the live API unchanged and abort if it gets busy."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'mxfp4-target-validation-20261102'
BINBASE=ROOT.parent/'ipc-investigation-20261102'
GROUPS={
 'assists':['other_assists.avx_to_sse:u','other_assists.sse_to_avx:u'],
 'frontend':['idq_uops_not_delivered.core:u','resource_stalls.any:u'],
 'ports015':['uops_dispatched_port.port_0:u','uops_dispatched_port.port_1:u','uops_dispatched_port.port_5:u'],
 'loadports':['uops_dispatched_port.port_2:u','uops_dispatched_port.port_3:u','uops_dispatched_port.port_4:u'],
 'resources':['resource_stalls.rs:u','resource_stalls.rob:u','resource_stalls.sb:u'],
 'memory_stalls':['cycle_activity.cycles_no_execute:u','cycle_activity.stalls_l1d_miss:u','cycle_activity.stalls_l2_miss:u']}
spec=importlib.util.spec_from_file_location('guard',BASE/'guarded_run.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

def api_tasks():
    result={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if (p/'comm').read_text().strip()!='llama-server':continue
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            result[int(p.name)]=int(fields[11])+int(fields[12])
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result

def api_cores(before,after,seconds):
    if set(before)!=set(after):raise RuntimeError('API process set changed; stop test')
    return sum(after[k]-before[k] for k in after)/os.sysconf('SC_CLK_TCK')/seconds

def parse_perf(path,group):
    events={}
    for line in path.read_text().splitlines():
        fields=line.split(',')
        if len(fields)<5 or line.startswith('#'):continue
        try:value=float(fields[0]);percent=float(fields[4])
        except ValueError:raise RuntimeError('Unsupported/not-counted perf event: '+line)
        events[fields[2]]={'value':value,'unit':fields[1],'running_percent':percent,'counter_runtime_ns':float(fields[3])}
    for name in ['cycles:u','instructions:u','task-clock',*GROUPS[group]]:
        if name not in events or events[name]['running_percent']<99.0:raise RuntimeError('Incomplete or multiplexed counters: '+name)
    return events

def run_one(label,variant,threads,tokens,repeats,group):
    folder=ROOT/'runs'/label;folder.mkdir(parents=True,exist_ok=False)
    ctl=folder/'control.fifo';ack=folder/'ack.fifo';os.mkfifo(ctl,0o600);os.mkfifo(ack,0o600)
    binary=BINBASE/('probe-'+variant);cpus='0' if threads==1 else ('0-11' if threads==12 else '0-23')
    command=['perf','stat','-x',',','-D','-1','--control',f'fifo:{ctl},{ack}',
      '-e','{'+','.join(['cycles:u','instructions:u',*GROUPS[group]])+'}',
      '-e','task-clock,context-switches,cpu-migrations,page-faults','-o',str(folder/'perf.csv'),'--',
      'numactl','--interleave=0,1','taskset','-c',cpus,str(binary),
      '--type','mxfp4','--real-shapes','--mode','up','--experts','6','--topk','6','--routing','shared',
      '--tokens',str(tokens),'--threads',str(threads),'--strict-cpu','--repeats',str(repeats),'--warmups','5']
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update(OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',LC_ALL='C',
               IPC_CONTROL_FIFO=str(ctl),IPC_ACK_FIFO=str(ack))
    initial=guard.snapshot();api=api_tasks();time.sleep(.3)
    check=api_tasks()
    if api_cores(api,check,.3)>.5:raise RuntimeError('API busy; do not start measurement')
    record={'label':label,'pmu_group':group,'variant':variant,'threads':threads,'tokens_per_expert':tokens,'repeats':repeats,
            'command':command,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'started_unix':time.time(),'api_pids':list(api),'initial':initial,'samples':[]}
    (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    process=None;error=None;started=time.monotonic();last_api=check;last_check=started;next_thermal=started
    try:
        with (folder/'stdout.jsonl').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            while process.poll() is None:
                now=time.monotonic();current_api=api_tasks()
                if api_cores(last_api,current_api,max(now-last_check,.001))>.5:raise RuntimeError('API inference became active; stop this measurement')
                last_api=current_api;last_check=now
                if now>=next_thermal:
                    s=guard.snapshot();record['samples'].append(s);next_thermal=now+2
                    if s['host_counters']!=initial['host_counters']:raise RuntimeError('Host ECC/throttle counters changed')
                if now-started>90:raise TimeoutError('bounded perf timeout')
                time.sleep(.25)
            if process.returncode:raise RuntimeError(f'perf/probe exited {process.returncode}')
        lines=[json.loads(x) for x in (folder/'stdout.jsonl').read_text().splitlines()]
        assert len(lines)==2 and lines[-1]['cases']==1 and lines[-1]['failures']==0 and lines[0]['pass']
        data=lines[0]
        diag=[json.loads(x) for x in (folder/'stderr.log').read_text().splitlines() if x.startswith('{')]
        roi=next(x for x in diag if 'perf_roi_wall_ms' in x)
        cpu_ms=next(x['roi_process_cpu_ms'] for x in diag if 'roi_process_cpu_ms' in x)
        masks=next(x['strict_worker_affinities'] for x in diag if 'strict_worker_affinities' in x)
        assert len(masks)==threads and all(len(x['cpus'])==1 for x in masks)
        assert roi['perf_controls_acknowledged']==2
        events=parse_perf(folder/'perf.csv',group)
        assert events['task-clock']['unit']=='msec'
        task_ms=events['task-clock']['value']
        ratio=task_ms/cpu_ms
        if not .9<=ratio<=1.1:raise RuntimeError('Perf inheritance/ROI CPU-time mismatch: '+str(ratio))
        ins=events['instructions:u']['value'];cycles=events['cycles:u']['value']
        summary={'label':label,'variant':variant,'threads':threads,'tokens_per_expert':tokens,
          'graph_median_ms':data['median_ms'],'graph_mean_ms':sum(data['times_ms'])/len(data['times_ms']),
          'roi_wall_ms':roi['perf_roi_wall_ms'],'process_cpu_ms':cpu_ms,'task_clock_ms':task_ms,
          'perf_to_process_cpu_time_ratio':ratio,'average_scheduled_threads':task_ms/roi['perf_roi_wall_ms'],
          'ipc':ins/cycles,'instructions_per_graph':ins/repeats,'cycles_per_graph':cycles/repeats,
          'pmu_group':group,
          'pmu_events_per_graph':{name:events[name]['value']/repeats for name in GROUPS[group]},
          'pmu_events_per_cycle':{name:events[name]['value']/cycles for name in GROUPS[group]},
          'events':events,'output_fnv1a64':data['output_fnv1a64'],'reference_nmse':data['sample_reference_nmse'],
          'roi_excludes_weight_generation_reference_and_dump':True,'api_unchanged':True}
        (folder/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k in ['label','graph_median_ms','ipc','instructions_per_graph','cycles_per_graph','pmu_group','pmu_events_per_cycle','average_scheduled_threads','perf_to_process_cpu_time_ratio']}),flush=True)
    except BaseException as exc:error=repr(exc);raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        record.update(finished_unix=time.time(),error=error,exitcode=process.returncode if process else None)
        (folder/'status.json').write_text(json.dumps(record,indent=2)+'\n');ctl.unlink();ack.unlink()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['all',*GROUPS],default='all');a=ap.parse_args()
    groups=list(GROUPS) if a.group=='all' else [a.group]
    for group in groups:
        configs=[(1,8,21),(12,8,101)] if group not in ['loadports','resources','memory_stalls'] else [(1,8,21)]
        for threads,tokens,repeats in configs:
            for i,variant in enumerate(['stock','mxfp4','mxfp4','stock']):
                run_one(f'{group}-t{threads}-{i}-{variant}',variant,threads,tokens,repeats,group)

if __name__=='__main__':main()
