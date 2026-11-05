#!/usr/bin/env python3
"""Bounded, serial measurements on this CPU/RAM. No GPU simulation or GPU execution."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time

SRC=Path(__file__).resolve().parent
LAB=SRC.parent
OUT=LAB/'results/server-study-20261023'
def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_ARG_','OMP_','GOMP_')) and k!='LD_LIBRARY_PATH'}
    env.update(LC_ALL='C',OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',
               OPENBLAS_NUM_THREADS='1',GGML_CPU_NUMA_FULL='0',GGML_CPU_NUMA_REPLICA='0')
    return env

def read(p):
    try:return Path(p).read_text().strip()
    except OSError:return None

def sample():
    t={'time':time.monotonic(),'rapl':{},'temperature_C':{},'proc_stat':read('/proc/stat')}
    for p in Path('/sys/class/powercap').glob('intel-rapl:*'):
        name=read(p/'name')
        if name and (name.startswith('package-') or name=='dram'):
            t['rapl'][p.name]={'name':name,'energy_uj':int(read(p/'energy_uj')),'range_uj':int(read(p/'max_energy_range_uj'))}
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        if read(hw/'name')!='coretemp':continue
        for p in hw.glob('temp*_input'):
            t['temperature_C'][hw.name+':'+(read(p.with_name(p.name.replace('_input','_label'))) or p.name)]=int(read(p))/1000
    t['ECC']={p.name:{k:read(p/k) for k in ['ce_count','ue_count']} for p in Path('/sys/devices/system/edac/mc').glob('mc[0-9]*')}
    t['throttles']={str(p):read(p) for p in Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*throttle_count')}
    return t

def power(a,b):
    seconds=b['time']-a['time'];r={}
    for key,x in a['rapl'].items():
        y=b['rapl'][key];r[key]={'name':x['name'],'watts':((y['energy_uj']-x['energy_uj'])%x['range_uj'])/1e6/seconds}
    return {'seconds':seconds,'domains':r,'package_plus_dram_W':sum(x['watts'] for x in r.values()),
            'note':'RAPL package+DRAM estimates, not PSU output or wall power; excludes motherboard/drives/fans.'}

class Counter:
    def __init__(self,name,options,folder):
        self.name=name;self.events=[]
        cr,self.cw=os.pipe();self.ar,aw=os.pipe()
        self.log=(folder/(name+'-perf.log')).open('w')
        cmd=['perf','stat','-D','-1','--control',f'fd:{cr},{aw}','-x',',','--no-big-num','-o',str(folder/(name+'.csv')),*options]
        self.cmd=cmd
        self.proc=subprocess.Popen(cmd,pass_fds=(cr,aw),stdout=self.log,stderr=self.log,start_new_session=True)
        os.close(cr);os.close(aw)
    def control(self,action):
        begin=time.monotonic();os.write(self.cw,(action+'\n').encode())
        ready,_,_=select.select([self.ar],[],[],10)
        if not ready:raise RuntimeError(f'perf {self.name} {action} ack timed out')
        reply=os.read(self.ar,128)
        if b'ack' not in reply:raise RuntimeError(f'perf {self.name} invalid ack: {reply!r}')
        self.events.append({'action':action,'start':begin,'ack':time.monotonic()})
    def close(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        os.close(self.cw);os.close(self.ar);self.log.close()

def run(label,mode,threads,scope,source,dest,MiB,seconds):
    folder=OUT/'measurements'/label;folder.mkdir(parents=True,exist_ok=False)
    command=[str(SRC/'memory-probe'),mode,str(threads),scope,str(source),str(dest),str(MiB),str(seconds)]
    report={'command':command,'label':label,'mode':mode,'threads':threads,'scope':scope,
            'source_node':source,'destination_node':dest,'MiB':MiB,'seconds_requested':seconds,
            'boot_id':read('/proc/sys/kernel/random/boot_id'),'telemetry':[],'counters':[],'valid':False,
            'notes':['CPU/RAM measurement only; no GPU or simulated GPU traffic.','Read GB/s counts input bytes; copy GB/s counts payload copied, not read+write traffic.','Latency is dependent pointer chasing with4KiB pages; includes cache/TLB effects.','IMC counters include all host activity.']}
    counters=[];child=None;turbo=None;observer=None;stop=threading.Event();errors=[]
    def observe():
        try:
            while not stop.is_set():
                t=sample();report['telemetry'].append(t)
                if t['temperature_C'] and max(t['temperature_C'].values())>=80:
                    errors.append('CPU temperature cutoff');os.killpg(child.pid,signal.SIGTERM);break
                if any(int(v.get('ue_count') or 0)>int(report['before']['ECC'].get(k,{}).get('ue_count') or 0) for k,v in t['ECC'].items()):
                    errors.append('New uncorrectable ECC event');os.killpg(child.pid,signal.SIGTERM);break
                stop.wait(.5)
        except Exception as exc:errors.append(repr(exc))
    print('Starting '+label,flush=True)
    try:
        with (folder/'stderr.txt').open('w') as stderr,(folder/'stdout.txt').open('w') as stdout,(folder/'turbostat.stderr').open('w') as terr:
            child=subprocess.Popen(command,env=environment(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1,start_new_session=True)
            report['pid']=child.pid
            while True:
                line=child.stdout.readline()
                if not line:raise RuntimeError('Probe exited before ready')
                stdout.write(line);stdout.flush();row=json.loads(line)
                if row.get('ready'):report['ready']=row;break
            core=['task-clock','cycles:u','instructions:u','context-switches','cpu-migrations','minor-faults','major-faults','cycle_activity.stalls_total:u','cycle_activity.stalls_l2_pending:u']
            counters.append(Counter('core',['-p',str(child.pid),'-e',','.join(core)],folder))
            counters.append(Counter('imc',['-a','-A','-e','uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/'],folder))
            turbo_cmd=['turbostat','--quiet','--Summary','--interval','1','--show','Busy%,Bzy_MHz,PkgWatt,RAMWatt,PkgTmp,CoreTmp','--out',str(folder/'turbostat.tsv')]
            turbo=subprocess.Popen(turbo_cmd,stdout=terr,stderr=terr,start_new_session=True)
            report['turbostat_command']=turbo_cmd
            for c in counters:c.control('enable')
            report['before']=sample()
            report['process_io_before']=read(f'/proc/{child.pid}/io')
            observer=threading.Thread(target=observe);observer.start()
            report['send_G_time']=time.monotonic();child.stdin.write('G');child.stdin.flush()
            line=child.stdout.readline();report['receive_done_time']=time.monotonic()
            if not line:raise RuntimeError('Probe exited during measurement')
            stdout.write(line);stdout.flush();row=json.loads(line);assert row.get('done');report['result']=row
            report['after']=sample()
            report['process_io_after']=read(f'/proc/{child.pid}/io')
            for c in counters:c.control('disable')
            stop.set();observer.join(timeout=5)
            child.stdin.write('Q');child.stdin.flush()
            for line in child.stdout:stdout.write(line)
            stdout.flush()
            report['exit_code']=child.wait(timeout=10)
            if turbo.poll() is None:turbo.send_signal(signal.SIGINT)
            turbo.wait(timeout=5)
            for c in counters:
                c.close();report['counters'].append({'name':c.name,'command':c.cmd,'control':c.events,'exit_code':c.proc.returncode})
            counters=[]
            report['power']=power(report['before'],report['after'])
            report['observer_errors']=errors
            report['temperature_max_C']=max(v for t in report['telemetry'] for v in t['temperature_C'].values())
            report['thermal_throttle_delta']={k:int(v)-int(report['before']['throttles'].get(k) or 0) for k,v in report['after']['throttles'].items()}
            report['ECC_delta']={k:{q:int(v[q] or 0)-int(report['before']['ECC'].get(k,{}).get(q) or 0) for q in ['ce_count','ue_count']} for k,v in report['after']['ECC'].items()}
            for name in ['core','imc']:
                text=(folder/(name+'.csv')).read_text()
                assert '<not counted>' not in text and '<not supported>' not in text,(name,text)
            assert report['exit_code']==0 and not errors
            assert '"post_validation":true' in (folder/'stdout.txt').read_text()
            report['valid']=True
    finally:
        stop.set()
        if observer:observer.join(timeout=5)
        for c in counters:c.close()
        for p in [child,turbo]:
            if p and p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        (folder/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'done':label,'valid':report['valid'],'result':report['result'],'package_plus_DRAM_RAPL_W':report['power']['package_plus_dram_W'],'max_CPU_C':report['temperature_max_C']}),flush=True)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','main'])
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();OUT=args.output.resolve()
    with (LAB/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=True)
        if args.mode=='smoke':
            run('smoke_read_v2','read',6,'node0',0,0,64,2)
            run('smoke_copy','copy',1,'node0',1,0,64,2)
            run('smoke_latency','latency',1,'node0',0,0,32,2)
            return
        baseline=[];a=sample()
        for i in range(10):time.sleep(1);baseline.append(sample())
        b=sample();(OUT/'idle-power.json').write_text(json.dumps({'before':a,'after':b,'telemetry':baseline,'power':power(a,b)},indent=2)+'\n')
        cases=[
          ('read_n0_local','read',6,'node0',0,0,1024),('read_n0_remote','read',6,'node0',1,1,1024),
          ('read_n1_local','read',6,'node1',1,1,1024),('read_n1_remote','read',6,'node1',0,0,1024),
          ('read_both_physical','read',12,'both',-1,-1,1024),('read_both_SMT','read',24,'both',-1,-1,1024),
          ('read_n0_SMT','read',12,'node0',0,0,1024),('read_n1_SMT','read',12,'node1',1,1,1024),
          ('copy_n0_local1','copy',1,'node0',0,0,512),('copy_n0_from_remote1','copy',1,'node0',1,0,512),
          ('copy_n0_to_remote1','copy',1,'node0',0,1,512),('copy_n1_local1','copy',1,'node1',1,1,512),
          ('copy_n0_local6','copy',6,'node0',0,0,512),('copy_both12','copy',12,'both',-1,-1,512),
          ('latency_n0_local','latency',1,'node0',0,0,128),('latency_n0_remote','latency',1,'node0',1,1,128),
          ('latency_n1_local','latency',1,'node1',1,1,128),('latency_n1_remote','latency',1,'node1',0,0,128)]
        for round in range(3):
            ordered=cases if round%2==0 else cases[::-1]
            for name,*parameters in ordered:run(f'{name}_r{round}',*parameters,4)
        run('read_both_SMT_sustained','read',24,'both',-1,-1,1024,40)
        (OUT/'complete.json').write_text(json.dumps({'complete':True,'measurements':55,'no_GPU_work':True,'binary_sha256':hashlib.sha256((SRC/'memory-probe').read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
