#!/usr/bin/env python3
"""Short synthetic requests to an already-running model; never starts/stops it."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import select
import signal
import struct
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parent;LAB=R.parent
CONFIG=LAB/'deepseek41-bringup-20261101/server-config.json'
WATCH=Path('/var/lib/xeon-watch/latest.json')

def watch():
    j=json.loads(WATCH.read_text())
    if time.time()-j['unix']>45 or j['errors'] or j['active_alerts']:raise RuntimeError('Hardware monitor stale or alarm')
    if j['gpu']['ecc_mode']!='Enabled' or abs(j['gpu']['power_limit_W']-150)>.5:raise RuntimeError('GPU policy mismatch')
    return {'unix':j['unix'],'cpu':j['cpu'],'gpu':j['gpu'],'host_counters':j.get('host_counters')}

def api_pid():
    pids=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if (p/'comm').read_text().strip()=='llama-server':pids.append(int(p.name))
        except OSError:pass
    if len(pids)!=1:raise RuntimeError('Expected exactly one live llama-server')
    return pids[0]

class Perf:
    def __init__(self,pid,folder,mode,events):
        self.folder=folder;self.mode=mode;self.controls=[]
        cr,self.cw=os.pipe();self.ar,aw=os.pipe()
        self.log=(folder/'perf-record.log').open('w')
        cmd=['perf','record','-q','-p',str(pid),'-D','-1','--control',f'fd:{cr},{aw}','--clockid','mono',
             '-m','1024','-o',str(folder/'perf.data')]
        if mode=='trace-sample':events=events+['cpu-clock/period=10000000/']
        cmd+=['-e',','.join(events)] if mode.startswith('trace') else ['-e','cpu-clock','-F','99']
        self.cmd=cmd;self.process=subprocess.Popen(cmd,pass_fds=(cr,aw),stdout=self.log,stderr=self.log)
        os.close(cr);os.close(aw)
        try:self.control('disable')
        except BaseException:
            self.close();raise
    def control(self,command):
        start=time.monotonic();os.write(self.cw,(command+'\n').encode())
        if not select.select([self.ar],[],[],10)[0]:raise RuntimeError('perf control timeout')
        answer=os.read(self.ar,256)
        if b'ack' not in answer:raise RuntimeError('perf ACK missing: '+repr(answer))
        self.controls.append({'command':command,'sent':start,'ack':time.monotonic()})
    def close(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        os.close(self.cw);os.close(self.ar);self.log.close()
        (self.folder/'perf-control.json').write_text(json.dumps({'command':self.cmd,'controls':self.controls,'exit':self.process.returncode},indent=2)+'\n')

def probes(mode):
    engine=LAB/'deepseek41-bringup-20261101';runtime=LAB/'gpu-stream-20261103/runtime'
    cuda=LAB/'v100-prep-20261024/cuda-12.9.2/lib/libcudart.so.12.9.79'
    items=[('decode',runtime/'libllama.so.0.8.0','_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_status','n_tokens=+4(%si):u32',True),
           ('cpu',runtime/'libggml-cpu.so.0.28.0','ggml_graph_compute','nodes=+4(%di):u32 first=+256(+0(+16(%di))):string',True),
           ('copy',runtime/'libggml-base.so.0.28.0','ggml_backend_tensor_copy','src=+256(%di):string dst=+256(%si):string',True),
           ('memcpy',cuda,'cudaMemcpyAsync','bytes=%dx:u64 kind=%cx:u32 stream=%r8:x64',False),
           ('memcpy2d',cuda,'cudaMemcpy2DAsync','width=%r8:u64 height=%r9:u64 kind=+8(%sp):u32 stream=+16(%sp):x64',False)]
    if mode in ['trace','trace-sample']:items += [
        ('copy_input',runtime/'libggml-base.so.0.28.0','_ZL29ggml_backend_sched_copy_inputP18ggml_backend_schedP24ggml_backend_sched_splitP11ggml_tensor','name=+256(%dx):string',True),
        ('gpu',engine/'build/bin/libggml-cuda.so.0.28.0','_ZL31ggml_backend_cuda_graph_computeP12ggml_backendP11ggml_cgraph','',True),
        ('sync_stream',cuda,'cudaStreamSynchronize','stream=%di:x64',True),
        ('sync_event',cuda,'cudaEventSynchronize','event=%di:x64',True),
        ('sync_device',cuda,'cudaDeviceSynchronize','',True),
        ('wait_event',cuda,'cudaStreamWaitEvent','stream=%di:x64 event=%si:x64',False),
        ('launch_graph',cuda,'cudaGraphLaunch','stream=%si:x64',False)]
    return items

def symbol_file_offset(library,symbol):
    addresses=set()
    for options in [['-an'],['-D','--defined-only']]:
        p=subprocess.run(['nm',*options,str(library)],capture_output=True,text=True,check=True)
        for line in p.stdout.splitlines():
            fields=line.split()
            if len(fields)>=3 and fields[-1].split('@',1)[0]==symbol:
                addresses.add(int(fields[0],16))
        if addresses:break
    if len(addresses)!=1:raise RuntimeError('Ambiguous or missing ELF symbol: '+symbol)
    address=addresses.pop()
    with library.open('rb') as f:
        header=f.read(64)
        if header[:6]!=b'\x7fELF\x02\x01':raise RuntimeError('Expected ELF64 little endian')
        phoff=struct.unpack_from('<Q',header,32)[0]
        phsize,phcount=struct.unpack_from('<HH',header,54)
        for i in range(phcount):
            f.seek(phoff+i*phsize);ph=f.read(phsize)
            kind,flags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',ph)
            if kind==1 and vaddr<=address<vaddr+filesz:return offset+address-vaddr
    raise RuntimeError('ELF symbol outside file-backed LOAD segment')

def tracefs():
    for path in ['/sys/kernel/tracing/uprobe_events','/sys/kernel/debug/tracing/uprobe_events']:
        p=Path(path)
        if p.exists():return p
    raise RuntimeError('uprobe tracefs unavailable')

def write_probe(path,definition):
    # tracefs command files reject the seek used by Python's append-mode stream.
    data=(definition+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_APPEND)
    try:
        if os.write(fd,data)!=len(data):raise RuntimeError('Incomplete tracefs command')
    finally:os.close(fd)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['baseline','trace','trace-lite','trace-sample','sample'],required=True)
    ap.add_argument('--tokens',type=int,default=96);ap.add_argument('--context-tokens',type=int,default=0);a=ap.parse_args()
    if not 8<=a.tokens<=128:ap.error('bounded test permits8..128 tokens')
    if a.context_tokens and not 128<=a.context_tokens<=4096:ap.error('context must be128..4096 or0')
    lock=(R/'.study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(CONFIG.read_text());key=Path(config['api_key_file']).read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url=f"http://{config['host']}:{config['port']}"
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    def get(path):
        with opener.open(urllib.request.Request(url+path,headers=headers),timeout=10) as r:return json.load(r)
    if get('/health').get('status')!='ok':raise RuntimeError('API not ready')
    if any(s.get('is_processing') for s in get('/slots')):raise RuntimeError('Another request is active')
    pid=api_pid();initial=watch()
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
    folder=R/'runs'/f'{stamp}-{a.mode}-n{a.tokens}-ctx{a.context_tokens}';folder.mkdir(parents=True,exist_ok=False)
    body={'prompt':'Complete this increasing list of positive integers, using commas: 1, 2, 3, 4, 5,',
          'n_predict':a.tokens,'temperature':0,'seed':42,'ignore_eos':True,'cache_prompt':False,'stream':True,'return_tokens':True}
    if a.context_tokens:
        def tokenize(content,special):
            req=urllib.request.Request(url+'/tokenize',data=json.dumps({'content':content,'add_special':special}).encode(),headers=headers)
            with opener.open(req,timeout=15) as r:return json.load(r)['tokens']
        prefix=tokenize('Synthetic background notes:',True);tail=tokenize('\n'+body['prompt'],False);filler=tokenize(' a',False)
        n=a.context_tokens-len(prefix)-len(tail)
        if n<0 or not filler:raise RuntimeError('Invalid synthetic token construction')
        body['prompt']=prefix+(filler*((n+len(filler)-1)//len(filler)))[:n]+tail
    record={'pid':pid,'mode':a.mode,'request':body,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            'initial_watch':initial,'perf_probes':[],'stream_events':[],'monitor_samples':[],'error':None}
    (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    perf=None;installed=[];group=f'xd{pid}';probe_file=None;monitor_stop=threading.Event();monitor_errors=[];response=None
    def observe():
        while not monitor_stop.wait(1):
            try:
                state=watch();record['monitor_samples'].append(state)
                if state['host_counters']!=initial['host_counters']:raise RuntimeError('ECC/throttle counter change')
            except Exception as e:monitor_errors.append(str(e));return
    observer=threading.Thread(target=observe,daemon=True)
    try:
        if a.mode.startswith('trace'):
            probe_file=tracefs();existing=probe_file.read_text()
            if group+'/' in existing:raise RuntimeError('Probe group already exists; inspect before reusing')
            for name,library,symbol,arguments,has_return in probes(a.mode):
                offset=symbol_file_offset(library,symbol)
                for ret in ([False,True] if has_return else [False]):
                    event=group+':'+name+('_ret' if ret else '')
                    definition=('r:' if ret else 'p:')+event.replace(':','/')+' '+str(library)+':0x'+format(offset,'x')+(' '+arguments if arguments and not ret else '')
                    write_probe(probe_file,definition)
                    record['perf_probes'].append({'definition':definition,'symbol':symbol,'library':str(library),'offset':offset})
                    installed.append(event)
            perf=Perf(pid,folder,a.mode,installed)
        elif a.mode=='sample':perf=Perf(pid,folder,a.mode,[])
        observer.start();start=time.monotonic();record['request_start_monotonic']=start
        response=opener.open(urllib.request.Request(url+'/completion',data=json.dumps(body).encode(),headers=headers),timeout=180)
        text='';token_ids=[];first=None;final=None
        for raw in response:
            if monitor_errors:raise RuntimeError(monitor_errors[0])
            if time.monotonic()-start>180:raise RuntimeError('request time bound')
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':break
            event=json.loads(payload);now=time.monotonic()
            if event.get('error'):raise RuntimeError('API streaming error')
            content=event.get('content','');text+=content;token_ids.extend(event.get('tokens') or [])
            record['stream_events'].append({'monotonic':now,'stop':event.get('stop',False),'content_chars':len(content),'tokens_in_event':len(event.get('tokens') or [])})
            if first is None and (content or event.get('tokens')):
                first=now;record['first_content_monotonic']=first
                if perf:perf.control('enable')
            if event.get('stop'):
                final={k:v for k,v in event.items() if k in ['timings','tokens_predicted','tokens_evaluated','tokens_cached','stop_type','stopped_eos','stopped_limit']}
                if perf:perf.control('disable')
                break
        if final is None:raise RuntimeError('No final completion event')
        record.update(final=final,elapsed_seconds=time.monotonic()-start,output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      output_chars=len(text),returned_token_count=len(token_ids),token_ids_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest())
    except BaseException as error:record['error']=repr(error);raise
    finally:
        if response:response.close()
        monitor_stop.set()
        if observer.is_alive():observer.join(timeout=3)
        if perf:perf.close()
        cleanup=[]
        for event in reversed(installed):
            try:
                write_probe(probe_file,'-:'+event.replace(':','/'))
                cleanup.append({'event':event,'removed':True})
            except OSError as error:cleanup.append({'event':event,'removed':False,'error':str(error)})
        record['probe_cleanup']=cleanup
        (folder/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'saved':str(folder),'mode':a.mode,'final':record['final'],'elapsed_s':record['elapsed_seconds'],'output_sha256':record['output_sha256']}),flush=True)

if __name__=='__main__':main()
