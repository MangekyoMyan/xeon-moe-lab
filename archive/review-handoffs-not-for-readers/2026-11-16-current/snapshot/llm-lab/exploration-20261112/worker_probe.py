#!/usr/bin/env python3
"""Bounded synthetic requests; no model startup, real conversations, or credentials in output."""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, json, socket, sys, threading, time, urllib.request

R = Path(__file__).resolve().parent
LAB = R.parent
CONFIG = LAB / 'deepseek41-bringup-20261101/server-config.json'
sys.path.insert(0, str(LAB))
from xeon_imc import ImcReader
spec = importlib.util.spec_from_file_location('historical_probe', LAB/'postcompact-20261103/live_probe.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--mode', choices=['workers'], default='workers')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--workload', choices=['count','code','japanese'], default='count')
    a=p.parse_args()
    assert 8 <= a.tokens <= 128 and 64 <= a.context <= 4096
    assert a.label.replace('-','').replace('_','').isalnum()
    lock=(R/'.benchmark.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes(); config=json.loads(raw)
    key=Path(config['api_key_file']).read_text().strip()
    url=f"http://{config['host']}:{config['port']}"
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def query(path,body=None):
        req=urllib.request.Request(url+path,headers=headers,data=None if body is None else json.dumps(body).encode())
        with opener.open(req,timeout=15) as res:return json.load(res)
    if query('/health').get('status')!='ok':raise RuntimeError('API not ready')
    if any(s.get('is_processing') for s in query('/slots')):raise RuntimeError('Another API request is active')
    pid=previous.api_pid(); initial=previous.watch()
    folder=R/'runs'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}-{a.workload}-{a.mode}-c{a.context}-n{a.tokens}'
    folder.mkdir(parents=True,exist_ok=False)
    instruction={
      'count':'Complete this increasing list of positive integers, using commas: 1, 2, 3, 4, 5,',
      'code':'Write a Python function that merges two sorted integer lists. Include a concise example.\n```python\n',
      'japanese':'以下の説明を続けてください。CPUとGPUが分担して計算する場合、まず考えるべきことは',
    }[a.workload]
    def tokens(text,special=False):return query('/tokenize',{'content':text,'add_special':special})['tokens']
    prefix=tokens('Synthetic benchmark background notes:',True)
    tail=tokens('\n'+instruction)
    filler=tokens(' The system stores numbers in memory and processes a sequence of tasks.')
    count=a.context-len(prefix)-len(tail)
    if count<0 or not filler:raise RuntimeError('Invalid synthetic prompt length')
    prompt=prefix+(filler*((count+len(filler)-1)//len(filler)))[:count]+tail
    body={'prompt':prompt,'n_predict':a.tokens,'temperature':0,'seed':42,'ignore_eos':True,
          'cache_prompt':False,'stream':True,'return_tokens':True}
    libs=sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                 if 'libggml' in line or 'libllama' in line})
    record={'label':a.label,'mode':a.mode,'pid':pid,'workload':a.workload,
      'context':a.context,'tokens':a.tokens,'request':body,'config_sha256':hashlib.sha256(raw).hexdigest(),
      'libraries':libs,'initial_watch':initial,'samples':[],'events':[],'error':None,
      'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Worker placement observation; do not use timings as uninstrumented performance'}
    stop=threading.Event(); failures=[]; response=None; perf=None; imc=ImcReader(); phase=['prefill']
    installed=[]; probe_file=None
    imc_lock=threading.Lock()
    def observe():
        while not stop.wait(2):
            try:
                state=previous.watch()
                if state['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counter change')
                with imc_lock: traffic=imc.sample()
                record['samples'].append({'monotonic':time.monotonic(),'phase':phase[0],'watch':state,'imc':traffic})
            except Exception as e:
                failures.append(str(e)); return
    worker=threading.Thread(target=observe,daemon=True)
    output=[]; output_tokens=[]; final=None; first=None
    try:
        cpu_paths=[Path(x) for x in libs if Path(x).name=='libggml-cpu.so.0.28.0']
        if len(cpu_paths)!=1:raise RuntimeError('Ambiguous CPU provider')
        library=cpu_paths[0];sha=hashlib.sha256(library.read_bytes()).hexdigest()
        if sha!='55d8f322ca5f45a95cdda07674f396554c568291cb91beeef3884f3a2051d859':raise RuntimeError('Worker probe requires the audited exact binary')
        offset=previous.symbol_file_offset(library,'ggml_graph_compute_thread.isra.0')+0xe7
        with library.open('rb') as f:
            f.seek(offset)
            if f.read(7)!=bytes.fromhex('488d8108010000'):raise RuntimeError('Probe instruction fingerprint mismatch')
        probe_file=previous.tracefs();group=f'xw{pid}_{int(time.time())%100000}'
        event=group+':worker';definition='p:'+event.replace(':','/')+' '+str(library)+':0x'+format(offset,'x')+' ith=+512(%sp):u32 nth=+516(%sp):u32'
        previous.write_probe(probe_file,definition);installed.append(event)
        record['probes']=[{'definition':definition,'library_sha256':sha}]
        perf=previous.Perf(pid,folder,'trace',installed)
        worker.start(); start=time.monotonic(); record['start_monotonic']=start
        response=opener.open(urllib.request.Request(url+'/completion',data=json.dumps(body).encode(),headers=headers),timeout=180)
        for raw_line in response:
            if failures:raise RuntimeError(failures[0])
            if time.monotonic()-start>240:raise RuntimeError('Synthetic request exceeded time bound')
            line=raw_line.decode().strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':break
            event=json.loads(payload); now=time.monotonic()
            if event.get('error'):raise RuntimeError('API error event')
            text=event.get('content',''); ids=event.get('tokens') or []
            output.append(text); output_tokens.extend(ids)
            record['events'].append({'time':now,'chars':len(text),'token_count':len(ids),'stop':event.get('stop',False)})
            if first is None and (text or ids):
                first=now; phase[0]='decode'
                with imc_lock: imc.sample()
                if perf:perf.control('enable')
            if event.get('stop'):
                if perf:perf.control('disable')
                final={k:v for k,v in event.items() if k in ['timings','tokens_predicted','tokens_evaluated','tokens_cached','stop_type','stopped_eos','stopped_limit']}
                break
        if final is None:raise RuntimeError('Missing completion event')
        record.update(final=final,elapsed_s=time.monotonic()-start,ttft_s=None if first is None else first-start,
                      output_sha256=hashlib.sha256(''.join(output).encode()).hexdigest(),
                      output_token_sha256=hashlib.sha256(json.dumps(output_tokens).encode()).hexdigest(),
                      output_token_count=len(output_tokens))
        if failures:raise RuntimeError(failures[0])
    except BaseException as e:
        record['error']=repr(e); raise
    finally:
        if response:response.close()
        stop.set()
        if worker.is_alive():worker.join(timeout=3)
        imc.close()
        if perf:perf.close()
        record['probe_cleanup']=[]
        for event in reversed(installed):
            try:
                previous.write_probe(probe_file,'-:'+event.replace(':','/'))
                record['probe_cleanup'].append({'event':event,'removed':True})
            except OSError as e:
                record['probe_cleanup'].append({'event':event,'removed':False,'error':str(e)})
        (folder/'RESULT.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'saved':str(folder),'timings':final.get('timings'),'ttft_s':record['ttft_s'],
                      'output_sha256':record['output_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
