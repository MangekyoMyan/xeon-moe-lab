#!/usr/bin/env python3
"""Bounded prefill timing on the loaded API; no model startup or runtime edits."""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, json, os, signal, subprocess, time, urllib.request

R = Path(__file__).resolve().parent
L = R.parent
E = L/'deepseek41-bringup-20261101'
spec = importlib.util.spec_from_file_location('previous_probe', L/'decode-observation-20261103/live_probe.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['baseline', 'trace', 'trace-deep', 'sample'])
    ap.add_argument('--tokens', type=int, default=2048)
    a = ap.parse_args()
    if not 1024 <= a.tokens <= 4096:
        raise RuntimeError('1024..4096 tokens only')
    R.mkdir(exist_ok=True)
    with (R/'.live.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((E/'server-config.json').read_text())
        key = Path(config['api_key_file']).read_text().strip()
        headers = {'Authorization':'Bearer '+key, 'Content-Type':'application/json'}
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f"http://{config['host']}:{config['port']}"
        def request(path, body=None, timeout=10):
            return op.open(urllib.request.Request(url+path, headers=headers,
                data=None if body is None else json.dumps(body).encode()), timeout=timeout)
        with request('/health') as f:
            if json.load(f).get('status') != 'ok': raise RuntimeError('API not ready')
        with request('/slots') as f:
            if any(s.get('is_processing') for s in json.load(f)): raise RuntimeError('API busy')
        pid = old.api_pid()
        initial = old.watch()
        maps = Path(f'/proc/{pid}/maps').read_text().splitlines()
        def library(stem):
            paths = {Path(line.split()[-1]) for line in maps if '/' in line and Path(line.split()[-1]).name.startswith(stem+'.so')}
            if len(paths) != 1: raise RuntimeError('Unexpected mapped library '+stem+': '+repr(paths))
            return paths.pop()
        libs = {x:library(x) for x in ['libllama','libggml-cpu','libggml-cuda','libggml-base','libcudart']}
        # Diverse synthetic notes; no user conversation is read or recorded.
        topics = [
            'Astronomy: compare the orbital periods of planets and explain why distance affects the year.',
            'Biology: describe how roots absorb water and how leaves exchange gases during photosynthesis.',
            'Software: write a Python function to merge two sorted lists without modifying either input.',
            'History: explain how printing affected the distribution of scientific knowledge in Europe.',
            'Mathematics: derive the sum of an arithmetic progression and check a numerical example.',
            'Cooking: distinguish steaming, boiling, baking and frying by heat and moisture transfer.',
            'Networking: contrast a packet route, a DNS lookup and an encrypted connection handshake.',
            'Economics: explain opportunity cost without assuming that market prices capture every cost.',
            'Geology: explain the differences between sedimentary, igneous and metamorphic rocks.',
            'Music: compare rhythm, melody, harmony and timbre using concrete listening examples.',
            '日本語の文章を読み、主語と述語の関係、および条件と結論の違いを説明してください。',
            'Consider a queue with arrivals and departures; calculate the remaining items after each event.',
        ]
        prompt = 'These are independent study notes. Summarize the main themes briefly.\n'
        for n in range(170):
            prompt += f'Note {n+1}: {topics[n%len(topics)]} Example values: {17*n+3}, {11*n+29}, {n*n+101}. Explain assumptions and exceptions.\n'
        with request('/tokenize', {'content':prompt,'add_special':True}, 15) as f:
            tokens = json.load(f)['tokens'][:a.tokens]
        if len(tokens) != a.tokens: raise RuntimeError('Synthetic input too short')
        folder = R/'live'/f'{time.time_ns()}-{a.mode}-n{a.tokens}'
        folder.mkdir(parents=True)
        record = {'pid':pid,'mode':a.mode,'input_tokens':len(tokens),'input_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'config_sha256':hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest(),
            'initial_watch':initial,'mapped_libraries':{k:str(v) for k,v in libs.items()},'error':None}
        (folder/'synthetic-input.json').write_text(json.dumps(tokens)+'\n')
        probes = [
            ('ubatch',libs['libllama'],'_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_status','n_tokens=+4(%si):u32',True),
            ('cpu',libs['libggml-cpu'],'ggml_graph_compute','nodes=+4(%di):u32 first=+256(+0(+16(%di))):string',True),
            ('copy',libs['libggml-base'],'ggml_backend_tensor_copy','src=+256(%di):string dst=+256(%si):string',True),
            ('gpu',libs['libggml-cuda'],'_ZL31ggml_backend_cuda_graph_computeP12ggml_backendP11ggml_cgraph','',True),
            ('mmid',libs['libggml-cuda'],'_ZL20ggml_cuda_mul_mat_idR25ggml_backend_cuda_contextP11ggml_tensor','',True),
            ('memcpy',libs['libcudart'],'cudaMemcpyAsync','bytes=%dx:u64 kind=%cx:u32 stream=%r8:x64',False),
            ('memcpy2d',libs['libcudart'],'cudaMemcpy2DAsync','width=%r8:u64 height=%r9:u64 kind=+8(%sp):u32 stream=+16(%sp):x64',False),
            ('sync_stream',libs['libcudart'],'cudaStreamSynchronize','stream=%di:x64',True),
            ('sync_event',libs['libcudart'],'cudaEventSynchronize','event=%di:x64',True),
            ('sync_device',libs['libcudart'],'cudaDeviceSynchronize','',True),
        ]
        if a.mode=='trace-deep':
            probes=[(n,l,s,a,True if n in ['memcpy','memcpy2d'] else ret) for n,l,s,a,ret in probes]
            probes += [('launch',libs['libcudart'],'cudaLaunchKernel','',True),
                       ('memset',libs['libcudart'],'cudaMemsetAsync','bytes=%dx:u64',True)]
        installed=[]; perf=None; dmon=None; response=None; pf=None
        group=f'xgp{pid}_{os.getpid()}'
        try:
            if a.mode.startswith('trace'):
                pf=old.tracefs()
                if group+'/' in pf.read_text():raise RuntimeError('Probe group already exists')
                record['probes']=[]
                for name,lib,symbol,arguments,ret in probes:
                    offset=old.symbol_file_offset(lib,symbol)
                    for returning in ([False,True] if ret else [False]):
                        event=group+':'+name+('_ret' if returning else '')
                        definition=('r:' if returning else 'p:')+event.replace(':','/')+' '+str(lib)+':0x'+format(offset,'x')+(' '+arguments if arguments and not returning else '')
                        old.write_probe(pf,definition);installed.append(event)
                        record['probes'].append({'definition':definition,'symbol':symbol})
                perf=old.Perf(pid,folder,'trace',installed)
            elif a.mode=='sample': perf=old.Perf(pid,folder,'sample',[])
            with (folder/'dmon.txt').open('w') as df:
                dmon=subprocess.Popen(['nvidia-smi','dmon','-i',config['environment']['CUDA_VISIBLE_DEVICES'],
                    '-s','pucvmet','-d','1','-c','180','-o','DT'],stdout=df,stderr=subprocess.STDOUT)
                body={'prompt':tokens,'n_predict':1,'temperature':0,'seed':42,'ignore_eos':True,
                      'cache_prompt':False,'stream':True,'return_tokens':True}
                record['start_unix']=time.time();record['start_monotonic']=time.monotonic()
                if perf:perf.control('enable')
                response=request('/completion',body,180)
                text='';final=None
                for raw in response:
                    if time.monotonic()-record['start_monotonic']>180:raise RuntimeError('Request bound exceeded')
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    value=line[5:].strip()
                    if value=='[DONE]':break
                    event=json.loads(value)
                    if event.get('error'):raise RuntimeError('API streaming error')
                    text+=event.get('content','')
                    if event.get('stop'):
                        record['end_monotonic']=time.monotonic();record['end_unix']=time.time()
                        if perf:perf.control('disable')
                        final={k:v for k,v in event.items() if k in ['timings','tokens_predicted','tokens_evaluated','tokens_cached','stop_type']}
                        break
                if final is None:raise RuntimeError('No completion timing')
                record['final']=final;record['output_sha256']=hashlib.sha256(text.encode()).hexdigest()
                record['final_watch']=old.watch()
                if record['final_watch']['host_counters']!=initial['host_counters']:raise RuntimeError('Host error counters changed')
        except BaseException as exc:
            record['error']=repr(exc);raise
        finally:
            if response:response.close()
            if dmon and dmon.poll() is None:
                dmon.terminate()
                try:dmon.wait(timeout=5)
                except subprocess.TimeoutExpired:dmon.kill();dmon.wait()
            if perf:perf.close()
            cleanup=[]
            for event in reversed(installed):
                try:old.write_probe(pf,'-:'+event.replace(':','/'));cleanup.append({'event':event,'removed':True})
                except OSError as error:cleanup.append({'event':event,'removed':False,'error':str(error)})
            record['cleanup']=cleanup
            record['config_unchanged']=hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest()==record['config_sha256']
            (folder/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'folder':str(folder),'mode':a.mode,'final':record.get('final'),'error':record['error']}),flush=True)


if __name__=='__main__':main()
