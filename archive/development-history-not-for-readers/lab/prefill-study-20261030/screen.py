#!/usr/bin/python3
"""Serialized, uncached prefill measurements using the guarded real server."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
R29 = LAB / 'moe-engine-20261029'
BASE = json.loads((ROOT / 'baseline-config.json').read_text())
KEY = Path(BASE['api_key_file']).read_text().strip()
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
URL = 'http://127.0.0.1:8001'
LONG_INPUT = False
VARIANTS = {
    'baseline': {},
    'pair': {'GGML_XV_PREFILL_PAIR': '1'},
    'rows4': {'GGML_CPU_Q8_MOE_ROWS4': '1'},
    'token-axis': {'GGML_XV_PREFILL_TOKEN_AXIS': '1'},
    'interval1': {'GGML_XV_ADAPTIVE_INTERVAL': '1'},
    'phase': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1'},
    'swap8': {'GGML_XV_PREFILL_SWAP_GAIN':'8'},
    'swap16': {'GGML_XV_PREFILL_SWAP_GAIN':'16'},
    'timing1': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1'},
    'timing4': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1','GGML_XV_ADAPTIVE_PREFILL_PER_LAYER':'4'},
    'quality4': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1','GGML_XV_ADAPTIVE_PREFILL_PER_LAYER':'4',
                 'GGML_XV_ADAPTIVE_VERIFY':'1','GGML_XV_ADAPTIVE_AUDIT':'1'},
    'token4': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1','GGML_XV_ADAPTIVE_PREFILL_PER_LAYER':'4'},
    'token512': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1','GGML_XV_ADAPTIVE_PREFILL_PER_LAYER':'4',
                 'GGML_XV_ADAPTIVE_STRENGTH':'512'},
    'quality-token4': {'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1','GGML_XV_ADAPTIVE_PREFILL_PER_LAYER':'4',
                       'GGML_XV_ADAPTIVE_VERIFY':'1','GGML_XV_ADAPTIVE_AUDIT':'1'},
    'stream': {'GGML_XV_STREAM_ACTIVE':'1','GGML_XV_STREAM_MIB':'256',
        'GGML_XV_ADAPTIVE':'0','GGML_XV_ADAPTIVE_PIPELINE':'0','GGML_XV_MLP_CPU_PAIR':'0',
        'GGML_CPU_MOE_PAIR':'0','GGML_XV_AUTO_STEP':'1','GGML_XV_DEFER_COMMIT':'0'},
}


def request(path, payload=None, timeout=900):
    req = urllib.request.Request(URL + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Authorization': 'Bearer '+KEY, 'Content-Type':'application/json'})
    return HTTP.open(req, timeout=timeout)


def json_request(path, payload=None):
    with request(path, payload) as response:
        return json.load(response)


def proc_snapshot(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    io = {k:int(v) for k,v in (line.split(':') for line in (p/'io').read_text().splitlines())}
    return dict(minflt=int(fields[7]), majflt=int(fields[9]), io=io)


def prompt_cases():
    english = '\n'.join(f'Record {i:03}: module {i%7} handles {i*13} items and stores results in region {i%5}.' for i in range(1,301 if LONG_INPUT else 25))
    japanese = '\n'.join(f'記録{i}: この計算機には二つのCPUと一つのGPUがあります。入力をまとめて処理し、必要なデータを転送します。' for i in range(1,11))
    return [('english', english+'\nSummarize these records in one sentence.'),
            ('japanese', japanese+'\n以上の内容を日本語で一文にまとめてください。')]


def query(name, text, pid):
    payload = dict(model='glm-4.5-air-q8', messages=[{'role':'user','content':text}],
        max_tokens=16, temperature=0, seed=42, stream=True, cache_prompt=False,
        logprobs=True, top_logprobs=5, stream_options={'include_usage':True})
    before = proc_snapshot(pid)
    start = time.monotonic()
    content = ''
    first = None
    timings = usage = None
    chunks = []
    done = False
    with request('/v1/chat/completions', payload) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            item = json.loads(data)
            chunks.append(item)
            for choice in item.get('choices', []):
                value = choice.get('delta', {}).get('content') or ''
                if value and first is None:
                    first = time.monotonic()-start
                content += value
            timings = item.get('timings') or timings
            usage = item.get('usage') or usage
    after = proc_snapshot(pid)
    assert done and content and timings and timings['cache_n']==0, (done,content,timings)
    result = dict(name=name, first_content_s=first, total_s=time.monotonic()-start,
                  content=content, timings=timings, usage=usage, chunks=chunks,
                  majflt=after['majflt']-before['majflt'],
                  disk_read_bytes=after['io']['read_bytes']-before['io']['read_bytes'])
    print(json.dumps({k:v for k,v in result.items() if k!='chunks'},ensure_ascii=False), flush=True)
    return result


def run_case(label, variant):
    folder = ROOT / 'runs' / label
    folder.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(BASE)
    config['environment'].update(VARIANTS[variant])
    config['environment']['GGML_XV_MEASURE_PHASES'] = '1'
    if variant in ('token4','token512','quality-token4'):
        old=str(R29/'variants/hybrid-reuse-v1/runtime')
        new=str(ROOT/'variants/token-history/runtime')
        config['environment']={k:v.replace(old,new) for k,v in config['environment'].items()}
        config['command']=[v.replace(old,new) for v in config['command']]
    if variant in ('timing1','timing4','quality4'):
        old=str(R29/'variants/hybrid-reuse-v1/runtime')
        new=str(ROOT/'variants/prefill-timing/runtime')
        config['environment']={k:v.replace(old,new) for k,v in config['environment'].items()}
        config['command']=[v.replace(old,new) for v in config['command']]
    if variant in ('phase','swap8','swap16'):
        old=str(R29/'variants/hybrid-reuse-v1/runtime')
        new=str(ROOT/'variants/simple-prefill/runtime')
        config['environment']={k:v.replace(old,new) for k,v in config['environment'].items()}
        config['command']=[v.replace(old,new) for v in config['command']]
    if variant == 'stream':
        old=str(R29/'variants/hybrid-reuse-v1/runtime')
        new=str(R29/'experiments/stream-active-agent/v2/variant/runtime')
        config['environment']={k:v.replace(old,new) for k,v in config['environment'].items()}
        config['command']=[v.replace(old,new) for v in config['command']]
        j=next(i for i,x in enumerate(config['command']) if x.startswith('LD_PRELOAD='))
        config['command'].insert(j+1,'GOMP_SPINCOUNT=0')
    (ROOT/'server-config.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    log = ROOT/'server.stderr.log'
    offset = log.stat().st_size if log.exists() else 0
    report = dict(label=label,variant=variant,started_unix=time.time(),requests=[])
    try:
        subprocess.run(['systemctl','start','llm-prefill-lab.service'],check=True)
        for _ in range(150):
            try:
                if json_request('/health').get('status')=='ok':
                    break
            except (OSError,ValueError):
                pass
            state=subprocess.check_output(['systemctl','show','llm-prefill-lab.service','-p','ActiveState'],text=True)
            if 'failed' in state:
                raise RuntimeError(state)
            time.sleep(2)
        else:
            raise RuntimeError('Health timeout')
        health=json.loads((ROOT/'health.json').read_text())
        pid=health['pid']
        report['warmup']=query('warmup','Reply with the single word READY.',pid)
        for name,text in prompt_cases():
            report['requests'].append(query(name,text,pid))
            (folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        report['success']=True
    finally:
        subprocess.run(['systemctl','stop','llm-prefill-lab.service'],check=True)
        raw=log.read_bytes()[offset:] if log.exists() else b''
        (folder/'stderr.log').write_bytes(raw)
        if (ROOT/'health.json').exists():
            report['hardware']=json.loads((ROOT/'health.json').read_text())
        report['stats']={}
        for line in raw.decode(errors='replace').splitlines():
            if line.startswith('XV_') and ' {' in line:
                key,value=line.split(' ',1)
                try:report['stats'].setdefault(key,[]).append(json.loads(value))
                except ValueError:pass
        report['finished_unix']=time.time()
        (folder/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert not (ROOT/'STOP_REQUESTED').exists(), 'Hardware stop; inspect before continuing'
    print('Completed',label,'phase intervals',json.dumps(report['stats'].get('XV_INTERVAL')),flush=True)
    return report


def main():
    global LONG_INPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('variants',nargs='+',choices=list(VARIANTS))
    parser.add_argument('--prefix',default='screen')
    parser.add_argument('--long',action='store_true')
    args=parser.parse_args()
    LONG_INPUT=args.long
    assert os.geteuid()==0
    state=subprocess.run(['systemctl','is-active','llm-openai-api.service'],text=True,capture_output=True)
    assert state.stdout.strip()=='inactive',state.stdout
    for index,variant in enumerate(args.variants):
        label=args.prefix+'-'+variant
        if len(set(args.variants)) != len(args.variants):label=args.prefix+'-'+str(index)+'-'+variant
        run_case(label,variant)


if __name__=='__main__':
    main()
