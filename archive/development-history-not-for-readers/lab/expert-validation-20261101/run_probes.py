#!/usr/bin/env python3
"""Bounded operator checks with existing libraries; no model start or service edits."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
CONFIG = LAB / 'deepseek41-bringup-20261101/server-config.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke', 'decode', 'prefill-hot', 'prefill-spread', 'prefill128-hot'])
    args = parser.parse_args()
    lock = (ROOT / '.probe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(CONFIG.read_text())
    key = Path(config['api_key_file']).read_text().strip()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check_idle():
        req = urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",
                                     headers={'Authorization': 'Bearer ' + key})
        with http.open(req, timeout=10) as response:
            slots = json.load(response)
        if any(s.get('is_processing') for s in slots):
            raise RuntimeError('API has an active request; diagnostic stopped')

    def check_watch():
        state = json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
        if time.time()-state['unix'] > 45 or state['errors'] or state['active_alerts']:
            raise RuntimeError('Existing hardware monitor reports stale data or an issue')
        gpu = state['gpu']
        if gpu['ecc_mode'] != 'Enabled' or abs(gpu['power_limit_W']-150) > .5:
            raise RuntimeError('GPU is outside the established operating policy')
        mem = {l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')}
        if mem['MemAvailable'] < 8*1024**2:
            raise RuntimeError('Insufficient host memory headroom')

    check_watch()
    check_idle()
    free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 4096:
        raise RuntimeError('Less than 4GiB free GPU memory for bounded diagnostic')
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(('LLAMA_', 'GGML_', 'OMP_', 'GOMP_')) or k in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']:
            env.pop(k)
    env.update(config['environment'])
    cases = json.loads((ROOT/'cases.json').read_text())
    if args.phase == 'smoke':
        cases = cases[:1]
    if args.phase == 'prefill128-hot':
        cases = [c for c in cases if c['type'] in ['IQ3_XXS', 'MXFP4']]
    tokens = 128 if args.phase == 'prefill128-hot' else (16 if args.phase.startswith('prefill') else 1)
    pattern = 'spread' if args.phase.endswith('spread') else 'hot'
    threads = 24 if tokens > 1 else 12
    out = ROOT/'results'/(time.strftime('%Y%m%d-%H%M%S')+'-'+args.phase)
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps({'phase':args.phase,'tokens':tokens,'threads':threads,
        'pattern':pattern,'binary_sha256':hashlib.sha256((ROOT/'expert_probe').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((ROOT/'expert_probe.cpp').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        'environment_overrides':config['environment'],'api_restart':False},indent=2)+'\n')
    rows = []
    for case in cases:
        check_watch()
        check_idle()
        command = ['numactl', '--physcpubind=0-23', '--interleave=0,1',
                   str(ROOT/'expert_probe'), case['shard'], case['tensor'],
                   str(tokens), pattern, str(threads), '42']
        started = time.monotonic()
        with (out/(case['name']+'.stdout')).open('w') as stdout, (out/(case['name']+'.stderr')).open('w') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            try:
                deadline = started + 90
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Per-case 90second bound exceeded')
                    check_watch()
                    check_idle()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        raw = (out/(case['name']+'.stdout')).read_text()
        result = json.loads(raw) if raw.strip().startswith('{') else None
        row = dict(case=case, result=result, exit_code=child.returncode, elapsed_s=time.monotonic()-started)
        rows.append(row)
        (out/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
        concise = {k:result.get(k) for k in ['pass','cpu_ms','gpu_resident_ms','pageable_upload_compute_ms','staged_upload_compute_ms','cpu_gpu_nmse']} if result else None
        print(json.dumps({'case':case['name'],'phase':args.phase,'exit_code':child.returncode,'result':concise}),flush=True)
        if child.returncode or not result or not result.get('pass'):
            raise RuntimeError('Operator check failed; see '+str(out))
    print('Saved:', out,flush=True)


if __name__ == '__main__':
    main()
