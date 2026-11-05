#!/usr/bin/env python3
"""Resume the interrupted CPU validation after the model cache is warm."""
import json
from pathlib import Path
import subprocess
import sys
import time

lab = Path(__file__).resolve().parent
boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
deadline = time.monotonic() + 1800
while True:
    reports = sorted((lab / 'results').glob('model-prewarm-*.json'))
    state = json.loads(reports[-1].read_text()) if reports else {}
    if state.get('boot_id') == boot_id and state.get('completed'):
        print('Model cache is warm; starting sequential CPU validation.', flush=True)
        break
    if time.monotonic() > deadline:
        raise SystemExit('Timed out waiting for prewarm.')
    time.sleep(5)

for threads in (12, 24):
    label = f'short64-ssse3-threads{threads}'
    command = [sys.executable, str(lab / 'bench_deepseek.py'),
               '--bench', str(lab / 'build-avx-unpack/bin/llama-bench'),
               '--label', label, '--placement', 'pinned_interleave', '--load-mode', 'mmap',
               '--threads', str(threads), '--batch', '512', '--ubatch', '512',
               '--prompt', '64', '--tokens', '0', '--repetitions', '2', '--timeout', '600']
    print(json.dumps({'starting': label}), flush=True)
    with (lab / 'results' / f'{label}-driver.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'completed': label}), flush=True)

for name, build, threads_batch, batch in (
    ('long971-baseline-resumed', 'build-native-noblas', 12, 256),
    ('long971-ssse3-resumed', 'build-avx-unpack', 24, 512),
):
    command = [sys.executable, str(lab / 'verify_deepseek.py'),
               '--binary', str(lab / build / 'bin/llama-completion'), '--label', name,
               '--threads', '12', '--threads-batch', str(threads_batch),
               '--batch', str(batch), '--ubatch', str(batch), '--note-repetitions', '48']
    print(json.dumps({'starting': name}), flush=True)
    with (lab / 'results' / f'{name}-driver.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'completed': name}), flush=True)
