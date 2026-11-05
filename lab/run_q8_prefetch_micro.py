#!/usr/bin/env python3
"""Screen Q8 prefetch distances sequentially; this is not an LLM speed claim."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

lab = Path(__file__).resolve().parent
root = lab / 'q8-prefetch-20261019'
out = lab / 'results/optimization-memory-sync-20261019/q8-micro'
out.mkdir(exist_ok=False)
source = lab / 'llama-private'
command = ['cc', '-std=gnu11', '-O3', '-march=native', '-fopenmp',
    '-I' + str(source / 'ggml/include'), '-I' + str(source / 'ggml/src'),
    str(lab / 'q8-prefetch-probe.c'), '-L' + str(lab / 'build-avx-deep/bin'),
    '-Wl,-rpath,' + str(lab / 'build-avx-deep/bin'), '-lggml-cpu', '-lggml-base', '-lm', '-ldl',
    '-o', str(root / 'probe')]
with (out / 'compile.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
report = {'compile_command': command, 'runs': [], 'sha256': {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [root / 'probe', *root.glob('*.so')]}}
for threads in (1, 12):
    for k, m in ((1024, 65536), (4096, 16384), (8192, 8192), (16384, 24)):
        name = f'k{k}-m{m}-t{threads}'
        command = ['numactl', '--physcpubind=0-11', '--interleave=0,1', str(root / 'probe'),
            str(root), str(k), str(m), str(threads), '24']
        with (out / (name + '.json')).open('w') as stdout, (out / (name + '.stderr')).open('w') as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=300)
        data = json.loads((out / (name + '.json')).read_text())
        medians = {str(d): statistics.median(r['seconds'] for r in data['runs'] if r['distance'] == d)
                   for d in (0, 8, 16, 32)}
        row = {'name': name, 'command': command, 'median_seconds': medians,
            'gain_percent': {d: 100 * (medians['0'] / s - 1) for d, s in medians.items()},
            'guarded_exact_cases': data['guarded_exact_cases']}
        report['runs'].append(row)
        (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
