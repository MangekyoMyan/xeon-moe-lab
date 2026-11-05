#!/usr/bin/env python3
"""Run isolated, identical token streams with balanced CPU masks."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
cases = [
    ('decode12-before-ssse3', 'build-native-noblas', 12, '0xffffff'),
    ('decode12-ssse3', 'build-avx-unpack', 12, '0xffffff'),
    ('decode10-ssse3', 'build-avx-unpack', 10, '0x7df'),
    ('decode8-ssse3', 'build-avx-unpack', 8, '0x3cf'),
    ('decode24-ssse3', 'build-avx-unpack', 24, '0xffffff'),
]
for label, build, threads, mask in cases:
    command = [sys.executable, str(lab / 'bench_deepseek.py'),
               '--bench', str(lab / build / 'bin/llama-bench'), '--label', label,
               '--placement', 'pinned_interleave', '--load-mode', 'mmap',
               '--threads', str(threads), '--cpu-mask', mask, '--prompt', '0',
               '--tokens', '32', '--batch', '512', '--ubatch', '512',
               '--repetitions', '3', '--timeout', '300']
    print(json.dumps({'starting': label, 'command': command}), flush=True)
    with (lab / 'results' / (label + '-driver.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    folder = Path((lab / 'results/deepseek-current.txt').read_text().strip())
    report = json.loads((folder / 'summary.json').read_text())
    assert report['arguments']['label'] == label
    print(json.dumps({'label': label, 'summary': str(folder / 'summary.json'),
                      'results': [{k: b[k] for k in ('n_threads', 'avg_ts', 'stddev_ts', 'samples_ts')}
                                  for b in report['benchmarks']]}), flush=True)
