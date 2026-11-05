#!/usr/bin/env python3
"""Run sequential diagnostic captures without changing the active setup."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
root = lab / 'results/bottleneck-observation-20261019'
cases = [('decode', 'locality', 2),
         ('prefill', 'locality', 1), ('decode', 'stalls', 3), ('prefill', 'stalls', 2)]
records = []
for phase, events, repeat in cases:
    key = f'{phase}-{events}-{repeat}'
    command = ['sudo', '-n', 'env', 'OMP_DYNAMIC=FALSE', 'OMP_THREAD_LIMIT=24',
               'OMP_MAX_ACTIVE_LEVELS=1', 'GGML_CPU_NUMA_REPLICA=0',
               'GGML_CPU_MOE_PAIR_TRACE=0', 'GGML_CPU_NUMA_REPLICA_TRACE=0',
               sys.executable, str(lab / 'observe_bottlenecks.py'), phase, events,
               '--output', str(root / key)]
    print('Starting ' + key, flush=True)
    with (root / (key + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    records.append({'key': key, 'exit_code': result.returncode, 'command': command})
    (root / 'run-index.json').write_text(json.dumps(records, indent=2) + '\n')
    if result.returncode:
        raise SystemExit('Capture failed: ' + key)
    print('Finished ' + key, flush=True)
