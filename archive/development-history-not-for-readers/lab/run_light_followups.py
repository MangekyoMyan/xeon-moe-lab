#!/usr/bin/env python3
"""Confirm the Q8 candidate and reassess synchronization with light telemetry."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
destination = out / 'light-followups.json'
if destination.exists(): raise SystemExit('Preserve the existing followups')
baseline = lab / 'build-avx-deep/bin/llama-bench'
jobs = [
    ('q8-reverse', lab / 'review-q8-prefetch8-20261019/build/bin/llama-bench', ['--order', 'BAAB']),
    ('native-barrier', lab / 'review-native-barrier-20261019/build/bin/llama-bench', []),
    ('spin10000', baseline, ['--candidate-env', 'GOMP_SPINCOUNT=10000']),
]
rows = []
for name, candidate, extra in jobs:
    command = [sys.executable, str(lab / 'run_cpu_ab_light.py'), '--lab', str(lab),
        '--baseline', str(baseline), '--candidate', str(candidate), '--tokens', '32', '--repetitions', '3',
        '--note', 'Light observer followup: ' + name] + extra
    print(json.dumps({'starting': name}), flush=True)
    log_path = out / ('light-' + name + '.log')
    with log_path.open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    raw = log_path.read_text()
    start = raw.rfind('\n{\n  "saved"')
    if start < 0: raise RuntimeError('Missing comparison summary')
    summary = json.loads(raw[start:])
    rows.append({'name': name, 'command': command, **summary})
    destination.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'finished': name, **summary}), flush=True)
