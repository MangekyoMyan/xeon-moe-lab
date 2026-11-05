#!/usr/bin/env python3
"""Complete reverse comparison, causal control, counters and model checks."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-numa-rows-20261020'
runner = lab / 'run_numa_rows_experiments.py'
moe = lab / 'review-moe-groups-20261020/build/bin/llama-bench'
jobs = [
    ('rows-reverse', [sys.executable, str(runner), 'confirm']),
    ('moe-control', [sys.executable, str(lab / 'run_cpu_ab_numa_rows.py'),
        '--lab', str(lab), '--baseline', str(moe), '--candidate', str(moe),
        '--baseline-env', 'GGML_CPU_MOE_GROUPS=0', '--candidate-env', 'GGML_CPU_MOE_GROUPS=1',
        '--baseline-env', 'GGML_CPU_NUMA_REPLICA=0', '--candidate-env', 'GGML_CPU_NUMA_REPLICA=0',
        '--baseline-env', 'GGML_CPU_NUMA_SHARD=0', '--candidate-env', 'GGML_CPU_NUMA_SHARD=0',
        '--note', 'Resolve sub-percent expert-grouping difference with identical executable']),
    ('numa-counters', [sys.executable, str(runner), 'observe']),
    ('rows-model', [sys.executable, str(runner), 'validate']),
    ('moe-model', [sys.executable, str(runner), 'moe-validate']),
]
index = out / 'finish-index.json'
if index.exists():
    raise SystemExit('Preserve existing completion index')
records = []
for name, command in jobs:
    print(json.dumps({'starting_finish_stage': name}), flush=True)
    path = out / (name + '-finish.log')
    with path.open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    row = {'name': name, 'command': command, 'exit_code': result.returncode, 'log': str(path)}
    records.append(row)
    index.write_text(json.dumps(records, indent=2) + '\n')
    result.check_returncode()
    if name == 'moe-control':
        text = path.read_text()
        start = text.rfind('\n{\n  "saved"')
        if start < 0:
            raise RuntimeError('No control summary')
        row.update(json.loads(text[start:]))
        index.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'finished_finish_stage': name,
        **{k: row[k] for k in ('saved', 'candidate_gain_percent') if k in row}}), flush=True)
