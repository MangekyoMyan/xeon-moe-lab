#!/usr/bin/env python3
"""Run matched comparisons and record the exact commands and result paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('suite', choices=['screen', 'confirm', 'observe', 'validate', 'moe-screen', 'moe-confirm', 'moe-validate'])
args = ap.parse_args()
lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-numa-rows-20261020'
base = lab / 'build-avx-prefetch/bin/llama-bench'
candidate = lab / 'review-numa-rows-20261020/build/bin/llama-bench'
moe = lab / 'review-moe-groups-20261020/build/bin/llama-bench'
index = out / (args.suite + '-index.json')
if index.exists():
    raise SystemExit('Preserve existing experiment index')
records = []

def run(name, command, timeout=1800):
    print(json.dumps({'starting': name}), flush=True)
    log_path = out / (name + '.log')
    with log_path.open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    row = {'name': name, 'command': command, 'exit_code': process.returncode, 'log': str(log_path)}
    records.append(row)
    index.write_text(json.dumps(records, indent=2) + '\n')
    process.check_returncode()
    text = log_path.read_text()
    if name.startswith('compare-'):
        start = text.rfind('\n{\n  "saved"')
        if start < 0:
            raise RuntimeError('No comparison summary')
        row.update(json.loads(text[start:]))
        index.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'finished': name, **{k: row[k] for k in ('saved', 'candidate_gain_percent') if k in row}}), flush=True)

def compare(name, a, b, a_mode, b_mode, extra=None):
    command = [sys.executable, str(lab / 'run_cpu_ab_numa_rows.py'), '--lab', str(lab),
        '--baseline', str(a), '--candidate', str(b),
        '--baseline-env', 'GGML_CPU_NUMA_REPLICA=0', '--candidate-env', 'GGML_CPU_NUMA_REPLICA=0',
        '--baseline-env', 'GGML_CPU_NUMA_SHARD=' + a_mode, '--candidate-env', 'GGML_CPU_NUMA_SHARD=' + b_mode,
        '--note', name + '; fresh boot and light observer'] + (extra or [])
    run('compare-' + name, command)

if args.suite == 'screen':
    run('physical-placement', [sys.executable, str(lab / 'capture_numa_rows.py')])
    run('control-placement', [sys.executable, str(lab / 'capture_numa_rows.py'), '--mode', '2'])
    compare('shard-current-abba', base, candidate, '0', '1')
    compare('locality-control-abba', candidate, candidate, '2', '1')
elif args.suite == 'confirm':
    compare('shard-current-baab', base, candidate, '0', '1', ['--order', 'BAAB'])
    first = json.loads((out / 'screen-index.json').read_text())
    first_gain = next(r['candidate_gain_percent'] for r in first if r['name'] == 'compare-shard-current-abba')
    if first_gain > 0.5 and records[-1]['candidate_gain_percent'] > 0.5:
        compare('disabled-current-abba', base, candidate, '0', '0')
        compare('shard-prefill-abba', base, candidate, '0', '1',
                ['--prompt', '512', '--tokens', '0', '--threads', '24', '--repetitions', '1'])
    else:
        print(json.dumps({'additional_adoption_checks_skipped': 'No repeatable gain above 0.5% in both screening orders; retain current version.'}), flush=True)
elif args.suite == 'observe':
    for variant, binary, mode in [('A', base, '0'), ('B', candidate, '1')]:
        run('locality-observe-' + variant, ['env', 'GGML_CPU_NUMA_SHARD=' + mode,
            'GGML_CPU_NUMA_REPLICA=0', 'GGML_CPU_NUMA_REPLICA_TRACE=0', 'OMP_DYNAMIC=FALSE',
            'OMP_THREAD_LIMIT=24', 'OMP_MAX_ACTIVE_LEVELS=1', sys.executable,
            str(lab / 'observe_numa_rows.py'), 'decode', 'locality', '--bench', str(binary),
            '--output', str(out / ('locality-observe-' + variant))])
elif args.suite == 'validate':
    run('full-model-validation', [sys.executable, str(lab / 'validate_numa_rows_model.py'),
        '--source', str(lab / 'review-numa-rows-20261020/source'),
        '--build', str(lab / 'review-numa-rows-20261020/build'),
        '--output', str(out / 'model-validation')])
elif args.suite in ('moe-screen', 'moe-confirm'):
    common = [sys.executable, str(lab / 'run_cpu_ab_numa_rows.py'), '--lab', str(lab),
        '--baseline', str(base), '--candidate', str(moe),
        '--baseline-env', 'GGML_CPU_NUMA_REPLICA=0', '--candidate-env', 'GGML_CPU_NUMA_REPLICA=0',
        '--baseline-env', 'GGML_CPU_NUMA_SHARD=0', '--candidate-env', 'GGML_CPU_NUMA_SHARD=0',
        '--baseline-env', 'GGML_CPU_MOE_GROUPS=0', '--candidate-env', 'GGML_CPU_MOE_GROUPS=1']
    if args.suite == 'moe-screen':
        run('compare-moe-current-abba', common + ['--note', 'Grouped experts versus current build; light observer'])
    else:
        run('compare-moe-current-baab', common + ['--order', 'BAAB', '--note', 'Grouped experts reverse comparison'])
        first = json.loads((out / 'moe-screen-index.json').read_text())[0]['candidate_gain_percent']
        if first > 0.5 and records[-1]['candidate_gain_percent'] > 0.5:
            controlled = common.copy()
            controlled[controlled.index('--baseline') + 1] = str(moe)
            run('compare-moe-same-build-abba', controlled + ['--note', 'Same executable with expert grouping disabled/enabled'])
            run('compare-moe-prefill-abba', common + ['--prompt', '512', '--tokens', '0',
                '--threads', '24', '--repetitions', '1', '--note', 'Grouped experts input regression check'])
        else:
            print(json.dumps({'additional_adoption_checks_skipped': 'No repeatable gain above 0.5% in both screening orders; retain current version.'}), flush=True)
elif args.suite == 'moe-validate':
    run('moe-full-model-validation', [sys.executable, str(lab / 'validate_numa_rows_model.py'),
        '--source', str(lab / 'review-moe-groups-20261020/source'),
        '--build', str(lab / 'review-moe-groups-20261020/build'), '--kind', 'groups',
        '--output', str(out / 'moe-model-validation')])
