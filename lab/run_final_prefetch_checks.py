#!/usr/bin/env python3
"""Build the permanent candidate and validate it without switching the launcher."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
destination = out / 'final-checks.json'
if destination.exists(): raise SystemExit('Preserve existing final checks')
records = []

def run(name, command, timeout):
    print(json.dumps({'starting': name}), flush=True)
    log_path = out / (name + '.log')
    with log_path.open('w') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    row = {'name': name, 'command': command, 'log': str(log_path), 'exit_code': process.returncode}
    records.append(row)
    destination.write_text(json.dumps(records, indent=2) + '\n')
    process.check_returncode()
    raw = log_path.read_text()
    if name.startswith('canonical-'):
        start = raw.rfind('\n{\n  "saved"')
        if start < 0: raise RuntimeError('Missing benchmark summary')
        row.update(json.loads(raw[start:]))
        destination.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'finished': name, **{k: v for k, v in row.items() if k in ('saved', 'candidate_gain_percent')}}), flush=True)

run('stage-permanent', [sys.executable, str(lab / 'stage_prefetch_candidate.py')], 120)
run('build-permanent', ['bash', str(lab / 'rebuild_deepseek_prefetch_staged.sh')], 1200)
run('validate-permanent', [sys.executable, str(lab / 'validate_prefetch_model.py'),
    '--source', str(lab / 'llama-private-prefetch'), '--build', str(lab / 'build-avx-prefetch'),
    '--output', str(out / 'permanent-model-validation')], 1200)
base = [sys.executable, str(lab / 'run_cpu_ab_light.py'), '--lab', str(lab),
    '--baseline', str(lab / 'build-avx-deep/bin/llama-bench'),
    '--candidate', str(lab / 'build-avx-prefetch/bin/llama-bench')]
for name, options in [
    ('canonical-decode-abba', ['--tokens', '32', '--repetitions', '3']),
    ('canonical-decode-baab', ['--tokens', '32', '--repetitions', '3', '--order', 'BAAB']),
    ('canonical-prefill-abba', ['--prompt', '512', '--tokens', '0', '--threads', '24', '--repetitions', '1']),
]:
    run(name, base + options + ['--note', 'Permanent Q8 prefetch candidate: ' + name + '; light observer'], 1800)
print(json.dumps({'saved': str(destination), 'launcher_activated': False}), flush=True)
