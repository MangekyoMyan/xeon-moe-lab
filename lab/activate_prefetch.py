#!/usr/bin/env python3
"""Activate only the exact staged build that passed the supplied checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--decode', type=Path, required=True)
parser.add_argument('--reverse', type=Path, required=True)
parser.add_argument('--prefill', type=Path, required=True)
parser.add_argument('--validation', type=Path, required=True)
args = parser.parse_args()
lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
destination = out / 'activation.json'
if destination.exists(): raise SystemExit('Preserve the existing activation record')
staged = json.loads((out / 'staging.json').read_text())
build = Path(staged['build'])
cpu = build / 'bin/libggml-cpu.so.0.25.3'
digest = hashlib.sha256(cpu.read_bytes()).hexdigest()
reports = {}
current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for name, path in [('decode', args.decode), ('reverse', args.reverse), ('prefill', args.prefill)]:
    data = json.loads(path.read_text())
    if not data['same_boot'] or not data['warm_cache_observed']:
        raise SystemExit('Comparison conditions differ: ' + name)
    if name != 'prefill' and data['candidate_gain_percent'] <= 0:
        raise SystemExit('No positive generation result: ' + name)
    if name == 'prefill' and data['candidate_gain_percent'] < -1:
        raise SystemExit('Review prefill regression before activation')
    for run in data['runs']:
        if run['boot_id'] != current_boot:
            raise SystemExit('Comparison belongs to another boot')
        if run['variant'] == 'B' and run['artifact_sha256'].get(str(cpu)) != digest:
            raise SystemExit('Candidate binary changed since benchmark')
        for artifact_path, expected in run['artifact_sha256'].items():
            if hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest() != expected:
                raise SystemExit('An artifact changed since benchmark: ' + artifact_path)
    reports[name] = {'path': str(path.resolve()), 'gain_percent': data['candidate_gain_percent']}
validation = json.loads(args.validation.read_text())
for key in ('complete', 'logits_bit_identical', 'responses_bit_identical', 'token_counts_identical', 'same_boot', 'warm_cache'):
    if not validation.get(key): raise SystemExit('Missing validation: ' + key)
if not validation['kernel_check']['bit_identical'] or not validation['kernel_check']['quantize_test_passed']:
    raise SystemExit('Kernel validation incomplete')
if next(r for r in validation['logits'] if r['variant'] == 'B')['library_sha256'] != digest:
    raise SystemExit('Candidate changed since logits validation')
for relative, expected in staged['files'].items():
    if hashlib.sha256((Path(staged['source']) / relative).read_bytes()).hexdigest() != expected['after']:
        raise SystemExit('Staged source changed')
for name, entry in staged['scripts'].items():
    if hashlib.sha256((lab / name).read_bytes()).hexdigest() != entry['original_sha256']:
        raise SystemExit('Active script changed: ' + name)
    if hashlib.sha256(Path(entry['staged']).read_bytes()).hexdigest() != entry['staged_sha256']:
        raise SystemExit('Staged script changed: ' + name)
backups = {'run_deepseek_flash.sh': 'run_deepseek_flash_before_prefetch.sh',
           'rebuild_deepseek.sh': 'rebuild_deepseek_before_prefetch.sh'}
if any((lab / name).exists() for name in backups.values()): raise SystemExit('Preserve existing rollback scripts')
for name, backup in backups.items(): shutil.copy2(lab / name, lab / backup)
for name, entry in staged['scripts'].items():
    temporary = lab / (name + '.prefetch-new')
    shutil.copy2(entry['staged'], temporary)
    os.replace(temporary, lab / name)
preference = lab / 'experiment-preferences.json'
data = json.loads(preference.read_text())
(out / 'preferences-before-activation.json').write_text(preference.read_text())
data['previous_active_build'] = data['active_build']
data['active_build'] = str(build)
data['previous_private_llama_source'] = data['private_llama_source']
data['private_llama_source'] = staged['source']
data['current_work'] = 'Adopted bounded AVX Q8 weight prefetch after matched light-telemetry generation/prefill and exact output checks. Existing OpenMP synchronization and wait defaults retained. Read the memory-sync optimization report for measurements and protocol correction.'
data['memory_sync_results'] = str(out / 'REPORT_JA.txt')
data['benchmark_driver'] = str(lab / 'bench_deepseek_light.py')
data['benchmark_comparison_driver'] = str(lab / 'run_cpu_ab_light.py')
preference.write_text(json.dumps(data, indent=2) + '\n')
record = {'activated': True, 'source': staged['source'], 'build': str(build), 'cpu_library_sha256': digest,
    'benchmarks': reports, 'validation': str(args.validation.resolve()),
    'backups': {k: str(lab / v) for k, v in backups.items()},
    'active_scripts_sha256': {k: hashlib.sha256((lab / k).read_bytes()).hexdigest() for k in backups}}
destination.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
