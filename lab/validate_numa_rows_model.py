#!/usr/bin/env python3
"""Compare full-model outputs and a fixed Japanese response for a selected build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--kind', choices=['rows', 'groups'], default='rows')
args = parser.parse_args()
lab = Path(__file__).resolve().parent
out = args.output.resolve()
out.mkdir(exist_ok=False)
model = lab / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf'
sources = {'A': lab / 'llama-private-prefetch', 'B': args.source.resolve()}
builds = {'A': lab / 'build-avx-prefetch', 'B': args.build.resolve()}
env = dict(os.environ, OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='24', OMP_MAX_ACTIVE_LEVELS='1',
           GGML_CPU_NUMA_REPLICA='0', GGML_CPU_MOE_PAIR_TRACE='0', GGML_CPU_NUMA_REPLICA_TRACE='0')
for key in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
    if os.environ.get(key): raise SystemExit('Unset ' + key)
report = {'logits': [], 'responses': [], 'complete': False, 'candidate_kind': args.kind,
          'variant_environment': {
              'A': {'GGML_CPU_NUMA_SHARD': '0', 'GGML_CPU_MOE_GROUPS': '0', 'GGML_CPU_NUMA_REPLICA': '0'},
              'B': {'GGML_CPU_NUMA_SHARD': '1' if args.kind == 'rows' else '0',
                    'GGML_CPU_MOE_GROUPS': '1' if args.kind == 'groups' else '0',
                    'GGML_CPU_NUMA_REPLICA': '0'}}}

def save():
    (out / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

kernel_source = lab / 'results/optimization-memory-sync-20261019/memory-sync-graph-probe.cpp'
kernel_binary = out / 'kernel-probe'
kernel_command = ['c++', '-std=c++17', '-O2', '-march=native', '-I' + str(sources['B'] / 'ggml/include'),
    str(kernel_source), '-L' + str(builds['B'] / 'bin'), '-Wl,-rpath,' + str(builds['B'] / 'bin'),
    '-lggml-cpu', '-lggml-base', '-pthread', '-o', str(kernel_binary)]
with (out / 'kernel-compile.log').open('w') as log:
    subprocess.run(kernel_command, stdout=log, stderr=subprocess.STDOUT, check=True)
with (out / 'kernel.log').open('w') as log:
    subprocess.run(['numactl', '--physcpubind=0-23', '--interleave=0,1', str(kernel_binary), str(out / 'kernel.bin')],
        stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=180)
reference = lab / 'results/optimization-memory-sync-20261019/baseline-checks/kernel.bin'
if (out / 'kernel.bin').read_bytes() != reference.read_bytes(): raise RuntimeError('Kernel outputs differ')
with (out / 'quantize.log').open('w') as log:
    subprocess.run([str(builds['B'] / 'bin/test-quantize-fns')], stdout=log, stderr=subprocess.STDOUT,
        env=env, check=True, timeout=180)
report['kernel_check'] = {'cases': 80, 'bit_identical': True, 'compile_command': kernel_command,
    'sha256': hashlib.sha256((out / 'kernel.bin').read_bytes()).hexdigest(), 'quantize_test_passed': True}
save()
print(json.dumps({'kernel_checks_passed': True}), flush=True)

for variant in 'AB':
    env['GGML_CPU_NUMA_SHARD'] = '1' if variant == 'B' and args.kind == 'rows' else '0'
    env['GGML_CPU_MOE_GROUPS'] = '1' if variant == 'B' and args.kind == 'groups' else '0'
    binary = out / (variant + '-logits-probe')
    command = ['c++', '-std=c++17', '-O2', '-I' + str(sources[variant] / 'include'),
        '-I' + str(sources[variant] / 'ggml/include'), str(lab / 'deepseek-logits-probe.cpp'),
        '-L' + str(builds[variant] / 'bin'), '-Wl,-rpath,' + str(builds[variant] / 'bin'),
        '-lllama', '-lggml-cpu', '-lggml-base', '-pthread', '-o', str(binary)]
    with (out / (variant + '-compile.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = out / (variant + '-logits.bin')
    run = ['numactl', '--physcpubind=0-23', '--interleave=0,1', str(binary), str(model), str(data)]
    print(json.dumps({'starting_logits': variant}), flush=True)
    with (out / (variant + '-logits.stdout')).open('w') as stdout, (out / (variant + '-logits.stderr')).open('w') as stderr:
        subprocess.run(run, stdout=stdout, stderr=stderr, env=env, check=True, timeout=600)
    raw = data.read_bytes()
    vocab, count, prefix = struct.unpack_from('<iii', raw)
    if (vocab, count, prefix) != (129280, 34, 127) or len(raw) != 12 + 160 * 4 + count * (8 + vocab * 4):
        raise RuntimeError('Unexpected logits artifact')
    report['logits'].append({'variant': variant, 'compile_command': command, 'run_command': run,
        'sha256': hashlib.sha256(raw).hexdigest(), 'logits_compared': vocab * count,
        'library_sha256': hashlib.sha256((builds[variant] / 'bin/libggml-cpu.so').read_bytes()).hexdigest()})
    save()
report['logits_bit_identical'] = len({r['sha256'] for r in report['logits']}) == 1
save()
if not report['logits_bit_identical']: raise RuntimeError('Full-model logits differ')
for index, variant in enumerate('ABBA'):
    env['GGML_CPU_NUMA_SHARD'] = '1' if variant == 'B' and args.kind == 'rows' else '0'
    env['GGML_CPU_MOE_GROUPS'] = '1' if variant == 'B' and args.kind == 'groups' else '0'
    label = f'{out.name}-short-{index}-{variant}'
    command = [sys.executable, str(lab / 'verify_deepseek_light.py'), '--binary', str(builds[variant] / 'bin/llama-completion'),
        '--label', label, '--threads', '12', '--threads-batch', '24', '--batch', '512', '--ubatch', '512', '--note-repetitions', '0']
    print(json.dumps({'starting_response': label}), flush=True)
    with (out / (label + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=600)
    paths = list((lab / 'results').glob('deepseek-functional-' + label + '-*/summary.json'))
    if len(paths) != 1: raise RuntimeError('Expected one fresh response summary')
    data = json.loads(paths[0].read_text())
    match = re.search(r'common_perf_print:\s+eval time\s*=.*?/\s*(\d+) runs', '\n'.join(data['timing_lines']))
    if not match: raise RuntimeError('Missing evaluated-token count')
    report['responses'].append({'variant': variant, 'summary': str(paths[0]), 'command': command,
        'response': data['response'], 'evaluated_tokens': int(match[1]), 'timings_ms': data['timings_ms'],
        'boot_id': data['boot_id'], 'max_disk_read_bytes': max(s.get('disk_read_bytes', 0) for s in data['observations'])})
    save()
report['responses_bit_identical'] = len({r['response'] for r in report['responses']}) == 1
report['token_counts_identical'] = len({r['evaluated_tokens'] for r in report['responses']}) == 1
report['same_boot'] = len({r['boot_id'] for r in report['responses']}) == 1
report['warm_cache'] = all(r['max_disk_read_bytes'] == 0 for r in report['responses'])
report['median_ms'] = {v: {phase: statistics.median(r['timings_ms'][phase] for r in report['responses'] if r['variant'] == v)
    for phase in ('load', 'prompt eval', 'eval')} for v in 'AB'}
report['generation_gain_percent'] = 100 * (report['median_ms']['A']['eval'] / report['median_ms']['B']['eval'] - 1)
report['complete'] = True
save()
if not all(report[k] for k in ('responses_bit_identical', 'token_counts_identical', 'same_boot', 'warm_cache')):
    raise RuntimeError('Response comparison conditions differ')
print(json.dumps({'saved': str(out / 'comparison.json'), 'logits_bit_identical': True,
    'responses_bit_identical': True, 'generation_gain_percent': report['generation_gain_percent']}), flush=True)
