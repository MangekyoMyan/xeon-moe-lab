#!/usr/bin/env python3
"""Build, check and time isolated candidates sequentially on the Xeon."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('variants', nargs='+', choices=['q8-prefetch8', 'native-barrier'])
args = parser.parse_args()
lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
records = []
env = dict(os.environ, OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='24', OMP_MAX_ACTIVE_LEVELS='1')
for key in ('GOMP_SPINCOUNT', 'OMP_WAIT_POLICY', 'GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR',
            'GGML_CPU_PARALLEL_TRACE', 'GGML_CPU_MOE_PAIR_TRACE', 'GGML_CPU_NUMA_REPLICA_TRACE'):
    env.pop(key, None)

def execute(command, directory, name, timeout=1200):
    with (directory / (name + '.stdout')).open('w') as stdout, (directory / (name + '.stderr')).open('w') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=True, timeout=timeout)
    return command

probe = out / 'memory-sync-graph-probe.cpp'
if not probe.exists():
    text = (lab / 'deep-kernel-graph-probe.cpp').read_text()
    anchor = '            {8192, 65, 1, 1, 1, 1, 1, false},'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '\n'.join([
        '            {1024, 67, 1, 1, 1, 1, 1, false},',
        '            {4096, 35, 1, 2, 2, 4, 2, true},',
        '            {16384, 24, 1, 1, 1, 1, 1, false},', anchor]))
    text = text.replace('const int threads : {1, 4}', 'const int threads : {1, 4, 12, 24}')
    text = text.replace('16u * 1024u * 1024u', '128u * 1024u * 1024u')
    probe.write_text(text)

for name in ['baseline'] + args.variants:
    root = lab / ('review-' + name + '-20261019')
    source = lab / 'llama-private' if name == 'baseline' else root / 'source'
    build = lab / 'build-avx-deep' if name == 'baseline' else root / 'build'
    checks = out / (name + '-checks')
    checks.mkdir(exist_ok=False)
    row = {'variant': name, 'commands': []}
    print(json.dumps({'starting_checks': name}), flush=True)
    if name != 'baseline':
        row['commands'].append(execute([sys.executable, str(lab / 'build_deep_candidate.py'), str(root)], checks, 'build'))
    flags = ['c++', '-std=c++17', '-O2', '-march=native', '-I' + str(source / 'ggml/include'),
        '-I' + str(source / 'ggml/src'), '-I' + str(source / 'ggml/src/ggml-cpu')]
    links = ['-L' + str(build / 'bin'), '-Wl,-rpath,' + str(build / 'bin'), '-lggml-cpu', '-lggml-base', '-pthread']
    tests = [('kernel', probe)]
    if name in ('baseline', 'native-barrier'):
        tests += [('barrier', lab / 'native-barrier-probe.cpp'), ('moe', lab / 'moe-pair-probe.cpp')]
    for label, file in tests:
        binary = checks / (label + '-probe')
        row['commands'].append(execute(flags + [str(file)] + links + ['-o', str(binary)], checks, label + '-compile'))
        command = ['numactl', '--physcpubind=0-23', '--interleave=0,1', str(binary)]
        if label != 'barrier': command.append(str(checks / (label + '.bin')))
        row['commands'].append(execute(command, checks, label, timeout=180))
        if label != 'barrier':
            digest = hashlib.sha256((checks / (label + '.bin')).read_bytes()).hexdigest()
            row[label + '_sha256'] = digest
            if name != 'baseline':
                reference = out / 'baseline-checks' / (label + '.bin')
                if digest != hashlib.sha256(reference.read_bytes()).hexdigest():
                    raise RuntimeError('Output differs from baseline: ' + label)
    if name != 'baseline':
        row['commands'].append(execute([str(build / 'bin/test-quantize-fns')], checks, 'quantize'))
    row['cpu_library_sha256'] = hashlib.sha256((build / 'bin/libggml-cpu.so').read_bytes()).hexdigest()
    records.append(row)
    (out / 'candidate-checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'checks_passed': name}), flush=True)
    if name != 'baseline':
        command = [sys.executable, str(lab / 'run_cpu_ab.py'), '--lab', str(lab),
            '--baseline', str(lab / 'build-avx-deep/bin/llama-bench'), '--candidate', str(build / 'bin/llama-bench'),
            '--tokens', '32', '--repetitions', '3', '--note', 'Memory-sync followup isolated ' + name]
        print(json.dumps({'starting_abba': name}), flush=True)
        row['commands'].append(execute(command, checks, 'decode-abba', timeout=1200))
        log = (checks / 'decode-abba.stdout').read_text()
        saved = log.rfind('\n{\n  "saved"')
        if saved < 0: raise RuntimeError('Missing ABBA summary')
        summary = json.loads(log[saved:])
        row['decode_abba'] = summary
        (out / 'candidate-checks.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({'finished_abba': name, **summary}), flush=True)
