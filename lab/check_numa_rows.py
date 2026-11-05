#!/usr/bin/env python3
"""Run exact-output and byte-transfer checks for the isolated NUMA experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

lab = Path(__file__).resolve().parent
root = lab / 'review-numa-rows-20261020'
out = lab / 'results/optimization-numa-rows-20261020'
source, build = root / 'source', root / 'build'
records = []
environment = dict(os.environ, OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='24',
                   OMP_MAX_ACTIVE_LEVELS='1', GGML_CPU_NUMA_REPLICA='0')

def run(name, command, env=environment, timeout=300):
    log_path = out / (name + '.log')
    with log_path.open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    records.append({'name': name, 'command': command, 'exit_code': result.returncode,
                    'log': str(log_path), 'shard_mode': env.get('GGML_CPU_NUMA_SHARD')})
    (out / 'checks.json').write_text(json.dumps(records, indent=2) + '\n')
    result.check_returncode()
    print(json.dumps({'passed': name}), flush=True)

probes = [
    ('shard', out / 'numa-shard-probe.cpp'),
    ('graph', lab / 'results/optimization-memory-sync-20261019/memory-sync-graph-probe.cpp'),
    ('moe', lab / 'moe-pair-probe.cpp'),
]
for name, code in probes:
    binary = out / (name + '-probe')
    command = ['c++', '-std=c++17', '-O2', '-march=native',
        '-I' + str(source / 'ggml/include'), '-I' + str(source / 'ggml/src'),
        '-I' + str(source / 'ggml/src/ggml-cpu'), str(code),
        '-L' + str(build / 'bin'), '-Wl,-rpath,' + str(build / 'bin'),
        '-lggml', '-lggml-cpu', '-lggml-base', '-pthread', '-o', str(binary)]
    run(name + '-compile', command)
    for mode in (('1', '2') if name == 'shard' else ('0',)):
        path = out / (name + '-' + mode + '.bin')
        run(name + '-' + mode, ['numactl', '--physcpubind=0-23', '--interleave=0,1',
            str(binary), str(path)], dict(environment, GGML_CPU_NUMA_SHARD=mode))
        records[-1]['output_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name == 'graph':
            reference = lab / 'results/optimization-memory-sync-20261019/baseline-checks/kernel.bin'
            assert path.read_bytes() == reference.read_bytes()
            records[-1]['baseline_bit_identical'] = True
        if name == 'shard' and mode == '2':
            assert path.read_bytes() == (out / 'shard-1.bin').read_bytes()
            records[-1]['placement_modes_bit_identical'] = True
run('quantize', [str(build / 'bin/test-quantize-fns')])
(out / 'checks.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'all_passed': True}))
