#!/usr/bin/env python3
"""Compare expert grouping against the active build on nonuniform graph cases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-numa-rows-20261020'
code = out / 'moe-groups-probe.cpp'
variants = [
    ('reference', lab / 'llama-private-prefetch', lab / 'build-avx-prefetch', '0'),
    ('disabled', lab / 'review-moe-groups-20261020/source', lab / 'review-moe-groups-20261020/build', '0'),
    ('enabled', lab / 'review-moe-groups-20261020/source', lab / 'review-moe-groups-20261020/build', '1'),
]
records = []
for name, source, build, mode in variants:
    binary = out / ('moe-groups-' + name)
    command = ['c++', '-std=c++17', '-O2', '-march=native', '-I' + str(source / 'ggml/include'),
        str(code), '-L' + str(build / 'bin'), '-Wl,-rpath,' + str(build / 'bin'),
        '-lggml', '-lggml-cpu', '-lggml-base', '-pthread', '-o', str(binary)]
    with (out / (name + '-moe-compile.log')).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = out / ('moe-groups-' + name + '.bin')
    env = dict(os.environ, GGML_CPU_MOE_GROUPS=mode, GGML_CPU_NUMA_REPLICA='0',
               GGML_CPU_NUMA_SHARD='0', OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='24', OMP_MAX_ACTIVE_LEVELS='1')
    run = ['numactl', '--physcpubind=0-23', '--interleave=0,1', str(binary), str(data)]
    with (out / (name + '-moe.log')).open('x') as log:
        subprocess.run(run, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=300)
    records.append({'variant': name, 'command': command, 'run_command': run,
                    'mode': mode, 'sha256': hashlib.sha256(data.read_bytes()).hexdigest()})
    (out / 'moe-checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'passed': name}), flush=True)
assert len({r['sha256'] for r in records}) == 1
print(json.dumps({'all_bit_identical': True, 'cases_per_variant': 576}))
