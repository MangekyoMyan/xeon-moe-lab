#!/usr/bin/env python3
"""Inspect row placement and worker binding in a separate diagnostic run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['1', '2'], default='1')
args = ap.parse_args()
lab = Path(__file__).resolve().parent
build = lab / 'review-numa-rows-20261020/build'
out = lab / 'results/optimization-numa-rows-20261020' / ('placement-' + args.mode)
out.mkdir(exist_ok=False)
model = lab / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf'
command = ['numactl', '--physcpubind=0-23', '--interleave=0,1', str(build / 'bin/llama-bench'),
    '-m', str(model), '-p', '0', '-n', '32', '-d', '0', '-t', '12', '--poll', '50',
    '-ngl', '0', '-r', '1', '-o', 'json', '--offline', '--progress', '-b', '512',
    '-ub', '512', '-lm', 'mmap', '--cpu-mask', '0xffffff', '--cpu-strict', '1']
env = dict(os.environ, GGML_CPU_NUMA_SHARD=args.mode, GGML_CPU_NUMA_REPLICA='0',
    GGML_CPU_NUMA_REPLICA_TRACE='1', OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='24',
    OMP_MAX_ACTIVE_LEVELS='1')
report = {'command': command, 'diagnostic_only': True, 'mode': args.mode,
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'library_sha256': hashlib.sha256((build / 'bin/libggml-cpu.so').read_bytes()).hexdigest()}
with (out / 'bench.json').open('w') as stdout, (out / 'bench.log').open('w') as stderr:
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
    start = time.monotonic()
    captured = False
    try:
        while child.poll() is None:
            if time.monotonic() - start > 300:
                raise TimeoutError('Placement diagnostic timed out')
            log = (out / 'bench.log').read_text()
            if not captured and 'generation run 1/1' in log:
                for name in ('maps', 'numa_maps', 'status', 'io'):
                    (out / ('proc-' + name + '.txt')).write_text(Path(f'/proc/{child.pid}/{name}').read_text())
                report['thread_affinity'] = {
                    p.parent.name: next(line.split(':', 1)[1].strip() for line in p.read_text().splitlines()
                                      if line.startswith('Cpus_allowed_list:'))
                    for p in Path(f'/proc/{child.pid}/task').glob('*/status')}
                captured = True
            time.sleep(0.5)
        report['exit_code'] = child.returncode
        if child.returncode or not captured:
            raise RuntimeError('Placement run failed')
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

log = (out / 'bench.log').read_text()
vmas = [tuple(int(v, 16) for v in line.split()[0].split('-'))
        for line in (out / 'proc-maps.txt').read_text().splitlines()]
numa = {int(line.split()[0], 16): line for line in (out / 'proc-numa_maps.txt').read_text().splitlines()}
regions = []
for node, address, size, name in re.findall(r'NUMA_SHARD_RANGE node=(\d+) address=(0x[0-9a-f]+) bytes=(\d+) name=(\S+)', log):
    node, address, size = int(node), int(address, 16), int(size)
    matches = [(lo, hi) for lo, hi in vmas if lo <= address and hi >= address + size]
    if len(matches) != 1:
        raise RuntimeError('Expected each row half within one VMA')
    line = numa[matches[0][0]]
    pages = {int(n): int(count) for n, count in re.findall(r'\bN(\d+)=(\d+)', line)}
    regions.append({'node': node, 'address': address, 'bytes': size, 'name': name,
        'numa_maps': line, 'pages_by_node': pages,
        'all_resident_pages_local': pages.get(node, 0) > 0 and all(n == node or count == 0 for n, count in pages.items())})
workers = [dict(ith=int(i), nth=int(t), node=int(n)) for i, t, n in
           re.findall(r'NUMA_SHARD_WORK ith=(\d+) nth=(\d+) node=(\d+)', log)]
tensors = [dict(name=n, bytes=int(s), offset=int(o)) for n, s, o in
           re.findall(r'NUMA_REPLICA_TENSOR (\S+) bytes=(\d+) offset=(\d+)', log)]
report.update(regions=regions, workers=workers, tensors=tensors,
    total_selected_bytes=sum(x['bytes'] for x in tensors),
    expected_worker_mapping=bool(workers) and all(w['nth'] == 12 and w['node'] == w['ith'] // 6 for w in workers),
    all_regions_local=bool(regions) and all(r['all_resident_pages_local'] for r in regions))
if args.mode == '2':
    copies = []
    for address, size in re.findall(r'NUMA_REPLICA_MAP node=0 address=(0x[0-9a-f]+) bytes=(\d+)', log):
        address, size = int(address, 16), int(size)
        matches = [(lo, hi) for lo, hi in vmas if lo <= address and hi >= address + size]
        if len(matches) != 1:
            raise RuntimeError('Expected one interleaved copy VMA')
        line = numa[matches[0][0]]
        pages = {int(n): int(count) for n, count in re.findall(r'\bN(\d+)=(\d+)', line)}
        copies.append({'address': address, 'bytes': size, 'numa_maps': line, 'pages_by_node': pages})
    report['interleaved_copies'] = copies
    assert copies and all(0.4 < c['pages_by_node'].get(0, 0) / sum(c['pages_by_node'].values()) < 0.6 for c in copies)
(out / 'placement.json').write_text(json.dumps(report, indent=2) + '\n')
assert len(tensors) == 107 and report['total_selected_bytes'] == 3253207040
assert report['expected_worker_mapping']
if args.mode == '1':
    assert len(regions) == 214 and report['all_regions_local']
print(json.dumps({'saved': str(out / 'placement.json'), 'all_regions_local': report['all_regions_local'],
                  'expected_worker_mapping': report['expected_worker_mapping']}))
