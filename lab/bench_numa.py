#!/usr/bin/env python3
"""Run sequential, offline NUMA comparisons using the existing llama.cpp build."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

LAB = Path(__file__).resolve().parent
BENCH = '/opt/llama.cpp/build/bin/llama-bench'
CASES = {
    'default_mmap': ([], 12, 'mmap', []),
    'socket0_local': (['--physcpubind=0-5', '--membind=0'], 6, 'none', ['--numa', 'numactl']),
    'socket1_local': (['--physcpubind=6-11', '--membind=1'], 6, 'none', ['--numa', 'numactl']),
    'dual_memory_node0': (['--physcpubind=0-11', '--membind=0'], 12, 'none', ['--numa', 'numactl']),
    'dual_interleaved': (['--physcpubind=0-11', '--interleave=0,1'], 12, 'none', ['--numa', 'numactl']),
    'dual_smt_interleaved': (['--physcpubind=0-23', '--interleave=0,1'], 24, 'none', ['--numa', 'numactl']),
    'mmap_distribute': ([], 12, 'mmap', ['--numa', 'distribute']),
}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def memory_snapshot(pid):
    nodes, anonymous = {}, {}
    try:
        for line in Path(f'/proc/{pid}/numa_maps').read_text().splitlines():
            parts = line.split()
            fields = dict(part.split('=', 1) for part in parts if '=' in part)
            page_kib = int(fields.get('kernelpagesize_kB', 4))
            for key, value in fields.items():
                if key.startswith('N') and key[1:].isdigit():
                    nodes[key] = nodes.get(key, 0) + int(value) * page_kib
                    if 'anon' in fields:
                        anonymous[key] = anonymous.get(key, 0) + int(value) * page_kib
        io = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
        return {'time': stamp(), 'resident_kib_by_node': nodes,
                'anonymous_kib_by_node': anonymous,
                'disk_read_bytes': int(io.get('read_bytes', 0))}
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['llama8b', 'melody26b'], required=True)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()
    pattern = 'Meta-Llama-3.1' if args.model == 'llama8b' else 'Melody1437'
    inventory = json.loads((LAB / 'model-inventory.json').read_text())
    model = Path(next(row['local_path'] for row in inventory['files'] if pattern in row['repository']))
    folder = LAB / 'results' / ('numa-' + args.model + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'))
    folder.mkdir()
    report = {'started_at': stamp(), 'model': str(model), 'results_dir': str(folder),
              'method': 'Short synthetic random-token decode, sequential runs, no downloads; prewarm file cache before comparisons; load-mode none uses anonymous weight allocations to make memory policy effective',
              'environment_overrides': {'OPENBLAS_NUM_THREADS': '1'}, 'runs': []}
    start = time.monotonic()
    buffer = bytearray(32 * 1024 * 1024)
    read_bytes = 0
    next_progress = 2 * 1024**3
    print(json.dumps({'prewarming': str(model), 'gib': model.stat().st_size / 1024**3, 'results': str(folder)}), flush=True)
    with model.open('rb', buffering=0) as stream:
        while size := stream.readinto(buffer):
            read_bytes += size
            if read_bytes >= next_progress:
                print(json.dumps({'prewarmed_gib': read_bytes / 1024**3, 'seconds': time.monotonic() - start}), flush=True)
                next_progress += 2 * 1024**3
    report['prewarm'] = {'bytes': read_bytes, 'seconds': time.monotonic() - start}
    del buffer
    env = dict(os.environ, **report['environment_overrides'])
    for name in args.cases:
        binding, threads, load_mode, numa = CASES[name]
        command = (['numactl'] + binding if binding else []) + [
            BENCH, '-m', str(model), '-p', '0', '-n', str(args.tokens),
            '-t', str(threads), '-ngl', '0', '-r', str(args.repetitions),
            '-o', 'json', '--offline', '--progress', '-lm', load_mode] + numa
        row = {'case': name, 'command': command, 'started_at': stamp(), 'observations': []}
        print(json.dumps({'starting': name, 'threads': threads, 'load_mode': load_mode}), flush=True)
        with (folder / (name + '.json')).open('w') as stdout, (folder / (name + '.log')).open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, env=env)
            start = time.monotonic()
            while process.poll() is None:
                observation = memory_snapshot(process.pid)
                if observation:
                    row['observations'].append(observation)
                if time.monotonic() - start > args.timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    row['timeout_seconds'] = args.timeout
                    break
                time.sleep(2)
            row['exit_code'] = process.wait()
            row['wall_seconds'] = time.monotonic() - start
        row['finished_at'] = stamp()
        if row['exit_code'] == 0:
            row['benchmarks'] = json.loads((folder / (name + '.json')).read_text())
            speeds = [item['avg_ts'] for item in row['benchmarks']]
            print(json.dumps({'finished': name, 'tokens_per_second': speeds, 'seconds': row['wall_seconds']}), flush=True)
        else:
            row['error_tail'] = (folder / (name + '.log')).read_text()[-3000:]
            print(json.dumps({'failed': name, 'exit_code': row['exit_code'], 'error': row['error_tail']}), flush=True)
        report['runs'].append(row)
        (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    report['finished_at'] = stamp()
    (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'saved': str(folder / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
