#!/usr/bin/env python3
"""Bounded offline DeepSeek benchmarks with phase, NUMA, CPU and disk telemetry."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

LAB = Path(__file__).resolve().parent
BENCH = '/opt/llama.cpp/build/bin/llama-bench'


def snapshot(pid):
    base = Path(f'/proc/{pid}')
    nodes, anon = {}, {}
    try:
        for line in (base / 'numa_maps').read_text().splitlines():
            fields = dict(p.split('=', 1) for p in line.split() if '=' in p)
            page = int(fields.get('kernelpagesize_kB', 4))
            for k, v in fields.items():
                if k.startswith('N') and k[1:].isdigit():
                    nodes[k] = nodes.get(k, 0) + int(v) * page
                    if 'anon' in fields:
                        anon[k] = anon.get(k, 0) + int(v) * page
        io = dict(line.split(':', 1) for line in (base / 'io').read_text().splitlines())
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (base / 'status').read_text().splitlines())
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return {'unix_time': time.time(), 'resident_kib_by_node': nodes,
                'anonymous_kib_by_node': anon, 'disk_read_bytes': int(io['read_bytes']),
                'cpu_seconds': (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'),
                'minor_faults': int(stat[7]), 'major_faults': int(stat[9]),
                'rss_kib': int(status.get('VmRSS', '0 kB').split()[0]),
                'hwm_kib': int(status.get('VmHWM', '0 kB').split()[0]),
                'available_kib': int(mem['MemAvailable'].split()[0])}
    except (OSError, ValueError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=BENCH)
    ap.add_argument('--label', default='')
    ap.add_argument('--load-mode', choices=['none', 'mmap', 'dio'])
    ap.add_argument('--batch', default='256')
    ap.add_argument('--ubatch', default='128')
    ap.add_argument('--cpu-mask', default='0xffffff')
    ap.add_argument('--blas-threads', default='24')
    ap.add_argument('--quant', choices=['IQ2_XXS-XL', 'Q2_K-XL'], default='IQ2_XXS-XL')
    ap.add_argument('--placement', choices=['default', 'mmap_interleave', 'anon_interleave', 'pinned_interleave'], default='default')
    ap.add_argument('--threads', default='12,24')
    ap.add_argument('--prompt', default='0')
    ap.add_argument('--tokens', default='32')
    ap.add_argument('--depth', default='0')
    ap.add_argument('--poll', default='50')
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=2400)
    args = ap.parse_args()
    directory = LAB / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF' / args.quant
    model = sorted(directory.glob('*00001-of-*.gguf'))[0]
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    suffix = f'-{args.label}' if args.label else ''
    folder = LAB / 'results' / f'deepseek-{args.quant}-{args.placement}{suffix}-{tag}'
    folder.mkdir()
    command = [args.bench, '-m', str(model), '-p', args.prompt, '-n', args.tokens,
               '-d', args.depth, '-t', args.threads, '--poll', args.poll,
               '-ngl', '0', '-r', str(args.repetitions), '-o', 'json',
               '--offline', '--progress', '-b', args.batch, '-ub', args.ubatch,
               '-lm', args.load_mode or ('none' if args.placement in ('anon_interleave', 'pinned_interleave') else 'mmap')]
    if args.placement != 'default':
        command = ['numactl', '--physcpubind=0-23', '--interleave=0,1'] + command
        if args.placement == 'pinned_interleave':
            # NUMA's inherited-mask mode would overwrite each worker's strict pin.
            command += ['--cpu-mask', args.cpu_mask, '--cpu-strict', '1']
        else:
            command += ['--numa', 'numactl']
    report = {'arguments': vars(args), 'command': command, 'started_unix': time.time(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'model_files_bytes': sum(p.stat().st_size for p in directory.glob('*.gguf')),
              'observations': [], 'phases': [], 'caveat': 'Synthetic random-token benchmark; system-wide counters require an otherwise idle machine.'}
    binary = Path(args.bench).resolve()
    artifacts = [binary] + [p for p in binary.parent.glob('*.so*') if not p.is_symlink()]
    report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (LAB / 'results/deepseek-current.txt').write_text(str(folder) + '\n')
    print(json.dumps({'starting': str(folder), 'command': command}), flush=True)
    start, progress, log_offset = time.monotonic(), 0, 0
    env = dict(os.environ, OPENBLAS_NUM_THREADS=args.blas_threads)
    report['environment'] = {k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_THREAD_TIMEOUT', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'GGML_CPU_TILED_MM', 'GGML_CPU_TILED_MM_FORCE', 'LD_LIBRARY_PATH') if k in env}
    with (folder / 'bench.json').open('w') as out, (folder / 'bench.log').open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env)
        report['pid'] = child.pid
        try:
            while child.poll() is None:
                sample = snapshot(child.pid)
                if sample:
                    sample['elapsed_seconds'] = time.monotonic() - start
                    report['observations'].append(sample)
                with (folder / 'bench.log').open() as log:
                    log.seek(log_offset)
                    for line in log:
                        if 'llama-bench: benchmark' in line or 'error' in line.lower():
                            phase = {'unix_time': time.time(), 'elapsed_seconds': time.monotonic() - start, 'message': line.strip()}
                            report['phases'].append(phase)
                            print(json.dumps(phase), flush=True)
                    log_offset = log.tell()
                elapsed = time.monotonic() - start
                if elapsed >= progress:
                    if sample:
                        masks = collections.Counter()
                        for status in Path(f'/proc/{child.pid}/task').glob('*/status'):
                            try:
                                mask = next(line.split(':', 1)[1].strip() for line in status.read_text().splitlines() if line.startswith('Cpus_allowed_list:'))
                                masks[mask] += 1
                            except (OSError, StopIteration):
                                pass
                        sample['thread_affinity_mask_counts'] = dict(masks)
                    print(json.dumps({'elapsed_seconds': round(elapsed), 'sample': sample}), flush=True)
                    (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
                    progress = elapsed + 30
                if elapsed > args.timeout:
                    report['timeout'] = True
                    break
                time.sleep(1)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            report['exit_code'] = child.wait()
    report['finished_unix'] = time.time()
    report['wall_seconds'] = time.monotonic() - start
    if report['exit_code'] == 0:
        report['benchmarks'] = json.loads((folder / 'bench.json').read_text())
    else:
        report['error_tail'] = (folder / 'bench.log').read_text()[-5000:]
    (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'saved': str(folder / 'summary.json'), 'exit_code': report['exit_code'],
                      'results': [{k: b[k] for k in ['n_threads', 'n_prompt', 'n_gen', 'avg_ts', 'stddev_ts']} for b in report.get('benchmarks', [])],
                      'error': report.get('error_tail')}), flush=True)
    raise SystemExit(0 if report['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
