#!/usr/bin/env python3
"""Compare isolated CPU variants with matched fresh-process ABBA workloads."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys


ENV_KEYS = {'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR'}


def environment_overrides(values, parser):
    overrides = {}
    for value in values:
        if '=' not in value:
            parser.error('Environment overrides must be KEY=VALUE.')
        key, setting = value.split('=', 1)
        if key not in ENV_KEYS or key in overrides or not setting:
            parser.error('Set each supported environment key at most once with a nonempty value.')
        if key == 'OMP_WAIT_POLICY' and setting.upper() not in ('ACTIVE', 'PASSIVE'):
            parser.error('OMP_WAIT_POLICY must be ACTIVE or PASSIVE.')
        if key == 'GOMP_SPINCOUNT' and not (setting.isdecimal() or setting.upper() == 'INFINITE'):
            parser.error('Use an unsigned decimal GOMP_SPINCOUNT or INFINITE.')
        if key in ('GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR') and setting not in ('0', '1'):
            parser.error(f'{key} must be 0 or 1.')
        overrides[key] = setting
    return overrides


def cpu_set(value):
    cpus = set()
    for part in value.strip().split(','):
        bounds = part.split('-')
        start = int(bounds[0])
        end = int(bounds[-1])
        cpus.update(range(start, end + 1))
    return cpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Baseline llama-bench')
    parser.add_argument('--candidate', type=Path, required=True, help='Candidate llama-bench')
    parser.add_argument('--prompt', type=int, default=0)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--cpu-mask', default='0xffffff')
    parser.add_argument('--poll', type=int, default=50)
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--note', default='')
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=2400)
    parser.add_argument('--order', choices=['ABBA', 'BAAB'], default='ABBA')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.prompt, args.tokens, args.depth) < 0 or not ((args.prompt > 0) ^ (args.tokens > 0)):
        parser.error('Choose exactly one timed phase: prompt > 0 or tokens > 0.')
    try:
        mask = int(args.cpu_mask, 16)
    except ValueError:
        parser.error('--cpu-mask must be hexadecimal.')
    if not 0 < mask <= 0xffffff or not 1 <= args.threads <= mask.bit_count():
        parser.error('Choose CPUs within 0-23 and at least one allowed CPU per worker.')
    if not 0 <= args.poll <= 100 or args.repetitions < 1 or args.timeout < 1:
        parser.error('Use poll 0-100 and positive repetitions/timeout.')
    args.cpu_mask = hex(mask)
    variant_environment = {
        'A': environment_overrides(args.baseline_env, parser),
        'B': environment_overrides(args.candidate_env, parser)}
    controlled_environment = {'OMP_DYNAMIC': 'FALSE', 'OMP_THREAD_LIMIT': '24', 'OMP_MAX_ACTIVE_LEVELS': '1'}
    effective_environment = {variant: {key: overrides.get(key, os.environ.get(key)) for key in sorted(ENV_KEYS)}
                             for variant, overrides in variant_environment.items()}
    lab = args.lab.resolve()
    binaries = {'A': args.baseline.resolve(), 'B': args.candidate.resolve()}
    if binaries['A'] == binaries['B'] and effective_environment['A'] == effective_environment['B']:
        parser.error('Choose different executables or explicitly different effective environment settings.')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    jobs = []
    for index, variant in enumerate(args.order):
        command = [sys.executable, str(lab / 'bench_deepseek.py'),
                   '--bench', str(binaries[variant]), '--label', f'deep-review-{stamp}-{index}-{variant}',
                   '--placement', 'pinned_interleave', '--load-mode', 'mmap',
                   '--threads', str(args.threads), '--batch', '512', '--ubatch', '512',
                   '--cpu-mask', args.cpu_mask, '--poll', str(args.poll),
                   '--prompt', str(args.prompt), '--tokens', str(args.tokens),
                   '--depth', str(args.depth), '--repetitions', str(args.repetitions),
                   '--timeout', str(args.timeout)]
        jobs.append({'variant': variant, 'command': command, 'environment_overrides': variant_environment[variant],
                     'effective_wait_environment': effective_environment[variant],
                     'controlled_environment': controlled_environment})
    if args.dry_run:
        print(json.dumps({'jobs': jobs}, indent=2))
        return 0
    if not (lab / 'bench_deepseek.py').is_file():
        parser.error('The lab directory must contain bench_deepseek.py.')
    for binary in binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f'Missing executable: {binary}')
    if any(os.environ.get(key) for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'GGML_BACKEND_PATH')):
        parser.error('Unset LD_LIBRARY_PATH, LD_PRELOAD, LD_AUDIT and GGML_BACKEND_PATH for this comparison; use each build with its own libraries.')
    if os.environ.get('GGML_CPU_PARALLEL_TRACE', '0') not in ('', '0'):
        parser.error('Disable GGML_CPU_PARALLEL_TRACE for throughput comparisons.')
    if any(os.environ.get(key, '0') not in ('', '0') for key in ('GGML_CPU_MOE_PAIR_TRACE', 'GGML_CPU_NUMA_REPLICA_TRACE')):
        parser.error('Disable experimental diagnostic traces for throughput comparisons.')
    if any(os.environ.get(key) for key in ('OMP_PROC_BIND', 'OMP_PLACES', 'GOMP_CPU_AFFINITY')):
        parser.error('Unset OpenMP affinity environment variables for a matched strict-pin comparison.')
    if not set(range(24)).issubset(os.sched_getaffinity(0)):
        parser.error('This driver uses the original server CPU IDs 0-23; they must all be allowed.')
    node_path = Path('/sys/devices/system/node/online')
    if not node_path.exists() or not {0, 1}.issubset(cpu_set(node_path.read_text())):
        parser.error('This driver requires the original NUMA nodes 0 and 1.')
    output = lab / 'results' / f'cpu-abba-{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'order': args.order, 'runs': [], 'adopted': False,
              'experiment_note': args.note, 'variant_environment': variant_environment,
              'effective_wait_environment': effective_environment,
              'controlled_environment': controlled_environment,
              'cpu_mask': args.cpu_mask, 'poll': args.poll,
              'bench_driver_sha256': hashlib.sha256((lab / 'bench_deepseek.py').read_bytes()).hexdigest(),
              'environment': {key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OMP_WAIT_POLICY',
                  'OMP_PROC_BIND', 'OMP_PLACES', 'GOMP_SPINCOUNT', 'GOMP_CPU_AFFINITY',
                  'GGML_CPU_TILED_MM', 'GGML_CPU_TILED_MM_FORCE') if key in os.environ},
              'note': 'Process means are the comparison units. No automatic configuration changes.'}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    reference_settings = None
    variant_artifacts = {}
    for index, job in enumerate(jobs):
        saved = None
        driver_log = output / f'{index}-{job["variant"]}.driver.log'
        with driver_log.open('w') as log:
            child = subprocess.Popen(job['command'], cwd=lab, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                     env=dict(os.environ, **job['environment_overrides'], **controlled_environment))
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(event, dict) and 'saved' in event:
                        saved = Path(event['saved']).resolve()
                exit_code = child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
            finally:
                child.stdout.close()
        if exit_code or saved is None or not saved.is_relative_to(lab / 'results'):
            raise RuntimeError(f'Benchmark failed; inspect {driver_log}')
        result = json.loads(saved.read_text())
        rows = result.get('benchmarks', [])
        if result.get('exit_code') != 0 or result.get('timeout') or len(rows) != 1:
            raise RuntimeError(f'Incomplete or mixed benchmark: {saved}')
        row = rows[0]
        expected = {'n_prompt': args.prompt, 'n_gen': args.tokens,
                    'n_depth': args.depth, 'n_threads': args.threads,
                    'n_batch': 512, 'n_ubatch': 512, 'n_gpu_layers': 0,
                    'load_mode': 'mmap', 'cpu_strict': True, 'poll': args.poll}
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Unexpected benchmark parameters: {saved}')
        try:
            observed_mask = int(row['cpu_mask'], 16)
        except (KeyError, TypeError, ValueError):
            observed_mask = None
        if observed_mask != mask:
            raise RuntimeError(f'Unexpected CPU affinity mask: {saved}')
        settings_keys = ('model_filename', 'model_size', 'model_n_params', 'backends',
                         'type_k', 'type_v', 'cpu_mask', 'repack', 'flash_attn')
        if any(key not in row for key in settings_keys):
            raise RuntimeError(f'Missing model/runtime identity: {saved}')
        settings = {key: row[key] for key in settings_keys}
        if reference_settings is None:
            reference_settings = settings
            report['model_and_runtime_settings'] = settings
        elif settings != reference_settings:
            raise RuntimeError(f'Model/runtime settings differ across runs: {saved}')
        for key in ('samples_ns', 'samples_ts'):
            values = row.get(key, [])
            if len(values) != args.repetitions or any(not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0 for value in values):
                raise RuntimeError(f'Invalid or incomplete {key}: {saved}')
        if not math.isfinite(row['avg_ts']) or row['avg_ts'] <= 0 or not math.isfinite(row['stddev_ts']) or row['stddev_ts'] < 0:
            raise RuntimeError(f'Invalid benchmark statistics: {saved}')
        bench_log = saved.parent / 'bench.log'
        errors = [line for line in bench_log.read_text().splitlines()
                  if 'error' in line.lower() or 'failed' in line.lower()]
        if errors:
            raise RuntimeError(f'Benchmark reported an error, even though it exited successfully: {saved}')
        artifacts = result.get('artifact_sha256')
        if not isinstance(artifacts, dict) or not artifacts or any(not isinstance(key, str) or
                not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                for key, value in artifacts.items()):
            raise RuntimeError(f'Missing or invalid artifact identity: {saved}')
        if job['variant'] in variant_artifacts and variant_artifacts[job['variant']] != artifacts:
            raise RuntimeError(f'Build artifacts changed during the comparison: {saved}')
        variant_artifacts[job['variant']] = artifacts
        observations = result.get('observations', [])
        observed_io = [item.get('disk_read_bytes') for item in observations]
        read_bytes = max(observed_io) if observed_io and all(isinstance(value, int) and value >= 0 for value in observed_io) else None
        run = {'variant': job['variant'], 'summary': str(saved),
               'summary_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
               'boot_id': result.get('boot_id'), 'avg_ts': row['avg_ts'],
               'stddev_ts': row['stddev_ts'], 'max_observed_read_bytes': read_bytes,
               'artifact_sha256': result.get('artifact_sha256'),
               'warm_cache_observed': read_bytes == 0}
        report['runs'].append(run)
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    means = {key: [run['avg_ts'] for run in report['runs'] if run['variant'] == key]
             for key in ('A', 'B')}
    medians = {key: statistics.median(values) for key, values in means.items()}
    report['process_mean_tok_s'] = means
    report['median_process_mean_tok_s'] = medians
    report['candidate_gain_percent'] = 100 * (medians['B'] / medians['A'] - 1)
    boot_ids = [run['boot_id'] for run in report['runs']]
    report['same_boot'] = bool(all(boot_ids)) and len(set(boot_ids)) == 1
    report['warm_cache_observed'] = all(run['warm_cache_observed'] for run in report['runs'])
    report['candidate_all_above_baseline'] = min(means['B']) > max(means['A'])
    report['interpretation'] = ('Repeat the opposite order and validate real answers before adoption. '
                                'Positive process I/O can be startup activity; inspect phase telemetry. '
                                'One-second I/O snapshots do not prove exact timed-phase residency.')
    path = output / 'comparison.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'saved': str(path), 'candidate_gain_percent': report['candidate_gain_percent'],
                      'same_boot': report['same_boot'], 'warm_cache_observed': report['warm_cache_observed'],
                      'adopted': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
