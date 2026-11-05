#!/usr/bin/env python3
"""Serial GLM validation and performance measurements with hardware monitoring."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
BASE = LAB / 'review-glm-q8-columns4-20261020/build'
CANDIDATE = LAB / 'review-q8-pipeline-20261025/build'
MODEL = LAB / 'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf'
sys.path.insert(0, str(LAB / 'board-study-20261024'))
import hardware_run as runner
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

def run_bench(label, kernel, schedule, threads=24, prompt=128, tokens=128, repetitions=2, depth=0, baseline=False, extra=None, allow_thp=False, extra_env=None, cpus='0-23'):
    build = BASE if baseline else CANDIDATE
    env = environment()
    env.update(GGML_CPU_Q8_PIPELINE=str(kernel), GGML_CPU_Q8_MOE_SCHEDULE=str(schedule))
    env.update(extra_env or {})
    selected_cpus = set()
    for component in cpus.split(','):
        values = component.split('-')
        start, end = int(values[0]), int(values[-1])
        assert 0 <= start <= end < 24
        selected_cpus.update(range(start, end + 1))
    assert threads <= len(selected_cpus)
    mask = hex(sum(1 << cpu for cpu in selected_cpus))
    command = ['numactl', '--physcpubind=' + cpus, '--interleave=0,1',
               str(build / 'bin/llama-bench'), '-m', str(MODEL), '-t', str(threads), '-p', str(prompt),
               '-n', str(tokens), '-d', str(depth), '-r', str(repetitions), '-b', '128', '-ub', '128',
               '-ngl', '0', '-lm', 'mmap', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '0',
               '--cpu-mask', mask, '--cpu-strict', '1', '--offline', '--progress', '-o', 'json']
    if not allow_thp:
        command = [sys.executable, str(LAB / 'without_thp.py'), *command]
    command += extra or []
    report = runner.run(label, command, timeout=3600, env=env)
    benchmarks = json.loads((OUT / label / 'stdout.txt').read_text())
    report.update(kernel=kernel, schedule=schedule, baseline=baseline, benchmarks=benchmarks, allow_thp=allow_thp,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), extra_environment=extra_env or {})
    samples = [x['process'] for x in report['samples'] if x.get('process')]
    report['max_disk_read_bytes'] = max((s['disk_read_bytes'] for s in samples), default=0)
    (OUT / label / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    result = {'label': label, 'kernel': kernel, 'schedule': schedule, 'threads': threads, 'baseline': baseline,
              'max_disk_read_bytes': report['max_disk_read_bytes'], 'benchmarks': benchmarks}
    print(json.dumps(result), flush=True)
    return result

def logits():
    reference = LAB / 'results/glm-air-q8-20261020/validation-review-glm-q8-columns4-20261020/C-logits.bin'
    expected = hashlib.sha256(reference.read_bytes()).hexdigest()
    results = []
    for kernel, schedule in [(0, 0), (4, 0), (8, 0), (0, 2), (0, 6), (4, 6), (8, 6)]:
        label = f'logits-k{kernel}-s{schedule}'
        binary = OUT / (label + '.bin')
        env = environment()
        env.update(GGML_CPU_Q8_PIPELINE=str(kernel), GGML_CPU_Q8_MOE_SCHEDULE=str(schedule))
        command = [sys.executable, str(LAB / 'without_thp.py'), 'numactl', '--physcpubind=0-23', '--interleave=0,1',
                   str(ROOT / 'candidate-logits-check'), str(MODEL), str(binary)]
        runner.run(label, command, timeout=1200, env=env)
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert actual == expected, label
        results.append({'label': label, 'bit_identical': True, 'sha256': actual, 'reference': str(reference), 'bytes': binary.stat().st_size})
        (OUT / 'model-logits-validation.json').write_text(json.dumps(results, indent=2) + '\n')
        print(label, 'bit-identical', flush=True)

def main():
    global CANDIDATE
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['logits', 'kernel-screen', 'schedule-screen', 'single'])
    parser.add_argument('--label')
    parser.add_argument('--kernel', type=int, default=0)
    parser.add_argument('--schedule', type=int, default=0)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--cpus', default='0-23')
    parser.add_argument('--prompt', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--allow-thp', action='store_true')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--env', action='append', default=[])
    args = parser.parse_args()
    if args.build:
        CANDIDATE = args.build.resolve()
    extra_env = dict(item.split('=', 1) for item in args.env)
    assert all(k.startswith(('GGML_CPU_', 'OMP_', 'GOMP_')) for k in extra_env)
    runner.OUT = OUT
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == 'logits':
                logits()
            elif args.stage == 'single':
                assert args.label
                run_bench(args.label, args.kernel, args.schedule, args.threads, args.prompt, args.tokens,
                          args.repetitions, args.depth, args.baseline, allow_thp=args.allow_thp, extra_env=extra_env, cpus=args.cpus)
            else:
                cases = [(0, 0), (4, 0), (8, 0), (8, 0), (4, 0), (0, 0)] if args.stage == 'kernel-screen' else [
                    (args.kernel, s) for s in [0, 1, 2, 3, 4, 5, 6, 0]]
                results = []
                for index, (kernel, schedule) in enumerate(cases):
                    label = f'{args.stage}-{index}-k{kernel}-s{schedule}-t{args.threads}'
                    results.append(run_bench(label, kernel, schedule, args.threads, args.prompt, args.tokens,
                                             args.repetitions, args.depth, cpus=args.cpus))
                    (OUT / (args.stage + '-summary.json')).write_text(json.dumps(results, indent=2) + '\n')
        finally:
            runner.fix_owner()

if __name__ == '__main__':
    main()
