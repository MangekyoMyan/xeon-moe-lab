#!/usr/bin/env python3
"""Continue with baseline controls, graph reuse, and matched NUMA placement."""
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

import run_model_tests as models
import page_placement

ROOT, LAB, OUT = models.ROOT, models.LAB, models.OUT
REUSE = LAB / 'review-q8-reuse-20261025'
runner = models.runner

def command(label, args):
    print('Starting', label, flush=True)
    with (OUT / (label + '.log')).open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)
    print('Finished', label, flush=True)

def full_logits():
    binary = ROOT / 'reuse-logits-check'
    args = ['g++', '-O2', '-std=c++17', '-march=ivybridge', '-fopenmp', '-UNDEBUG',
            '-I', str(REUSE / 'source/include'), '-I', str(REUSE / 'source/ggml/include'),
            str(ROOT / 'logits-check.cpp'), '-L', str(REUSE / 'build/bin'),
            '-Wl,-rpath,' + str(REUSE / 'build/bin'), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-o', str(binary)]
    command('compile-reuse-logits', args)
    reference = LAB / 'results/glm-air-q8-20261020/validation-review-glm-q8-columns4-20261020/C-logits.bin'
    expected = hashlib.sha256(reference.read_bytes()).hexdigest()
    configs = [{}, {'GGML_CPU_Q8_COLS2': '1', 'GGML_CPU_Q8_DENSE_COLS': '4'},
               {'GGML_CPU_Q8_MOE_PAIR': '1', 'GGML_CPU_Q8_MOE_SCHEDULE': '6', 'GGML_CPU_Q8_DENSE_ROWS': '2'},
               {'GGML_CPU_Q8_COLS2': '1', 'GGML_CPU_Q8_DENSE_COLS': '4', 'GGML_CPU_Q8_MOE_PAIR': '1',
                'GGML_CPU_Q8_MOE_SCHEDULE': '6', 'GGML_CPU_Q8_DENSE_ROWS': '2'}]
    results = []
    for index, config in enumerate(configs):
        label = f'reuse-logits-{index}'
        destination = OUT / (label + '.bin')
        env = models.environment()
        env.update(config)
        args = [sys.executable, str(LAB / 'without_thp.py'), 'numactl', '--physcpubind=0-23', '--interleave=0,1',
                str(binary), str(models.MODEL), str(destination)]
        runner.run(label, args, 1200, env)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert actual == expected, label
        results.append({'label': label, 'config': config, 'bit_identical': True, 'sha256': actual})
        (OUT / 'reuse-logits-validation.json').write_text(json.dumps(results, indent=2) + '\n')
        print(label, 'bit-identical', flush=True)

def main():
    runner.OUT = OUT
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        print('Waiting for initial serial experiments', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads((OUT / 'kernel-screen-summary.json').read_text())
        assert len(previous) == 6
        migrated = False
        try:
            controls = []
            for index, allow in enumerate([False, True, True, False]):
                controls.append(models.run_bench(f'original-thp-control-{index}-{int(allow)}', 0, 0, prompt=0,
                                                tokens=128, repetitions=2, baseline=True, allow_thp=allow))
            (OUT / 'original-thp-controls.json').write_text(json.dumps(controls, indent=2) + '\n')
            command('reuse-build-resume', ['cmake', '--build', str(REUSE / 'build'), '--target', 'llama-bench',
                                         'llama-completion', 'test-backend-ops', 'test-quantize-fns', '-j', '8'])
            runner.run('reuse-graph-validation-driver', [sys.executable, str(ROOT / 'validate_reuse.py')], 1200,
                       models.environment(), keep_root=True)
            full_logits()
            models.CANDIDATE = REUSE / 'build'
            models.run_bench('op-timing-current', 0, 0, prompt=0, tokens=128, repetitions=1,
                             extra_env={'GGML_CPU_OP_TIMING': '1'})
            schedules = []
            for index, (schedule, dense_rows) in enumerate([(0, 0), (1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (6, 2), (0, 0)]):
                schedules.append(models.run_bench(f'reuse-schedule-{index}-s{schedule}-d{dense_rows}', 0, schedule,
                                                 prompt=0, tokens=96, repetitions=2,
                                                 extra_env={'GGML_CPU_Q8_DENSE_ROWS': str(dense_rows)}))
                (OUT / 'reuse-schedule-summary.json').write_text(json.dumps(schedules, indent=2) + '\n')
            print('Migrating cached Q8 weight pages to row-local nodes', flush=True)
            migrated = True
            page_placement.process('rowlocal', 'placement-rowlocal')
            command('rowlocal-stripes-before', [sys.executable, str(ROOT / 'inspect_page_stripes.py'), '--label', 'rowlocal-stripes-before'])
            rowlocal = []
            for index, local_schedule in enumerate([False, True, True, False]):
                schedule, dense_rows = (6, 2) if local_schedule else (0, 0)
                rowlocal.append(models.run_bench(f'rowlocal-model-{index}-s{schedule}-d{dense_rows}', 0, schedule,
                                                prompt=0, tokens=128, repetitions=2,
                                                extra_env={'GGML_CPU_Q8_DENSE_ROWS': str(dense_rows)}))
                (OUT / 'rowlocal-model-summary.json').write_text(json.dumps(rowlocal, indent=2) + '\n')
            command('rowlocal-stripes-after', [sys.executable, str(ROOT / 'inspect_page_stripes.py'), '--label', 'rowlocal-stripes-after'])
            page_placement.process('interleave', 'placement-restored-interleave')
            migrated = False
            restored = []
            for index, (schedule, dense_rows) in enumerate([(6, 2), (0, 0)]):
                restored.append(models.run_bench(f'interleave-restored-{index}-s{schedule}-d{dense_rows}', 0, schedule,
                                                prompt=0, tokens=128, repetitions=2,
                                                extra_env={'GGML_CPU_Q8_DENSE_ROWS': str(dense_rows)}))
            (OUT / 'interleave-restored-summary.json').write_text(json.dumps(restored, indent=2) + '\n')
            print('Middle sequence complete; weight pages restored to interleave', flush=True)
        finally:
            if migrated:
                page_placement.process('interleave', 'placement-emergency-restored-interleave')
            runner.fix_owner()

if __name__ == '__main__':
    main()
