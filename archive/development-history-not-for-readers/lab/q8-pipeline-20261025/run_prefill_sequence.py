#!/usr/bin/env python3
"""Measure counters, prompt reuse, pairing, and physical versus SMT workers."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import run_model_tests as models

ROOT, LAB, OUT = models.ROOT, models.LAB, models.OUT
runner = models.runner

def main():
    runner.OUT = OUT
    models.CANDIDATE = LAB / 'review-q8-reuse-20261025/build'
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        print('Waiting for middle sequence and placement restoration', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert len(json.loads((OUT / 'interleave-restored-summary.json').read_text())) == 2
        try:
            owner = LAB.stat()
            with (OUT / 'unprivileged-page-migration.stdout').open('w') as stdout, (OUT / 'unprivileged-page-migration.stderr').open('w') as stderr:
                test = subprocess.run([sys.executable, str(ROOT / 'page_placement.py'), 'selftest', '--unprivileged'],
                                      stdout=stdout, stderr=stderr, user=owner.st_uid, group=owner.st_gid)
            (OUT / 'unprivileged-page-migration-result.json').write_text(json.dumps({'returncode': test.returncode}) + '\n')
            for event in ['stalls', 'locality']:
                label = 'current-counter-' + event
                args = [sys.executable, str(LAB / 'without_thp.py'), sys.executable,
                        str(LAB / 'observe_glm_counters_light.py'), 'decode', event,
                        '--output', str(OUT / (label + '-capture')), '--bench', str(models.BASE / 'bin/llama-bench'),
                        '--model', str(models.MODEL), '--threads', '24', '--tokens', '128', '--window', '18']
                runner.run(label + '-driver', args, 600, models.environment(), keep_root=True)
            configs = {'A': {}, 'B': {'GGML_CPU_Q8_COLS2': '1'},
                       'C': {'GGML_CPU_Q8_DENSE_COLS': '2'}, 'D': {'GGML_CPU_Q8_DENSE_COLS': '4'},
                       'E': {'GGML_CPU_Q8_COLS2': '1', 'GGML_CPU_Q8_DENSE_COLS': '4'}}
            results = []
            for index, variant in enumerate('ABCDEEDCBA'):
                item = models.run_bench(f'prefill-reuse-{index}-{variant}', 0, 0, prompt=128, tokens=0, repetitions=3,
                                        extra_env=configs[variant])
                item['variant'] = variant
                results.append(item)
                (OUT / 'prefill-reuse-summary.json').write_text(json.dumps(results, indent=2) + '\n')
            pair_results = []
            for index, pair in enumerate([0, 1, 1, 0]):
                pair_results.append(models.run_bench(f'pair-decode-{index}-{pair}', 0, 0, prompt=0, tokens=128, repetitions=2,
                                                     extra_env={'GGML_CPU_Q8_MOE_PAIR': str(pair)}))
                (OUT / 'pair-decode-summary.json').write_text(json.dumps(pair_results, indent=2) + '\n')
            thread_results = []
            for index, threads in enumerate([24, 12, 12, 24]):
                thread_results.append(models.run_bench(f'physical-smt-{index}-t{threads}', 0, 0, threads=threads,
                                                       prompt=0, tokens=128, repetitions=2,
                                                       cpus='0-11' if threads == 12 else '0-23'))
                (OUT / 'physical-smt-summary.json').write_text(json.dumps(thread_results, indent=2) + '\n')
            print('Prefill, pairing, and physical-core comparisons complete', flush=True)
        finally:
            runner.fix_owner()

if __name__ == '__main__':
    main()
