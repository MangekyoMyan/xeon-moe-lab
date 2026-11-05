#!/usr/bin/env python3
"""Build and test tile geometry only after earlier measurements are finished."""
import fcntl
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import run_model_tests as models

ROOT, LAB, OUT = models.ROOT, models.LAB, models.OUT
TILES = LAB / 'review-q8-tiles-20261025'
runner = models.runner

def main():
    runner.OUT = OUT
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        print('Waiting for prefill, pairing, and thread comparisons', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert len(json.loads((OUT / 'physical-smt-summary.json').read_text())) == 4
        try:
            if not TILES.exists():
                runner.run('tile-build-driver', [sys.executable, str(ROOT / 'prepare_tiles.py')], 1200, models.environment())
            else:
                assert json.loads((OUT / 'tile-build-driver/run.json').read_text())['returncode'] == 0
                runner.run('tile-rebuild-driver', ['cmake', '--build', str(TILES / 'build'), '--target', 'llama-bench',
                                                  'llama-completion', 'test-backend-ops', 'test-quantize-fns', '-j', '8'],
                           1200, models.environment())
            runner.run('tile-graph-validation-driver', [sys.executable, str(ROOT / 'validate_tiles.py')],
                       1200, models.environment(), keep_root=True)
            command = ['g++', '-O3', '-std=c++17', '-march=ivybridge', '-fopenmp', '-UNDEBUG',
                       '-I', str(TILES / 'source/ggml/include'), '-I', str(TILES / 'source/ggml/src'),
                       '-I', str(TILES / 'source/ggml/src/ggml-cpu'), str(ROOT / 'tile-probe.cpp'),
                       '-L', str(TILES / 'build/bin'), '-Wl,-rpath,' + str(TILES / 'build/bin'),
                       '-lggml-cpu', '-lggml-base', '-o', str(ROOT / 'tile-probe')]
            runner.run('compile-tile-probe', command, 300, models.environment())
            for k, m, n, threads in [(32, 1, 2, 24), (96, 35, 7, 24), (1408, 65, 33, 12), (4096, 17, 3, 7)]:
                label = f'tile-edge-k{k}-m{m}-n{n}-t{threads}'
                runner.run(label, ['numactl', '--interleave=0,1', str(ROOT / 'tile-probe'),
                                  str(k), str(m), str(n), str(threads), 'check'], 300, models.environment())
            summaries = []
            shapes = [(4096, 12288, 32), (4096, 12288, 128), (12288, 4096, 32), (12288, 4096, 128),
                      (4096, 1408, 128), (1408, 4096, 128)]
            for k, m, n in shapes:
                for threads in [12, 24]:
                    label = f'tile-micro-k{k}-m{m}-n{n}-t{threads}'
                    runner.run(label, ['numactl', '--interleave=0,1', str(ROOT / 'tile-probe'), str(k), str(m),
                                      str(n), str(threads)], 600, models.environment())
                    rows = [json.loads(line) for line in (OUT / label / 'stdout.txt').read_text().splitlines()]
                    baseline = {row['round']: row['matrix_seconds'] for row in rows if row['tile'] == 0}
                    result = {'label': label, 'K': k, 'M': m, 'N': n, 'threads': threads, 'tiles': {}}
                    for code in [0, 12, 14, 22, 23, 24, 32, 41, 42]:
                        selected = [row for row in rows if row['tile'] == code]
                        result['tiles'][str(code)] = {
                            'median_Gop_s': statistics.median(row['dense_multiply_add_Gop_s'] for row in selected),
                            'median_speed_ratio': statistics.median(baseline[row['round']] / row['matrix_seconds'] for row in selected)}
                    summaries.append(result)
                    (OUT / 'tile-micro-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
                    print(json.dumps(result), flush=True)
            weights = {(4096, 12288): 2.46, (12288, 4096): 2.46, (4096, 1408): 0.552, (1408, 4096): 0.276}
            scores = {}
            for code in [12, 14, 22, 23, 24, 32, 41]:
                selected = [r for r in summaries if r['threads'] == 24 and r['N'] == 128]
                scores[code] = math.exp(sum(weights[(r['K'], r['M'])] * math.log(r['tiles'][str(code)]['median_speed_ratio'])
                                           for r in selected) / sum(weights[(r['K'], r['M'])] for r in selected))
            best = sorted((code for code in scores if scores[code] > 1.005), key=lambda c: -scores[c])[:2]
            (OUT / 'tile-micro-selection.json').write_text(json.dumps({'weighted_scores': scores, 'selected': best}, indent=2) + '\n')
            print('Selected tile candidates', best, scores, flush=True)
            # Full graph checks also cover the changed prefill row assignment.
            import run_mid_sequence as validation
            validation.REUSE = TILES
            validation.ROOT = ROOT / 'tile-validation'
            validation.OUT = OUT / 'tile-validation'
            shutil.copyfile(ROOT / 'logits-check.cpp', validation.ROOT / 'logits-check.cpp')
            runner.OUT = validation.OUT
            validation.full_logits()
            runner.OUT = OUT
            models.CANDIDATE = TILES / 'build'
            variants = [('baseline', {}), ('columns4', {'GGML_CPU_Q8_DENSE_COLS': '4'})]
            variants += [(f'tile{code}', {'GGML_CPU_Q8_SGEMM_TILE': str(code)}) for code in best]
            comparisons = []
            for index, (name, env) in enumerate(variants + list(reversed(variants))):
                result = models.run_bench(f'tile-model-{index}-{name}', 0, 0, prompt=128, tokens=0, repetitions=3,
                                          extra_env=env)
                result['variant'] = name
                comparisons.append(result)
                (OUT / 'tile-model-prefill-summary.json').write_text(json.dumps(comparisons, indent=2) + '\n')
            print('Tile exploration and real-model comparison complete', flush=True)
        finally:
            runner.OUT = OUT
            runner.fix_owner()

if __name__ == '__main__':
    main()
