#!/usr/bin/env python3
"""Screen exact-order kernels, recording whether model warm-up overlaps."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
sys.path.insert(0, str(LAB / 'board-study-20261024'))
import hardware_run as runner
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['explore', 'confirm'], required=True)
    parser.add_argument('--variants')
    args = parser.parse_args()
    runner.OUT = OUT
    variants = args.variants or ','.join(json.loads((ROOT / 'kernel-variants.json').read_text()))
    lock = None
    if args.stage == 'confirm':
        lock = (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'explore':
        cases = [(k, 96, 1, 6, 0.20, 4) for k in [1408, 4096, 10944, 12288]]
    else:
        cases = [(k, m, t, 0, 0.4, 6) for k, m in [(1408, 131072), (4096, 65536), (10944, 24576), (12288, 24576)] for t in [1, 12, 24]]
    summaries = []
    try:
        for k, m, threads, cpu, seconds, rounds in cases:
            label = f'kernel-{args.stage}-k{k}-m{m}-t{threads}'
            policy = ['--membind=1'] if args.stage == 'explore' else ['--interleave=0,1']
            command = ['numactl', *policy, str(ROOT / 'kernel-probe'), str(ROOT / 'kernels.so'), variants,
                       'bench', str(k), str(m), str(threads), str(seconds), str(rounds), str(cpu)]
            runner.run(label, command, 600, environment())
            rows = [json.loads(line) for line in (OUT / label / 'stdout.txt').read_text().splitlines() if line.startswith('{')]
            base = {x['round']: x['seconds_per_matrix'] for x in rows if x['variant'] == 'baseline'}
            summary = {'label': label, 'overlaps_cold_warmup': args.stage == 'explore', 'variants': {}}
            for name in variants.split(','):
                selected = [x for x in rows if x['variant'] == name]
                summary['variants'][name] = {
                    'median_weight_GB_s': statistics.median(x['weight_GB_s'] for x in selected),
                    'median_speedup_pct': 100 * (statistics.median(base[x['round']] / x['seconds_per_matrix'] for x in selected) - 1),
                    'round_speedup_pct': [100 * (base[x['round']] / x['seconds_per_matrix'] - 1) for x in selected],
                }
            summaries.append(summary)
            (OUT / f'kernel-{args.stage}-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
    finally:
        runner.fix_owner()
        if lock:
            lock.close()

if __name__ == '__main__':
    main()
