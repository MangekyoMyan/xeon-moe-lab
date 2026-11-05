#!/usr/bin/env python3
"""Wait for the cold load, then run non-overlapping exploratory measurements."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name

def main():
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        print('Waiting for cold warm-up lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = json.loads((OUT / 'cold-warmup/run.json').read_text())
        assert report.get('returncode') == 0 and 'error' not in report
    commands = [
        ['page_placement.py', 'inspect', '--label', 'placement-before'],
        ['run_kernel_screen.py', '--stage', 'confirm', '--variants', 'baseline,u4,u8,pf0,pf16,pf32,pf32_t1'],
        ['run_numa_screen.py'],
        ['run_model_tests.py', 'logits'],
        ['run_model_tests.py', 'kernel-screen'],
    ]
    for index, arguments in enumerate(commands):
        label = f'sequence-{index}-' + arguments[0].removesuffix('.py')
        print('Starting', label, flush=True)
        with (OUT / (label + '.log')).open('w') as output:
            subprocess.run([sys.executable, str(ROOT / arguments[0]), *arguments[1:]],
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        print('Finished', label, flush=True)

if __name__ == '__main__':
    main()
