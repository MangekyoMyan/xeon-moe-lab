#!/usr/bin/env python3
"""Separate DRAM locality, worker ordering, and Q8 arithmetic throughput."""
import fcntl
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
sys.path.insert(0, str(LAB / 'board-study-20261024'))
import hardware_run as runner
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

def main():
    runner.OUT = OUT
    summaries = []
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for k, m in [(1408, 131072), (4096, 65536)]:
                for threads in [12, 24]:
                    cases = [('interleave', 'linear'), ('interleave', 'socket'),
                             ('rowlocal', 'socket'), ('opposite', 'socket'), ('all0', 'socket')]
                    if threads == 12:
                        cases.remove(('interleave', 'socket'))
                    for layout, order in cases:
                        label = f'numa-screen-k{k}-t{threads}-{layout}-{order}'
                        env = environment()
                        env.update(IVB_PROBE_LAYOUT=layout, IVB_PROBE_ORDER=order)
                        command = ['numactl', '--interleave=0,1', str(ROOT / 'kernel-probe'), str(ROOT / 'kernels.so'),
                                   'baseline,u4,u8', 'bench', str(k), str(m), str(threads), '0.5', '4', '0']
                        runner.run(label, command, 600, env)
                        rows = [json.loads(line) for line in (OUT / label / 'stdout.txt').read_text().splitlines() if line.startswith('{')]
                        result = {'label': label, 'K': k, 'M': m, 'threads': threads, 'layout': layout, 'order': order,
                                  'variants': {name: statistics.median(x['weight_GB_s'] for x in rows if x['variant'] == name)
                                               for name in ['baseline', 'u4', 'u8']}}
                        summaries.append(result)
                        (OUT / 'numa-screen-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
                        print(json.dumps(result), flush=True)
        finally:
            runner.fix_owner()

if __name__ == '__main__':
    main()
