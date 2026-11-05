#!/usr/bin/env python3
"""Run CPU experiments serially under the shared benchmark lock."""
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
sys.path.insert(0, str(LAB))
import run_quantize_tradeoff as runner
from run_mxfp4_study import audit

OUT = LAB / 'results/mxfp4-opt-20261023'


def main():
    OUT.mkdir(exist_ok=True)
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.OUT = OUT
        before = audit()
        (OUT / 'before.json').write_text(json.dumps(before, indent=2) + '\n')
        build = json.loads((ROOT / 'build.json').read_text())
        variants = ','.join(x['name'] for x in build['variants'])
        dual = ['numactl', '--physcpubind=0-23', '--interleave=0,1']
        runner.run('validation', dual + [str(ROOT / 'probe'), str(ROOT), variants, 'validate'])
        cases = [(5120, 24, 1), (2304, 24, 1), (5120, 24, 24), (2304, 24, 24),
                 (5120, 65536, 1), (5120, 65536, 12), (5120, 65536, 24),
                 (2304, 131072, 24), (5120, 2304, 24), (2304, 5120, 24)]
        for k, m, threads in cases:
            label = f'screen-k{k}-m{m}-t{threads}'
            runner.run(label, dual + [str(ROOT / 'probe'), str(ROOT), variants, str(k), str(m), str(threads), '.25', '4'])
        after = audit()
        assert before == after
        (OUT / 'after-screen.json').write_text(json.dumps(after, indent=2) + '\n')
        (OUT / 'screen-complete.json').write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        owner = LAB.stat()
        if os.geteuid() == 0 and OUT.exists():
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
