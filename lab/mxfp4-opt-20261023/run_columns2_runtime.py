#!/usr/bin/env python3
"""Compare two-input, four-input, and combined sharing under one build."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import run_columns_runtime as previous
import run_followup as study

ROOT, LAB, OUT, BUILD = previous.ROOT, previous.LAB, previous.OUT, previous.BUILD


def environment(mode):
    return dict(previous.ENV, GGML_CPU_MXFP4_COLS4='1' if mode in ('four', 'both') else '0', GGML_CPU_MXFP4_COLS2='1' if mode in ('two', 'both') else '0')


def verify():
    checks = []
    for mode in ('two', 'both'):
        env = environment(mode)
        study.run('columns2-backend-' + mode, [str(ROOT / 'columns-backend-check')], env)
        data = [json.loads(line) for line in (OUT / ('columns2-backend-' + mode) / 'stdout.txt').read_text().splitlines()]
        assert data[-1]['passed'] and data[-1]['cases'] == 1296
        for prefix, binary, reference in [('short', ROOT / 'logits-columns', OUT / 'model-logits-baseline.bin'), ('long', ROOT / 'logits-long-columns', OUT / 'logits-long-baseline.bin')]:
            label = 'columns2-logits-' + mode + '-' + prefix
            target = OUT / (label + '.bin')
            study.run(label, [str(binary), str(study.MODEL), str(target)], env)
            assert target.read_bytes() == reference.read_bytes()
            checks.append({'mode': mode, 'prefix': prefix, 'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bit_equal': True})
        (OUT / 'columns2-validation.json').write_text(json.dumps(checks, indent=2) + '\n')


def screen():
    order = ['off', 'two', 'four', 'both']
    for repeat in range(2):
        for mode in (order if repeat == 0 else list(reversed(order))):
            study.run(f'columns2-screen-{mode}-r{repeat}', [str(BUILD / 'build/bin/llama-bench'), '-m', str(study.MODEL), '-t', '24', '-p', '128,512', '-n', '0', '-r', '2', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '1', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json'], environment(mode))


def main():
    mode = sys.argv[1]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        before = study.audit()
        {'verify': verify, 'screen': screen}[mode]()
        assert before == study.audit()
        (OUT / ('columns2-' + mode + '-complete.json')).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
