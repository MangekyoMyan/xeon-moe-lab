#!/usr/bin/env python3
"""Validate and measure the isolated CPU_REPACK implementation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import run_followup as study

ROOT, LAB, OUT = study.ROOT, study.LAB, study.OUT
PACKED = LAB / 'review-mxfp4-packed-20261023/build/bin'
ENV = {'GGML_CPU_MXFP4_PACKED': '1', 'GGML_CPU_MXFP4_ROWS': '0', 'GGML_CPU_MXFP4_PAIR': '0'}


def verify():
    study.run('packed-backend-check', [str(ROOT / 'packed-backend-check')], ENV)
    rows = [json.loads(line) for line in (OUT / 'packed-backend-check/stdout.txt').read_text().splitlines()]
    assert rows[-1]['passed'] and rows[-1]['cases'] > 0
    dump = OUT / 'packed-model-logits.bin'
    study.run('packed-model-logits', [str(ROOT / 'logits-packed'), str(study.MODEL), str(dump)], ENV)
    log = (OUT / 'packed-model-logits/stderr.txt').read_text()
    assert 'CPU_REPACK' in log
    reference = OUT / 'model-logits-baseline.bin'
    assert dump.read_bytes() == reference.read_bytes()
    (OUT / 'packed-model-validation.json').write_text(json.dumps({'bit_equal': True, 'bytes': dump.stat().st_size, 'sha256': hashlib.sha256(dump.read_bytes()).hexdigest(), 'packed_buffer_logged': True, 'backend': rows[-1]}, indent=2) + '\n')


def screen():
    for threads in (12, 24):
        study.run(f'packed-model-screen-t{threads}', [str(PACKED / 'llama-bench'), '-m', str(study.MODEL), '-t', str(threads), '-p', '128', '-n', '128', '-r', '3', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '1', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json'], ENV)


def main():
    mode = sys.argv[1]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        before = study.audit()
        {'verify': verify, 'screen': screen}[mode]()
        assert before == study.audit()
        (OUT / ('packed-runtime-' + mode + '-complete.json')).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
