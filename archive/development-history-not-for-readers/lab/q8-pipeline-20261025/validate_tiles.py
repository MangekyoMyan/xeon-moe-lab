#!/usr/bin/env python3
"""Reuse established graph checks and add all TinyBLAS tile choices."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import validate_reuse as checks

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent

def main():
    checks.CANDIDATE = LAB / 'review-q8-tiles-20261025'
    checks.ROOT = ROOT / 'tile-validation'
    checks.OUT = LAB / 'results/q8-pipeline-20261025/tile-validation'
    checks.ROOT.mkdir(exist_ok=False)
    checks.OUT.mkdir(exist_ok=False)
    owner = LAB.stat()
    with (checks.OUT / 'unprivileged-model-pages.stdout').open('w') as stdout, (checks.OUT / 'unprivileged-model-pages.stderr').open('w') as stderr:
        check = subprocess.run([sys.executable, str(ROOT / 'check_unprivileged_model_pages.py')],
                               stdout=stdout, stderr=stderr, user=owner.st_uid, group=owner.st_gid)
    (checks.OUT / 'unprivileged-model-pages-result.json').write_text(json.dumps({'returncode': check.returncode}) + '\n')
    shutil.copyfile(ROOT / 'moe-check.cpp', checks.ROOT / 'moe-check.cpp')
    checks.main()
    records = []
    expected = hashlib.sha256((checks.OUT / 'reuse-baseline-dense.bin').read_bytes()).hexdigest()
    for code in [12, 14, 22, 23, 24, 32, 41, 42]:
        label = f'tile-{code}-dense'
        binary = checks.OUT / (label + '.bin')
        env = checks.environment()
        env['GGML_CPU_Q8_SGEMM_TILE'] = str(code)
        checks.call(label, ['numactl', '--interleave=0,1', str(checks.ROOT / 'reuse-candidate-dense'), str(binary)], env)
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert actual == expected, label
        records.append({'tile': code, 'bit_identical': True, 'sha256': actual, 'bytes': binary.stat().st_size})
        print(label, 'PASS', flush=True)
    env = checks.environment()
    env['GGML_CPU_Q8_MOE_SCHEDULE'] = '2'
    binary = checks.OUT / 'balanced-expert-groups.bin'
    checks.call('balanced-expert-groups', ['numactl', '--interleave=0,1', str(checks.ROOT / 'reuse-candidate-moe'), str(binary)], env)
    assert binary.read_bytes() == (checks.OUT / 'reuse-baseline-moe.bin').read_bytes()
    (checks.OUT / 'all-tiles-validation.json').write_text(json.dumps(records, indent=2) + '\n')
    print('All tiles and balanced expert groups passed', flush=True)

if __name__ == '__main__':
    main()
