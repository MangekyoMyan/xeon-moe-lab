#!/usr/bin/env python3
"""Verify kernels and row schedules against retained baseline graph outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
BASE = LAB / 'review-glm-q8-columns4-20261020'
CANDIDATE = LAB / 'review-q8-pipeline-20261025'
OLD = LAB / 'results/glm-air-q8-20261020/validation-review-glm-q8-columns4-20261020'
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

def call(label, command, env=None):
    with (OUT / (label + '.stdout')).open('w') as stdout, (OUT / (label + '.stderr')).open('w') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=True)

def main():
    # Wider thread counts exercise row tails and unequal expert groups.
    source = (OLD / 'moe.cpp').read_text().replace('{1, 4, 12, 24}', '{1, 3, 4, 6, 7, 8, 12, 16, 24}')
    (ROOT / 'moe-check.cpp').write_text(source)
    for kind in ['dense', 'logits']:
        (ROOT / (kind + '-check.cpp')).write_text((OLD / (kind + '.cpp')).read_text())
    for name, build in [('baseline', BASE), ('candidate', CANDIDATE)]:
        for kind in ['moe', 'dense', 'logits']:
            binary = ROOT / (name + '-' + kind + '-check')
            command = ['g++', '-O2', '-std=c++17', '-march=ivybridge', '-fopenmp', '-UNDEBUG',
                       '-I', str(build / 'source/include'), '-I', str(build / 'source/ggml/include'),
                       str(ROOT / (kind + '-check.cpp')), '-L', str(build / 'build/bin'),
                       '-Wl,-rpath,' + str(build / 'build/bin'), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base',
                       '-o', str(binary)]
            call('compile-' + name + '-' + kind, command)
    checks = []
    expected = {}
    for kernel, schedule in [(0, 0), (4, 0), (8, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (4, 6), (8, 6)]:
        for kind in ['moe', 'dense']:
            if kind == 'dense' and schedule:
                continue
            if kind not in expected:
                dest = OUT / ('baseline-' + kind + '.bin')
                call('baseline-' + kind, ['numactl', '--interleave=0,1', str(ROOT / ('baseline-' + kind + '-check')), str(dest)], environment())
                expected[kind] = hashlib.sha256(dest.read_bytes()).hexdigest()
            label = f'candidate-{kind}-k{kernel}-s{schedule}'
            dest = OUT / (label + '.bin')
            env = environment()
            env.update(GGML_CPU_Q8_PIPELINE=str(kernel), GGML_CPU_Q8_MOE_SCHEDULE=str(schedule))
            call(label, ['numactl', '--interleave=0,1', str(ROOT / ('candidate-' + kind + '-check')), str(dest)], env)
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert actual == expected[kind], label
            checks.append({'label': label, 'sha256': actual, 'bytes': dest.stat().st_size, 'bit_identical': True})
            (OUT / 'graph-validation.json').write_text(json.dumps(checks, indent=2) + '\n')
            print(label, 'PASS', flush=True)
    env = environment()
    env.update(GGML_CPU_Q8_PIPELINE='8', GGML_CPU_Q8_MOE_SCHEDULE='3')
    call('test-backend-ops', [str(CANDIDATE / 'build/bin/test-backend-ops'), '-b', 'CPU', '-o', 'MUL_MAT,MUL_MAT_ID', '-p', 'q8_0'], env)
    call('test-quantize-fns', [str(CANDIDATE / 'build/bin/test-quantize-fns')], env)
    print('All graph, backend, and quantization checks passed', flush=True)

if __name__ == '__main__':
    main()
