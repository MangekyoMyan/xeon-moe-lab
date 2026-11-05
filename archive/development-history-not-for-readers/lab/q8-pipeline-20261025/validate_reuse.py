#!/usr/bin/env python3
"""Exercise column tails, broadcasting, row views, and optional paired outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
BASE = LAB / 'review-glm-q8-columns4-20261020'
CANDIDATE = LAB / 'review-q8-reuse-20261025'
OLD = LAB / 'results/glm-air-q8-20261020/validation-review-glm-q8-columns4-20261020'
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

def call(label, command, env=None):
    with (OUT / (label + '.stdout')).open('w') as stdout, (OUT / (label + '.stderr')).open('w') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=True)

def main():
    dense = (OLD / 'dense.cpp').read_text()
    start = dense.index('        const shape shapes[] = {')
    end = dense.index('        ggml_quantize_init', start)
    cases = []
    for k in [96, 1408, 4096, 12288]:
        for n in [1, 2, 3, 4, 5, 7, 17, 31, 32, 33]:
            cases.append(f'            {{{k}, 35, {n}, 1, 1, 2, 2, true}},')
    cases += ['            {4096, 17, 7, 2, 2, 4, 2, false},',
              '            {1408, 65, 5, 2, 2, 4, 2, true},',
              '            {32, 1, 1, 1, 1, 1, 1, false},',
              '            {4096, 1408, 1, 1, 1, 1, 1, false},',
              '            {1408, 4096, 1, 1, 1, 1, 1, true},',
              '            {12288, 4096, 1, 1, 1, 1, 1, false},']
    dense = dense[:start] + '        const shape shapes[] = {\n' + '\n'.join(cases) + '\n        };\n' + dense[end:]
    dense = dense.replace('{1, 4, 12, 24}', '{1, 3, 4, 12, 24}')
    dense = dense.replace('16u * 1024u * 1024u', '128u * 1024u * 1024u')
    (ROOT / 'reuse-dense-check.cpp').write_text(dense)
    moe = (ROOT / 'moe-check.cpp').read_text()
    anchor = '    const shape shapes[] = {'
    extra = '''
        {64, 35, 3, 2, 3, false, false, false, false},
        {1408, 35, 3, 2, 5, false, false, false, true},
        {4096, 17, 4, 3, 7, true, false, false, false},
        {96, 35, 3, 2, 2, false, false, true, false},'''
    moe = moe.replace(anchor, anchor + extra)
    (ROOT / 'reuse-moe-check.cpp').write_text(moe)
    expected = {}
    for name, root in [('baseline', BASE), ('candidate', CANDIDATE)]:
        for kind in ['dense', 'moe']:
            binary = ROOT / ('reuse-' + name + '-' + kind)
            command = ['g++', '-O2', '-std=c++17', '-march=ivybridge', '-fopenmp', '-UNDEBUG',
                       '-I', str(root / 'source/ggml/include'), str(ROOT / ('reuse-' + kind + '-check.cpp')),
                       '-L', str(root / 'build/bin'), '-Wl,-rpath,' + str(root / 'build/bin'),
                       '-lggml', '-lggml-cpu', '-lggml-base', '-o', str(binary)]
            call('compile-reuse-' + name + '-' + kind, command)
            if name == 'baseline':
                dest = OUT / ('reuse-baseline-' + kind + '.bin')
                call('reuse-baseline-' + kind, ['numactl', '--interleave=0,1', str(binary), str(dest)], environment())
                expected[kind] = hashlib.sha256(dest.read_bytes()).hexdigest()
    configs = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 2, 0, 0, 0), (0, 4, 0, 0, 0),
               (1, 4, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 6, 0), (1, 4, 1, 6, 2),
               (0, 0, 0, 0, 1), (0, 0, 0, 6, 2)]
    results = []
    for index, (cols2, dense_cols, pair, schedule, dense_rows) in enumerate(configs):
        env = environment()
        env.update(GGML_CPU_Q8_COLS2=str(cols2), GGML_CPU_Q8_DENSE_COLS=str(dense_cols),
                   GGML_CPU_Q8_MOE_PAIR=str(pair), GGML_CPU_Q8_MOE_SCHEDULE=str(schedule),
                   GGML_CPU_Q8_DENSE_ROWS=str(dense_rows),
                   GGML_CPU_MOE_PAIR_TRACE='1', GGML_CPU_OP_TIMING='1' if index == 0 else '0')
        for kind in ['dense', 'moe']:
            label = f'reuse-{index}-{kind}'
            dest = OUT / (label + '.bin')
            call(label, ['numactl', '--interleave=0,1', str(ROOT / ('reuse-candidate-' + kind)), str(dest)], env)
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert actual == expected[kind], label
            stderr = (OUT / (label + '.stderr')).read_text()
            if pair and kind == 'moe':
                assert 'MOE_PAIR' in stderr, 'Pairing must actually execute'
            if index == 0:
                assert 'OP_TIMING\t' in stderr and 'OP_TIMING_DROPPED\t0' in stderr
            results.append({'label': label, 'config': [cols2, dense_cols, pair, schedule, dense_rows], 'bytes': dest.stat().st_size,
                            'sha256': actual, 'bit_identical': True})
            (OUT / 'reuse-graph-validation.json').write_text(json.dumps(results, indent=2) + '\n')
            print(label, 'PASS', flush=True)
    env = environment()
    env.update(GGML_CPU_Q8_COLS2='1', GGML_CPU_Q8_DENSE_COLS='4', GGML_CPU_Q8_MOE_PAIR='1', GGML_CPU_Q8_MOE_SCHEDULE='6', GGML_CPU_Q8_DENSE_ROWS='2')
    call('reuse-test-backend-ops', [str(CANDIDATE / 'build/bin/test-backend-ops'), '-b', 'CPU', '-o', 'MUL_MAT,MUL_MAT_ID', '-p', 'q8_0'], env)
    print('Reuse graph and backend checks passed', flush=True)

if __name__ == '__main__':
    main()
