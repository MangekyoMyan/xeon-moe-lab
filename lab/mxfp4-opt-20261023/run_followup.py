#!/usr/bin/env python3
"""Serial follow-up measurements; no timing overlaps with downloads or builds."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
sys.path.insert(0, str(LAB))
import run_quantize_tradeoff as runner
from glm_q8_experiment import environment
from run_mxfp4_study import audit

OUT = LAB / 'results/mxfp4-opt-20261023'
MODEL = LAB / 'models/ggml-org--gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf'
BASE = LAB / 'review-glm-q8-columns4-20261020/build/bin'
CANDIDATE = LAB / 'review-mxfp4-20261023/build/bin'
PREFIX = ['python3', str(LAB / 'without_thp.py'), 'numactl', '--physcpubind=0-23', '--interleave=0,1']


def run(label, command, extra=None, timeout=1200):
    runner.environment = lambda: dict(environment(), **(extra or {}))
    return runner.run(label, PREFIX + command, timeout)


def rows():
    variants = 'baseline,narrow,rows2,rows4,rows2-biased,rows4-biased'
    run('rows-validation', [str(ROOT / 'rows-probe'), str(ROOT), variants, 'validate'])
    for k, m, threads in [(5120, 96, 1), (2304, 96, 1), (5120, 96, 24), (2304, 96, 24),
                          (5120, 65536, 12), (5120, 65536, 24), (2304, 131072, 24), (5120, 2304, 24)]:
        run(f'rows-k{k}-m{m}-t{threads}', [str(ROOT / 'rows-probe'), str(ROOT), variants, str(k), str(m), str(threads), '.4', '6'])


def prefetch():
    variants = 'baseline,rows4-biased,rows4-pf8,rows4-pf16,rows4-pf32'
    run('prefetch-validation', [str(ROOT / 'rows-probe'), str(ROOT), variants, 'validate'])
    for k, m, threads in [(5120, 96, 24), (5120, 65536, 12), (5120, 65536, 24),
                          (2304, 131072, 24), (2880, 96, 24), (2880, 131072, 24)]:
        run(f'prefetch-k{k}-m{m}-t{threads}', [str(ROOT / 'rows-probe'), str(ROOT), variants, str(k), str(m), str(threads), '.4', '6'])


def packed():
    variants = 'baseline,narrow,rows4-biased,rows4-packed,rows4-packed-pf8,rows4-packed-pf16'
    run('packed-validation', [str(ROOT / 'packed-probe'), str(ROOT), variants, 'validate'])
    for k, m, threads in [(5120, 96, 1), (5120, 96, 24), (5120, 65536, 12), (5120, 65536, 24),
                          (2304, 131072, 24), (2880, 96, 24), (2880, 131072, 24), (2880, 2880, 24)]:
        run(f'packed-k{k}-m{m}-t{threads}', [str(ROOT / 'packed-probe'), str(ROOT), variants, str(k), str(m), str(threads), '.4', '6'])


def graphs():
    comparisons = []
    for threads in (12, 24):
        for selected, experts in ((1, 1), (1, 64), (6, 64)):
            for repeat in range(2):
                order = ['baseline', 'candidate', 'rows2', 'rows4', 'paired']
                if repeat % 2:
                    order.reverse()
                for variant in order:
                    label = f'graph-{variant}-t{threads}-s{selected}-e{experts}-r{repeat}'
                    dump = OUT / (label + '.bin')
                    binary = ROOT / ('graph-baseline' if variant == 'baseline' else 'graph-candidate')
                    run(label, [str(binary), str(threads), str(selected), str(experts), 'each', '3', 'both', str(dump)], {'GGML_CPU_MXFP4_PAIR': '1' if variant == 'paired' else '0', 'GGML_CPU_MXFP4_ROWS': '4' if variant in ('rows4', 'paired') else ('2' if variant == 'rows2' else '0')})
                    comparisons.append({'label': label, 'threads': threads, 'selected': selected, 'experts': experts, 'sha256': hashlib.sha256(dump.read_bytes()).hexdigest(), 'bytes': dump.stat().st_size})
                    matches = [x for x in comparisons if (x['threads'], x['selected'], x['experts']) == (threads, selected, experts)]
                    assert len({x['sha256'] for x in matches}) == 1
                    (OUT / 'graph-comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')


def smoke():
    assert json.loads((OUT / 'model-download.json').read_text())['complete']
    for variant, binary in [('baseline', BASE / 'llama-bench'), ('candidate', CANDIDATE / 'llama-bench')]:
        run('model-smoke-' + variant, [str(binary), '-m', str(MODEL), '-t', '12', '-p', '32', '-n', '16', '-r', '1', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '0', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json'])


def verify():
    run('verify-quantize', [str(CANDIDATE / 'test-quantize-fns')])
    for rows in (2, 4):
        run(f'verify-backend-rows{rows}', [str(CANDIDATE / 'test-backend-ops'), '-b', 'CPU', '-o', 'MUL_MAT,MUL_MAT_ID', '-p', 'type_a=mxfp4'], {'GGML_CPU_MXFP4_ROWS': str(rows)}, timeout=1800)
    comparisons = []
    for variant, rows, paired in [('baseline', 0, 0), ('candidate', 0, 0), ('rows2', 2, 0), ('rows4', 4, 0), ('paired', 4, 1)]:
        dump = OUT / ('model-logits-' + variant + '.bin')
        binary = ROOT / ('logits-baseline' if variant == 'baseline' else 'logits-candidate')
        run('model-logits-' + variant, [str(binary), str(MODEL), str(dump)], {'GGML_CPU_MXFP4_ROWS': str(rows), 'GGML_CPU_MXFP4_PAIR': str(paired)})
        comparisons.append({'variant': variant, 'bytes': dump.stat().st_size, 'sha256': hashlib.sha256(dump.read_bytes()).hexdigest()})
        (OUT / 'model-logits-comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        assert len({r['sha256'] for r in comparisons}) == 1


def modelscreen():
    for threads in (12, 24):
        for variant, rows, pair in [('baseline', 0, 0), ('candidate', 0, 0), ('rows2', 2, 0), ('rows4', 4, 0), ('paired', 4, 1)]:
            binary = (BASE if variant == 'baseline' else CANDIDATE) / 'llama-bench'
            run(f'model-screen-{variant}-t{threads}', [str(binary), '-m', str(MODEL), '-t', str(threads), '-p', '128', '-n', '128', '-r', '3', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '0', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json'], {'GGML_CPU_MXFP4_ROWS': str(rows), 'GGML_CPU_MXFP4_PAIR': str(pair)})


def main():
    mode = sys.argv[1]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.OUT = OUT
        before = audit()
        if mode == 'memory-followup':
            prefetch()
            packed()
        else:
            {'rows': rows, 'prefetch': prefetch, 'packed': packed, 'graphs': graphs, 'smoke': smoke, 'verify': verify, 'modelscreen': modelscreen}[mode]()
        assert audit() == before
        (OUT / (mode + '-complete.json')).write_text(json.dumps({'complete': True, 'runtime_and_settings_unchanged': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        owner = LAB.stat()
        if os.geteuid() == 0:
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
