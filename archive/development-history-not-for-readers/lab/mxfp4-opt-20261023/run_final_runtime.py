#!/usr/bin/env python3
"""Validate direct/parallel packing and measure startup without changing OS policy."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import run_columns_runtime as previous
import run_followup as study

ROOT, LAB, OUT = study.ROOT, study.LAB, study.OUT
BUILD = LAB / 'review-mxfp4-final-20261023'
ENV = {'GGML_CPU_MXFP4_PACKED': '1', 'GGML_CPU_MXFP4_ROWS': '0', 'GGML_CPU_MXFP4_PAIR': '0', 'GGML_CPU_MXFP4_COLS4': '1', 'GGML_CPU_MXFP4_PARALLEL_PACK': '1'}


def verify():
    for name, source in [('final-large-pack-check', 'large-pack-check.cpp'), ('final-backend-check', 'packed-backend-check.cpp'), ('final-logits', 'logits-probe.cpp'), ('final-logits-long', 'logits-long-probe.cpp')]:
        previous.compile_probe(name, ROOT / source, BUILD)
    study.run('final-large-pack-check', [str(ROOT / 'final-large-pack-check')], ENV)
    rows = [json.loads(line) for line in (OUT / 'final-large-pack-check/stdout.txt').read_text().splitlines()]
    assert rows[-1]['passed'] and rows[-1]['bytes_roundtrip_checked'] == 141004800
    study.run('final-backend-check', [str(ROOT / 'final-backend-check')], ENV)
    backend = [json.loads(line) for line in (OUT / 'final-backend-check/stdout.txt').read_text().splitlines()]
    assert backend[-1]['passed'] and backend[-1]['cases'] == 1296
    checks = []
    for kind, binary, reference in [('short', ROOT / 'final-logits', OUT / 'model-logits-baseline.bin'), ('long', ROOT / 'final-logits-long', OUT / 'logits-long-baseline.bin')]:
        target = OUT / ('final-logits-' + kind + '.bin')
        study.run('final-logits-' + kind, [str(binary), str(study.MODEL), str(target)], ENV)
        assert target.read_bytes() == reference.read_bytes()
        checks.append({'prefix': kind, 'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bit_equal': True})
    (OUT / 'final-runtime-validation.json').write_text(json.dumps({'large_pack': rows[-1], 'backend': backend[-1], 'logits': checks}, indent=2) + '\n')


def startup():
    modes = ['baseline', 'old-pack', 'direct', 'parallel']
    answers = []
    for repeat in range(2):
        for mode in (modes if repeat == 0 else list(reversed(modes))):
            if mode == 'baseline':
                binary, env = study.BASE / 'llama-completion', {}
            elif mode == 'old-pack':
                binary, env = LAB / 'review-mxfp4-packed-20261023/build/bin/llama-completion', dict(ENV, GGML_CPU_MXFP4_PARALLEL_PACK='0')
            else:
                binary, env = BUILD / 'build/bin/llama-completion', dict(ENV, GGML_CPU_MXFP4_PARALLEL_PACK='1' if mode == 'parallel' else '0')
            label = f'startup-{mode}-r{repeat}'
            command = [str(binary), '-m', str(study.MODEL), '-t', '24', '-tb', '24', '-c', '4096', '-b', '128', '-ub', '128', '-n', '1024', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--cpu-mask-batch', '0xffffff', '--cpu-strict-batch', '1', '-lm', 'none', '-lzm', 'off', '-ngl', '0', '--device', 'none', '--no-kv-offload', '--no-op-offload', '--jinja', '--reasoning', 'auto', '--reasoning-effort', 'low', '--temp', '0', '--seed', '42', '--offline', '--single-turn', '--simple-io', '--color', 'off', '--no-display-prompt', '--perf', '-p', '17と25を足した答えを、数字だけで答えてください。']
            if mode == 'baseline':
                command.append('--no-repack')
            study.run(label, command, env)
            answer = (OUT / label / 'stdout.txt').read_bytes()
            assert b'<|channel|>final<|message|>42 ' in answer
            answers.append({'label': label, 'sha256': hashlib.sha256(answer).hexdigest()})
            assert len({r['sha256'] for r in answers}) == 1
            (OUT / 'startup-answer-comparison.json').write_text(json.dumps(answers, indent=2) + '\n')


def verify_reuse():
    env = dict(ENV, GGML_CPU_REPACK_REUSE_BUFFER='1')
    checks = []
    for kind, binary, reference in [('short', ROOT / 'final-logits', OUT / 'model-logits-baseline.bin'), ('long', ROOT / 'final-logits-long', OUT / 'logits-long-baseline.bin')]:
        target = OUT / ('reuse-logits-' + kind + '.bin')
        study.run('reuse-logits-' + kind, [str(binary), str(study.MODEL), str(target)], env)
        assert target.read_bytes() == reference.read_bytes()
        checks.append({'prefix': kind, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bit_equal': True})
    (OUT / 'reuse-validation.json').write_text(json.dumps(checks, indent=2) + '\n')


def startup_reuse():
    answers = []
    modes = ['baseline', 'parallel', 'reuse']
    for repeat in range(2):
        for mode in (modes if repeat == 0 else list(reversed(modes))):
            binary = study.BASE / 'llama-completion' if mode == 'baseline' else BUILD / 'build/bin/llama-completion'
            env = {} if mode == 'baseline' else dict(ENV, GGML_CPU_REPACK_REUSE_BUFFER='1' if mode == 'reuse' else '0')
            label = f'startup-reuse-{mode}-r{repeat}'
            command = [str(binary), '-m', str(study.MODEL), '-t', '24', '-tb', '24', '-c', '4096', '-b', '128', '-ub', '128', '-n', '1024', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--cpu-mask-batch', '0xffffff', '--cpu-strict-batch', '1', '-lm', 'none', '-lzm', 'off', '-ngl', '0', '--device', 'none', '--no-kv-offload', '--no-op-offload', '--jinja', '--reasoning', 'auto', '--reasoning-effort', 'low', '--temp', '0', '--seed', '42', '--offline', '--single-turn', '--simple-io', '--color', 'off', '--no-display-prompt', '--perf', '-p', '17と25を足した答えを、数字だけで答えてください。']
            if mode == 'baseline': command.append('--no-repack')
            study.run(label, command, env)
            raw = (OUT / label / 'stdout.txt').read_bytes()
            assert b'<|channel|>final<|message|>42 ' in raw
            answers.append({'label': label, 'sha256': hashlib.sha256(raw).hexdigest()})
            assert len({r['sha256'] for r in answers}) == 1
            (OUT / 'startup-reuse-answers.json').write_text(json.dumps(answers, indent=2) + '\n')


def main():
    mode = sys.argv[1]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        before = study.audit()
        {'verify': verify, 'startup': startup, 'verify-reuse': verify_reuse, 'startup-reuse': startup_reuse}[mode]()
        assert before == study.audit()
        (OUT / ('final-runtime-' + mode + '-complete.json')).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for folder in (ROOT, OUT):
                for path in [folder, *folder.rglob('*')]:
                    os.chown(path, owner.st_uid, owner.st_gid)
