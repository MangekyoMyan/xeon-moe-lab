#!/usr/bin/env python3
"""Validate four-input sharing and compare real prefill throughput."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import run_followup as study

ROOT, LAB, OUT = study.ROOT, study.LAB, study.OUT
BUILD = LAB / 'review-mxfp4-cols4-20261023'
ENV = {'GGML_CPU_MXFP4_PACKED': '1', 'GGML_CPU_MXFP4_ROWS': '0', 'GGML_CPU_MXFP4_PAIR': '0', 'GGML_CPU_MXFP4_COLS4': '1'}


def compile_probe(name, source, base):
    command = ['c++', '-O3', '-march=ivybridge', '-std=c++17', '-fopenmp', '-I' + str(base / 'source/include'), '-I' + str(base / 'source/ggml/include'), '-I' + str(base / 'source/ggml/src'), str(source), '-L' + str(base / 'build/bin'), '-Wl,-rpath,' + str(base / 'build/bin'), '-lllama', '-lggml-cpu', '-lggml-base', '-o', str(ROOT / name)]
    subprocess.run(command, check=True)
    (OUT / (name + '-build.json')).write_text(json.dumps(command, indent=2) + '\n')


def verify():
    compile_probe('columns-backend-check', ROOT / 'packed-backend-check.cpp', BUILD)
    compile_probe('logits-columns', ROOT / 'logits-probe.cpp', BUILD)
    study.run('columns-backend-check', [str(ROOT / 'columns-backend-check')], ENV)
    rows = [json.loads(line) for line in (OUT / 'columns-backend-check/stdout.txt').read_text().splitlines()]
    assert rows[-1]['passed'] and rows[-1]['cases'] == 1296
    dump = OUT / 'columns-model-logits.bin'
    study.run('columns-model-logits', [str(ROOT / 'logits-columns'), str(study.MODEL), str(dump)], ENV)
    assert dump.read_bytes() == (OUT / 'model-logits-baseline.bin').read_bytes()
    validation = {'short_prefix_bit_equal': True, 'sha256': hashlib.sha256(dump.read_bytes()).hexdigest(), 'backend': rows[-1]}
    source = (ROOT / 'logits-probe.cpp').read_text()
    source = source.replace('if (count < 160)', 'if (count < 544)')
    source = source.replace('const int32_t prefix = 127;', 'const int32_t prefix = 511;')
    source = source.replace('160 * sizeof(llama_token)', '(prefix + records - 1) * sizeof(llama_token)')
    long_source = ROOT / 'logits-long-probe.cpp'
    long_source.write_text(source)
    comparisons = []
    for variant, base in [('baseline', LAB / 'review-glm-q8-columns4-20261020'), ('columns', BUILD)]:
        name = 'logits-long-' + variant
        compile_probe(name, long_source, base)
        output = OUT / (name + '.bin')
        study.run(name, [str(ROOT / name), str(study.MODEL), str(output)], ENV if variant == 'columns' else {})
        comparisons.append(hashlib.sha256(output.read_bytes()).hexdigest())
    assert len(set(comparisons)) == 1
    validation.update(long_prefix_bit_equal=True, long_sha256=comparisons[0])
    (OUT / 'columns-model-validation.json').write_text(json.dumps(validation, indent=2) + '\n')


def screen():
    for repeat, variants in enumerate([['off', 'on'], ['on', 'off']]):
        for variant in variants:
            env = dict(ENV, GGML_CPU_MXFP4_COLS4='1' if variant == 'on' else '0')
            study.run(f'columns-model-{variant}-r{repeat}', [str(BUILD / 'build/bin/llama-bench'), '-m', str(study.MODEL), '-t', '24', '-p', '128,512', '-n', '128', '-r', '2', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '1', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json'], env)


def main():
    mode = sys.argv[1]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        before = study.audit()
        {'verify': verify, 'screen': screen}[mode]()
        assert before == study.audit()
        (OUT / ('columns-runtime-' + mode + '-complete.json')).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for folder in (ROOT, OUT):
                for path in [folder, *folder.rglob('*')]:
                    os.chown(path, owner.st_uid, owner.st_gid)
