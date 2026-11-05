#!/usr/bin/env python3
"""Confirm the selected runtime against the baseline in balanced run order."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import run_followup as study

LAB, ROOT, OUT = study.LAB, study.ROOT, study.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['bench', 'responses', 'context'])
    ap.add_argument('--threads', type=int, choices=[12, 24], required=True)
    ap.add_argument('--rows', choices=['0', '2', '4'], required=True)
    ap.add_argument('--pair', choices=['0', '1'], required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--packed', action='store_true')
    ap.add_argument('--final-runtime', action='store_true')
    ap.add_argument('--prompts', default='128')
    ap.add_argument('--order', default='ABBABAAB')
    args = ap.parse_args()
    assert re.fullmatch('[a-zA-Z0-9-]+', args.tag)
    assert re.fullmatch(r'\d+(,\d+)*', args.prompts)
    assert all(1 <= int(x) <= 8192 for x in args.prompts.split(','))
    assert re.fullmatch('[AB]{2,8}', args.order) and args.order.count('A') == args.order.count('B')
    if args.final_runtime:
        args.packed = True
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        before = study.audit()
        env = {'GGML_CPU_MXFP4_ROWS': args.rows, 'GGML_CPU_MXFP4_PAIR': args.pair}
        candidate = LAB / 'review-mxfp4-packed-20261023/build/bin' if args.packed else study.CANDIDATE
        if args.packed:
            env['GGML_CPU_MXFP4_PACKED'] = '1'
        if args.final_runtime:
            candidate = LAB / 'review-mxfp4-final-20261023/build/bin'
            env.update(GGML_CPU_MXFP4_COLS4='1', GGML_CPU_MXFP4_PARALLEL_PACK='1', GGML_CPU_REPACK_REUSE_BUFFER='1')
        (OUT / (args.tag + '-config.json')).write_text(json.dumps(vars(args), indent=2) + '\n')
        if args.mode in ('bench', 'context'):
            for index, variant in enumerate(args.order if args.mode == 'bench' else 'ABBA'):
                binary = (study.BASE if variant == 'A' else candidate) / 'llama-bench'
                command = [str(binary), '-m', str(study.MODEL), '-t', str(args.threads), '-p', args.prompts if args.mode == 'bench' else '0', '-n', '256' if args.mode == 'bench' else '128', '-d', '0' if args.mode == 'bench' else '512', '-r', '2', '-b', '128', '-ub', '128', '-ngl', '0', '-lm', 'none', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '0', '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '-o', 'json']
                if variant == 'B' and args.packed:
                    command[command.index('--repack') + 1] = '1'
                study.run(f'{args.tag}-{index}-{variant}', command, env if variant == 'B' else {})
        else:
            prompts = [
                ('arithmetic', '17と25を足した答えを、数字だけで答えてください。', 1024),
                ('recall', 'この会話で使う合言葉は「青い傘731」です。合言葉だけをそのまま答えてください。', 1024),
                ('numa', 'NUMAとは何かを、CPUが2個あるサーバーを例にして、日本語で説明してください。専門用語は説明を添え、200〜300字程度にまとめてください。', 1024),
            ]
            comparisons = []
            for topic, prompt, count in prompts:
                for index, variant in enumerate('ABBA'):
                    label = f'{args.tag}-{topic}-{index}-{variant}'
                    binary = (study.BASE if variant == 'A' else candidate) / 'llama-completion'
                    command = [str(binary), '-m', str(study.MODEL), '-t', str(args.threads), '-tb', '24', '-c', '4096', '-b', '128', '-ub', '128', '-n', str(count), '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--cpu-mask-batch', '0xffffff', '--cpu-strict-batch', '1', '-lm', 'none', '-lzm', 'off', '--no-repack', '-ngl', '0', '--device', 'none', '--no-kv-offload', '--no-op-offload', '--jinja', '--reasoning', 'auto', '--reasoning-effort', 'low', '--temp', '0', '--seed', '42', '--offline', '--single-turn', '--simple-io', '--color', 'off', '--no-display-prompt', '--perf', '-p', prompt]
                    if variant == 'B' and args.packed:
                        command.remove('--no-repack')
                    study.run(label, command, env if variant == 'B' else {})
                    raw = (OUT / label / 'stdout.txt').read_bytes()
                    assert b'<|channel|>final<|message|>' in raw
                    log = (OUT / label / 'stderr.txt').read_text()
                    runs = re.findall(r'\beval time\s*=.*?/\s*(\d+)\s+runs', log)
                    assert runs and int(runs[-1]) < count - 1
                    comparisons.append({'label': label, 'topic': topic, 'variant': variant, 'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode()})
                    same_topic = [r for r in comparisons if r['topic'] == topic]
                    assert len({r['sha256'] for r in same_topic}) == 1
                    (OUT / (args.tag + '-comparison.json')).write_text(json.dumps(comparisons, indent=2, ensure_ascii=False) + '\n')
        assert before == study.audit()
        (OUT / (args.tag + '-complete.json')).write_text(json.dumps({'complete': True, 'normal_runtime_and_settings_unchanged': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
