#!/usr/bin/env python3
"""Exercise the actual user launcher and compare completed Japanese responses."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import run_followup as study

LAB, OUT = study.LAB, study.OUT
LAUNCHER = LAB / 'run_gpt_oss_mxfp4_optimized.sh'


def main():
    template = (OUT / 'model-chat-template.txt').read_text()
    assert template.count('strftime_now("%Y-%m-%d")') == 1
    fixed_template = OUT / 'pinned-chat-template.jinja'
    fixed_template.write_text(template.replace('strftime_now("%Y-%m-%d")', '"2026-10-23"'))
    prompts = [('arithmetic', '17と25を足した答えを、数字だけで答えてください。'), ('recall', 'この会話で使う合言葉は「青い傘731」です。合言葉だけをそのまま答えてください。'), ('numa', 'NUMAとは何かを、CPUが2個あるサーバーを例にして、日本語で説明してください。専門用語は説明を添え、200〜300字程度にまとめてください。')]
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study.runner.OUT = OUT
        study.runner.environment = study.environment
        before = study.audit()
        results = []
        original_snapshot = study.runner.snapshot
        for topic, prompt in prompts:
            label = 'launcher-verified-' + topic
            effective = {}
            def observe(pid):
                row = original_snapshot(pid)
                if not effective:
                    try:
                        if Path(f'/proc/{pid}/exe').resolve().name == 'llama-completion':
                            entries = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                            effective.update(dict(x.decode().split('=', 1) for x in entries if x.startswith((b'GGML_', b'OMP_')) and b'=' in x))
                    except OSError:
                        pass
                return row
            study.runner.snapshot = observe
            command = ['bash', str(LAUNCHER), '--single-turn', '--chat-template-file', str(fixed_template), '--reasoning-effort', 'low', '--temp', '0', '--seed', '42', '-p', prompt]
            study.runner.run(label, command)
            actual = (OUT / label / 'stdout.txt').read_bytes()
            reference = (OUT / ('final-japanese-v2-' + topic + '-0-A') / 'stdout.txt').read_bytes()
            assert actual == reference
            for name in ('GGML_CPU_MXFP4_PACKED', 'GGML_CPU_MXFP4_COLS4', 'GGML_CPU_MXFP4_PARALLEL_PACK', 'GGML_CPU_REPACK_REUSE_BUFFER'):
                assert effective.get(name) == '1'
            results.append({'topic': topic, 'bit_equal_text': True, 'sha256': hashlib.sha256(actual).hexdigest(), 'effective_environment': effective})
            (OUT / 'launcher-validation-progress.json').write_text(json.dumps(results, indent=2) + '\n')
        assert before == study.audit()
        (OUT / 'launcher-validation.json').write_text(json.dumps({'passed': True, 'launcher_sha256': hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(), 'responses': results, 'date_pinned_to': '2026-10-23', 'date_note': 'Native template injects the current local date. Testing crossed midnight; only the test template was pinned to the original baseline date.'}, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
