#!/usr/bin/env python3
"""Check prompt recall, a normal response, and worker affinity."""
import collections
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path
import subprocess
import time
from bench_deepseek_light import LAB, snapshot

ap = argparse.ArgumentParser()
ap.add_argument('--binary')
ap.add_argument('--label', default='')
ap.add_argument('--threads', default='24')
ap.add_argument('--threads-batch', default='12')
ap.add_argument('--batch', default='256')
ap.add_argument('--ubatch', default='256')
ap.add_argument('--cpu-mask', default='0xffffff')
ap.add_argument('--cpu-mask-batch', default='0xffffff')
ap.add_argument('--note-repetitions', type=int, default=8)
ap.add_argument('--load-mode', default='mmap')
args = ap.parse_args()

tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
suffix = f'-{args.label}' if args.label else ''
folder = LAB / 'results' / f'deepseek-functional{suffix}-{tag}'
folder.mkdir()
prompt = ('以下のメモを読んで、最後の質問に一行の日本語で答えてください。\n'
          '検証用コード: BLUE-731。\n' +
          'This is a short test note. Keep the verification code in mind while reading the remaining notes.\n' * args.note_repetitions +
          '質問: 検証用コード、日本の首都、17+25の答えを教えてください。説明は不要です。')
(folder / 'prompt.txt').write_text(prompt)
if args.binary:
    model = LAB / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf'
    launcher = ['numactl', '--physcpubind=0-23', '--interleave=0,1', args.binary,
                '-m', str(model), '-t', args.threads, '-tb', args.threads_batch,
                '-ngl', '0', '-c', '4096', '-b', args.batch, '-ub', args.ubatch,
                '--load-mode', args.load_mode, '--cpu-mask', args.cpu_mask, '--cpu-strict', '1',
                '--cpu-mask-batch', args.cpu_mask_batch, '--cpu-strict-batch', '1', '--jinja', '--offline']
else:
    launcher = [str(LAB / 'run_deepseek_flash.sh')]
command = launcher + ['-f', str(folder / 'prompt.txt'),
           '--single-turn', '--simple-io', '--color', 'off', '--no-display-prompt',
           '--reasoning', 'off', '--temp', '0', '--seed', '42', '-n', '64', '--perf']
report = {'arguments': vars(args), 'command': command, 'started_unix': time.time(), 'observations': [],
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
if args.binary:
    binary = Path(args.binary).resolve()
    artifacts = [binary] + [p for p in binary.parent.glob('*.so*') if not p.is_symlink()]
else:
    artifacts = [LAB / 'run_deepseek_flash.sh']
report['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
print(json.dumps({'results': str(folder), 'command': command}), flush=True)
start, progress = time.monotonic(), 0
with (folder / 'response.txt').open('w') as out, (folder / 'stderr.log').open('w') as err:
    child = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
    try:
        while child.poll() is None:
            sample = snapshot(child.pid)
            if sample:
                masks = collections.Counter()
                for path in Path(f'/proc/{child.pid}/task').glob('*/status'):
                    try:
                        for line in path.read_text().splitlines():
                            if line.startswith('Cpus_allowed_list:'):
                                masks[line.split(':', 1)[1].strip()] += 1
                    except OSError:
                        pass
                sample['thread_affinity_mask_counts'] = dict(masks)
                sample['elapsed_seconds'] = time.monotonic() - start
                report['observations'].append(sample)
            elapsed = time.monotonic() - start
            if elapsed >= progress:
                print(json.dumps({'elapsed_seconds': round(elapsed), 'sample': sample}), flush=True)
                (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
                progress = elapsed + 30
            if elapsed > 1200:
                report['timeout'] = True
                break
            time.sleep(2)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        report['exit_code'] = child.wait()
report['finished_unix'] = time.time()
report['wall_seconds'] = time.monotonic() - start
report['response'] = (folder / 'response.txt').read_text()
report['expected_strings_present'] = {s: s in report['response'] for s in ['BLUE-731', '東京', '42']}
report['semantic_checks_passed'] = all(report['expected_strings_present'].values())
stderr = (folder / 'stderr.log').read_text()
report['timing_lines'] = [line for line in stderr.splitlines() if 'common_perf_print:' in line or 'llama_perf_context_print:' in line]
report['timings_ms'] = {
    name: float(match.group(1))
    for name in ('load', 'prompt eval', 'eval')
    if (match := re.search(r'(?:llama_perf_context_print|common_perf_print):\s+' + name + r' time\s*=\s*([\d.]+) ms', stderr))
}
(folder / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'saved': str(folder / 'summary.json'), 'exit_code': report['exit_code'],
                  'response': report['response'], 'stderr_tail': (folder / 'stderr.log').read_text()[-4000:]}, ensure_ascii=False), flush=True)
raise SystemExit(0 if report['exit_code'] == 0 and report['semantic_checks_passed'] else 1)
