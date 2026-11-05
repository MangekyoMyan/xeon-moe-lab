#!/usr/bin/env python3
"""Collect measured results for the private SSSE3 unpack optimization."""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

lab = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--batch-threads', type=int, choices=[12,24], required=True)
ap.add_argument('--before-functional', default='long971-baseline-resumed')
ap.add_argument('--after-functional', default='long971-ssse3-resumed')
args = ap.parse_args()
runs = {}
for p in (lab / 'results').glob('deepseek-*/summary.json'):
    s = json.loads(p.read_text())
    label = s.get('arguments', {}).get('label')
    if label:
        runs.setdefault(label, []).append((p, s))

def benchmark(label):
    pairs = runs[label]
    assert len(pairs) == 1, label
    p, s = pairs[0]
    assert s['exit_code'] == 0 and len(s['benchmarks']) == 1, label
    b = s['benchmarks'][0]
    return {'source': str(p), 'arguments': s['arguments'], 'command': s['command'],
            'seconds': b['avg_ns']/1e9, 'tok_s': b['avg_ts'], 'stddev_tok_s': b['stddev_ts'],
            'samples_tok_s': b['samples_ts'],
            'disk_read_bytes': max(x['disk_read_bytes'] for x in s['observations']),
            'peak_rss_gib': max(x['rss_kib'] for x in s['observations'])/2**20,
            'artifact_sha256': s.get('artifact_sha256', {})}

def functional(label):
    p, s = runs[label][0]
    assert len(runs[label]) == 1 and s['exit_code'] == 0 and s['semantic_checks_passed']
    timings = '\n'.join(s['timing_lines'])
    prompt_tokens = int(re.search(r'prompt eval time\s*=.*?ms /\s*(\d+) tokens', timings).group(1))
    return {'source': str(p), 'command': s['command'], 'timings_ms': s['timings_ms'],
            'boot_id': s.get('boot_id'), 'artifact_sha256': s.get('artifact_sha256', {}),
            'prompt_tokens': prompt_tokens,
            'response': s['response'], 'checks': s['expected_strings_present'],
            'disk_read_bytes': max(x['disk_read_bytes'] for x in s['observations']),
            'timing_lines': s['timing_lines']}

prefill_labels = ['long512-ub256-baseline', 'long512-ub512', 'long512-ssse3-ub512',
                  'long512-ssse3-threads24-screen', 'long512-ssse3-threads24-confirm']
decode_labels = ['decode12-before-ssse3', 'decode12-ssse3', 'decode10-ssse3',
                 'decode8-ssse3', 'decode24-ssse3']
report = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model': 'DeepSeek-V4-Flash IQ2_XXS-XL, existing 73.126 GiB GGUF',
    'hardware': '2x Xeon E5-2630L v2, 12 physical cores / 24 logical CPUs, AVX and SSSE3, no AVX2',
    'source_commit': '436f6f89e1e581249900b37a5b8a12a36a6d0912',
    'patches': ['patches/completion-template-kwargs.patch', 'patches/ssse3-iq-sign-unpack.patch'],
    'prefill': {label: benchmark(label) for label in prefill_labels},
    'decode': {label: benchmark(label) for label in decode_labels},
    'functional': {label: functional(label) for label in [args.before_functional, args.after_functional]},
    'adopted_launcher_smoke': functional('adopted-ssse3-launcher'),
    'gpu': json.loads((lab/'results/gtx550ti-inventory.json').read_text()),
    'memory_counters': json.loads((lab/'results/long512-optimization-measurement-notes.json').read_text()),
    'profile': str(lab/'results/long512-ub256-profile.txt'),
    'validation': {'exact_byte_cases': 262144, 'IQ2_IQ3_matmul_cases_passed': 212,
                   'quantization_failures': 0, 'self_review': str(lab/'results/avx-unpack-self-review.json')},
    'adopted': {'build': str(lab/'build-avx-unpack'), 'threads_generation': 12,
                'threads_prompt': args.batch_threads, 'batch': 512, 'ubatch': 512,
                'OpenMP': True, 'BLAS': False, 'load_mode': 'mmap',
                'gpu_layers': 0,
                'selection_reason': 'Prioritize long input processing; keep generation on 12 workers. On the measured 64-token input, 24 prompt workers are slightly slower than 12.',
                'placement': 'OS interleave nodes 0,1; strict per-thread CPU pins',
                'launcher': str(lab/'run_deepseek_flash.sh'),
                'previous_launcher': str(lab/'run_deepseek_flash_before_ssse3.sh')},
    'limitations': ['Warm file-cache measurements; cold HDD startup is not improved by this patch.',
                    'Synthetic token streams are matched using a fresh process and identical workload arguments.',
                    '24-thread screening has one timed repetition; the confirmation has two.',
                    'Natural long-prompt comparison is one fixed prompt per build, not a broad quality evaluation.',
                    'Contexts beyond the tested 971-token fixed prompt remain unmeasured.',
                    'The initial ubatch-256 baseline includes a short 99 Hz sampling interval.',
                    'Synthetic long-prefill/decode sweeps preceded GPU installation; the resumed functional pair and short-prefill pair share the post-installation boot.',
                    'The interrupted long1011-ssse3 functional run is excluded; its prompt actually contained 971 tokens.'],
}
for label in ['short64-ssse3-threads12', 'short64-ssse3-threads24']:
    if label in runs: report['prefill'][label] = benchmark(label)
if 'short64-ssse3-threads12' in report['prefill'] and 'short64-ssse3-threads24' in report['prefill']:
    short12 = report['prefill']['short64-ssse3-threads12']
    short24 = report['prefill']['short64-ssse3-threads24']
    report['short_prompt_tradeoff'] = {
        'tokens': 64,
        'seconds_12_threads': short12['seconds'],
        'seconds_24_threads': short24['seconds'],
        'extra_seconds_with_24_threads': short24['seconds'] - short12['seconds'],
    }
a=report['prefill']['long512-ub256-baseline']['tok_s']
b=report['prefill']['long512-ssse3-ub512']['tok_s']
c=report['prefill']['long512-ssse3-threads24-confirm']['tok_s']
report['gains']={'prefill12_vs_previous_percent':(b/a-1)*100,
                 'prefill24_vs_previous_percent':(c/a-1)*100,
                 'isolated_SSSE3_prefill_percent':(b/report['prefill']['long512-ub512']['tok_s']-1)*100,
                 'decode12_change_percent':(report['decode']['decode12-ssse3']['tok_s']/report['decode']['decode12-before-ssse3']['tok_s']-1)*100}
before = report['functional'][args.before_functional]
after = report['functional'][args.after_functional]
assert before['boot_id'] and before['boot_id'] == after['boot_id'], 'Functional pair must share one boot.'
assert before['prompt_tokens'] == after['prompt_tokens'] == 971
report['gains']['natural_prompt_time_reduction_percent'] = (1-after['timings_ms']['prompt eval']/before['timings_ms']['prompt eval'])*100
report['gains']['natural_prompt_throughput_increase_percent'] = (before['timings_ms']['prompt eval']/after['timings_ms']['prompt eval']-1)*100
report['limitations'].append('The fixed-prompt answer has only eight timed generation steps: evaluation time was 3.462 s before and 3.822 s after. Use the separate matched 32-token, three-repetition decode sweep for the generation comparison; no generation speedup is claimed.')
report['launcher_sha256'] = hashlib.sha256((lab/'run_deepseek_flash.sh').read_bytes()).hexdigest()
report['final_artifact_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (lab/'build-avx-unpack/bin').iterdir() if p.is_file() and not p.is_symlink()}
p=lab/'results/deepseek-avx-optimization.json'
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'saved':str(p),'gains':report['gains'],'adopted':report['adopted']},indent=2))
