#!/usr/bin/env python3
"""Collect exact experiment evidence and verify the active runtime is intact."""
import hashlib
import json
from pathlib import Path
import statistics
from summarize_bottleneck_observations import summarize

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-numa-rows-20261020'
experiment = json.loads((out / 'experiment.json').read_text())
comparisons = []
seen = set()
for index in sorted(out.glob('*-index.json')):
    for row in json.loads(index.read_text()):
        path = Path(row.get('saved', ''))
        if not path.is_file() or path.name != 'comparison.json' or path in seen:
            continue
        data = json.loads(path.read_text())
        if 'candidate_gain_percent' not in data:
            continue
        seen.add(path)
        resource = []
        for run in data['runs']:
            result = json.loads(Path(run['summary']).read_text())
            observations = result.get('observations', [])
            phase = next((r for r in result['phases'] if 'generation run 1/' in r['message'] or 'prompt run 1/' in r['message']), None)
            resource.append({'variant': run['variant'], 'summary': run['summary'],
                'max_rss_gib': max(s['rss_kib'] for s in observations) / 1024**2,
                'max_hwm_gib': max(s['hwm_kib'] for s in observations) / 1024**2,
                'first_timed_marker_s_approx': phase['elapsed_seconds'] if phase else None,
                'max_disk_read_bytes': run['max_observed_read_bytes'],
                'observer_cpu_s': result.get('observer_cpu_seconds')})
        comparisons.append({'name': row['name'], 'path': str(path),
            'order': data['order'], 'process_means': data['process_mean_tok_s'],
            'medians': data['median_process_mean_tok_s'], 'gain_percent': data['candidate_gain_percent'],
            'same_boot': data['same_boot'], 'warm_cache': data['warm_cache_observed'],
            'all_candidate_means_above_baseline': data['candidate_all_above_baseline'],
            'environment': data.get('effective_wait_environment'), 'resources': resource})

validations = {}
for name in ('model-validation', 'moe-model-validation'):
    path = out / name / 'comparison.json'
    if not path.exists():
        raise RuntimeError('Missing full model validation: ' + name)
    data = json.loads(path.read_text())
    assert all(data.get(k) for k in ('complete', 'logits_bit_identical', 'responses_bit_identical', 'token_counts_identical', 'same_boot', 'warm_cache'))
    validations[name] = data
observations = {}
for variant in 'AB':
    result = summarize(out / ('locality-observe-' + variant))
    if not result['valid']:
        raise RuntimeError('Invalid locality sample')
    observations[variant] = result

freeze = json.loads((out / 'freeze-before.json').read_text())
source = lab / 'llama-private-prefetch'
after_source = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}
after_settings = {p: Path(p).read_text().strip() for p in freeze['settings']}
after_runtime = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in experiment['active_sha256']}
integrity = {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'same_boot': freeze['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'source_files_checked': len(after_source), 'source_unchanged': freeze['source_hashes'] == after_source,
    'settings_count': len(after_settings), 'settings_unchanged': freeze['settings'] == after_settings,
    'active_artifacts_unchanged_before_metadata_update': experiment['active_sha256'] == after_runtime,
    'settings': after_settings, 'active_sha256_before_metadata_update': after_runtime}
assert all(integrity[k] for k in ('same_boot', 'source_unchanged', 'settings_unchanged', 'active_artifacts_unchanged_before_metadata_update'))
(out / 'freeze-after.json').write_text(json.dumps(integrity, indent=2) + '\n')

report = {'date_jst': '2026-10-20', 'experiment': experiment, 'comparisons': comparisons,
    'full_model_validation': validations, 'locality_observations': observations,
    'placement': {mode: json.loads((out / ('placement-' + mode) / 'placement.json').read_text()) for mode in ('1', '2')},
    'numa_unit_checks': json.loads((out / 'checks.json').read_text()),
    'moe_unit_checks': json.loads((out / 'moe-checks.json').read_text()),
    'integrity': integrity,
    'limits': [
        'Synthetic warm single-stream generation: p0,n32,depth0,12 workers,3 repetitions per process.',
        'Two independent processes per variant and order; no statistical confidence interval is claimed.',
        'Locality counters are a separate approximate 18-second window per variant. Retired DRAM load uops are not transfer bytes.',
        'Q8 copies and row scheduling cover 107 selected weight tensors, not the whole model.',
        'Full-vocabulary logits match at 34 positions of one fixed 160-token sequence; not a universal all-prompts guarantee.',
        'One fixed Japanese response, no cold-load/RAM expansion/GPU/power or broad-context speed claim.',
        'Candidate prefill512 performance checks were skipped when the small decode gains failed the screening criterion; no prefill512 speed equivalence is claimed.']}
(out / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    'comparisons': [{k: c[k] for k in ('name', 'gain_percent', 'medians', 'all_candidate_means_above_baseline')} for c in comparisons],
    'responses': {k: v['generation_gain_percent'] for k, v in validations.items()},
    'locality': {k: {key: value for key, value in v['metrics'].items() if key in (
        'remote_share_of_local_plus_remote_retired_dram_load_uops', 'dram_read_GB_s',
        'cpu_temperature_max_C', 'throttle_delta_sum_including_smt_duplicates', 'window_disk_read_bytes',
        'major_faults', 'minor_faults')} for k, v in observations.items()},
    'runtime_intact': True}, indent=2))
