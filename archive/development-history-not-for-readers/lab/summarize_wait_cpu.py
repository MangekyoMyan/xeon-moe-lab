#!/usr/bin/env python3
"""Report CPU use in matched interior windows of the spin-count experiments."""
import json
from pathlib import Path
import statistics

lab = Path(__file__).resolve().parent
results = []
for name, stamp, marker, length in [
    ('decode', '20261019T133330525248Z', 'generation run 1/3', 10),
    ('prefill', '20261019T133738076468Z', 'prompt run 1/2', 30),
]:
    comparison = lab / 'results' / ('cpu-abba-' + stamp) / 'comparison.json'
    data = json.loads(comparison.read_text())
    rows = []
    for run in data['runs']:
        raw = json.loads(Path(run['summary']).read_text())
        start = next(p['elapsed_seconds'] for p in raw['phases'] if p['message'].endswith(marker)) + 1
        samples = [s for s in raw['observations'] if start <= s['elapsed_seconds'] <= start + length]
        first, last = samples[0], samples[-1]
        wall = last['elapsed_seconds'] - first['elapsed_seconds']
        if wall < length - 3: raise RuntimeError('Incomplete telemetry window')
        rows.append({'variant': run['variant'], 'summary': run['summary'], 'wall_seconds': wall,
                     'cpu_equivalent': (last['cpu_seconds'] - first['cpu_seconds']) / wall,
                     'disk_bytes': last['disk_read_bytes'] - first['disk_read_bytes']})
    medians = {v: statistics.median(r['cpu_equivalent'] for r in rows if r['variant'] == v) for v in 'AB'}
    results.append({'phase': name, 'rows': rows, 'median_cpu_equivalent': medians,
                    'cpu_equivalent_change_percent': 100 * (medians['B'] / medians['A'] - 1),
                    'throughput_change_percent': data['candidate_gain_percent']})
out = lab / 'results/optimization-memory-sync-20261019/wait-cpu-summary.json'
out.write_text(json.dumps({'runs': results,
    'scope': 'Approximate interior windows from 1-second process telemetry. CPU time is not energy or useful compute.'}, indent=2) + '\n')
print(json.dumps(results, indent=2))
