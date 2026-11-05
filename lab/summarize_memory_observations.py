#!/usr/bin/env python3
"""Summarize thresholded load samples without treating weight as wall time."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent / 'results/bottleneck-observation-20261019'
results = {}
for phase in ('decode', 'prefill'):
    csvpath = root / (phase + '-memory-symbols.csv')
    samplespath = root / (phase + '-memory-samples.txt')
    if not csvpath.exists() or not samplespath.exists():
        continue
    capture = json.loads((root / (phase + '-memory-1') / 'capture.json').read_text())
    if not capture['valid_sample']:
        raise ValueError('Invalid capture')
    text = csvpath.read_text()
    lost = int(re.search(r'Total Lost Samples:\s*(\d+)', text).group(1))
    if lost:
        raise ValueError('Lost samples')
    declared_weight = int(re.search(r'Total weight\s*:\s*(\d+)', text).group(1))
    symbols = collections.defaultdict(lambda: [0, 0])
    sources = collections.defaultdict(lambda: [0, 0])
    categories = collections.defaultdict(lambda: [0, 0])
    for row in csv.reader(text.splitlines(), delimiter=';'):
        if not row or row[0].startswith('#') or '%' not in row[0]:
            continue
        count, weight = int(row[1]), int(row[2])
        source, symbol = row[3].strip(), row[4].strip()
        cat = 'other'
        if 'ggml_vec_dot_q8_0' in symbol:
            cat = 'Q8_dot'
        elif 'ggml_vec_dot_iq2_xxs' in symbol:
            cat = 'IQ2_dot'
        elif 'tinyBLAS_Q0_AVX' in symbol:
            cat = 'Q8_tinyBLAS'
        elif 'tiled_run_microtile_avx<32, false, 128' in symbol:
            cat = 'IQ2_tiled'
        for dest, key in ((symbols, symbol), (sources, source), (categories, cat)):
            dest[key][0] += count; dest[key][1] += weight
    all_weights = []
    per_source = collections.defaultdict(list)
    for line in samplespath.read_text().splitlines():
        match = re.match(r'^\s*\d+\s+\d+\s+.*\|LVL ([^|]+)\|.*?\s+(\d+)\s*$', line)
        if not match:
            raise ValueError('Unparsed sample line: ' + line[:120])
        source, weight = match.group(1).strip(), int(match.group(2))
        all_weights.append(weight); per_source[source].append(weight)
    total_samples = sum(n for n, w in sources.values())
    if sum(w for n, w in sources.values()) != declared_weight or sum(all_weights) != declared_weight or total_samples != len(all_weights):
        raise ValueError('Sample/weight counts disagree')
    def percentile(data, p):
        return sorted(data)[min(len(data) - 1, int((len(data) - 1) * p))]
    def describe(data):
        return {'count': len(data), 'mean_use_latency_cycles': statistics.mean(data),
                'median_use_latency_cycles': statistics.median(data),
                'p95_use_latency_cycles': percentile(data, .95)}
    def rows(data):
        return [{'name': key, 'samples': n, 'sample_share': n / total_samples,
                 'sum_use_latency_cycles': w, 'use_latency_weight_share': w / declared_weight,
                 'mean_use_latency_cycles': w / n} for key, (n, w) in sorted(data.items(), key=lambda kv: -kv[1][1])]
    results[phase] = {'valid': True, 'lost_samples': lost, 'capture': capture['arguments'],
                      'all_sample_statistics': describe(all_weights),
                      'sources': rows(sources), 'categories': rows(categories), 'symbols': rows(symbols),
                      'source_distributions': {s: describe(weights) for s, weights in per_source.items()},
                      'caveat': 'Randomly selected loads with use-latency >30 cycles, period1000. Weights are summed sampled use-latency, not elapsed time or an additive bottleneck share. LFB hits can hide the eventual backing source. Includes CPU pipeline queuing; not pure DRAM latency.'}
    print(json.dumps({'phase': phase, 'all_sample_statistics': results[phase]['all_sample_statistics'],
                      'sources': results[phase]['sources'][:4], 'categories': results[phase]['categories']}, ensure_ascii=False))
(root / 'memory-summary.json').write_text(json.dumps(results, indent=2) + '\n')
