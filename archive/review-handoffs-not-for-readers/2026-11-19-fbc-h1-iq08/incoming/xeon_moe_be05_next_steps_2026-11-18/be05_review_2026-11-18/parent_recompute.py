#!/usr/bin/env python3
"""Parent's separate read-only recomputation from be05 RUN.json files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path(__file__).parent / 'source')
a = p.parse_args()
q = a.source / 'q8-avx1'
arms = ['control', 'candidate', 'candidate', 'control', 'candidate', 'control', 'control', 'candidate']
runs = [json.loads((q / 'runs-v2' / f'timing-{i}-{mode}' / 'RUN.json').read_text()) for i, mode in enumerate(arms)]
pair_indices = [(0, 1), (3, 2), (5, 4), (6, 7)]
published = json.loads((q / 'FFN_ANALYSIS.json').read_text())
medians, spread, all_samples = {}, [], 0
for i, run in enumerate(runs):
    for b in run['blocks']:
        values = [(s['end_ns'] - s['begin_ns']) / s['calls'] / 1e6 for s in b['samples']]
        assert len(values) == 5
        all_samples += len(values)
        medians[i, b['cell'], b['threads']] = median(values)
        spread.append({'process': i, 'mode': arms[i], 'cell': b['cell'], 'threads': b['threads'],
                       'first_fixture': b['cell'] == run['spec']['cells'][0],
                       'median_ms': median(values), 'sample_ms': values, 'max_over_min': max(values) / min(values)})
conditions = []
for cell in [1, 5, 6]:
    for threads in [12, 24]:
        ratios = [medians[b, cell, threads] / medians[c, cell, threads] for c, b in pair_indices]
        row = {'cell': cell, 'threads': threads, 'ratios': ratios, 'median': median(ratios),
               'wins': sum(x < 1 for x in ratios), 'min': min(ratios), 'max': max(ratios)}
        saved = next(x for x in published['rows'] if x['cell'] == cell and x['threads'] == threads)
        assert row['median'] == saved['median_time_ratio'] and row['wins'] == saved['pairs_faster']
        conditions.append(row)
first24 = [x['max_over_min'] for x in spread if x['threads'] == 24 and x['first_fixture']]
later24 = [x['max_over_min'] for x in spread if x['threads'] == 24 and not x['first_fixture']]

# Static logical work count for the IQ cells. These are not durations or a speed bound.
work = []
for threads in [12, 24]:
    calls = []
    for k, input_columns in [(5120, 1), (5120, 1), (2304, 6)]:
        blocks = k // 256
        for column in range(input_columns):
            calls += [((i + 1) * blocks // threads) - (i * blocks // threads) for i in range(threads)]
    work.append({'graph_threads': threads, 'from_float_calls': len(calls),
                 'blocks_per_call_histogram': dict(sorted(Counter(calls).items())),
                 'logical_Q8_K_blocks': sum(calls)})
assert all(x['logical_Q8_K_blocks'] == 94 for x in work)
logical_dot_blocks = 6 * (2304 * 20 + 2304 * 20 + 5120 * 9)
assert logical_dot_blocks == 829440

build = json.loads((q / 'BUILD.json').read_text())
numeric = json.loads((q / 'NUMERIC.json').read_text())
for v in ['control', 'candidate']:
    f = q / v / 'source/ggml/src/ggml-cpu/arch/x86/quants.c'
    assert hashlib.sha256(f.read_bytes()).hexdigest() == build['variants'][v]['source_sha256']
    assert '-mno-avx2' in build['variants'][v]['compile'] and '-mno-fma' in build['variants'][v]['compile']
    assert all(int(x['unequal_bytes']) == 0 for row in numeric['runs'] if row['mode'] == v for x in row['rows'])

result = {'status': 'pass', 'scope': 'Parent-authored raw timing recomputation and static work counts; no new engine run',
          'timing_samples': all_samples, 'paired_conditions': conditions,
          'process_sample_statistics': spread,
          'order_association': {'first24_count': len(first24), 'first24_range': [min(first24), max(first24)],
                                'first24_median': median(first24), 'later24_count': len(later24),
                                'later24_median': median(later24)},
          'static_work_counts': work, 'logical_weight_input_blocks': logical_dot_blocks,
          'folded_rhs_bytes_by_projection': [20 * 400, 20 * 400, 6 * 9 * 400],
          'limits': ['Work counts are not Amdahl time fractions or measured call traces',
                     'Order grouping is post hoc; no sample removed and no negative-control subtraction',
                     'Build/numeric successes are checked as recorded evidence, not recreated ELF/probe executions']}
print(json.dumps(result, ensure_ascii=False, indent=2))
