#!/usr/bin/env python3
"""Summarize diagnostic graph intervals without calling them CPU stall fractions."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results' / ROOT.name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    args = parser.parse_args()
    folder = OUT / args.label
    text = (folder / 'stderr.txt').read_text()
    assert 'OP_TIMING_DROPPED\t0' in text
    keys = ['operation', 'weight', 'type', 'K', 'M', 'N1', 'N2', 'experts', 'fused',
            'calls', 'total_us', 'worker0_compute_us', 'min_us', 'max_us']
    rows = []
    for line in text.splitlines():
        if not line.startswith('OP_TIMING\t'):
            continue
        values = line.split('\t')[1:]
        assert len(values) == len(keys)
        record = dict(zip(keys, values))
        for key in keys[3:]:
            record[key] = int(record[key])
        record['worker0_outer_barrier_us'] = record['total_us'] - record['worker0_compute_us']
        if record['type'] == 'q8_0':
            payload = record['K'] * record['M'] * 34 / 32
            if record['operation'] == 'MUL_MAT_ID':
                payload *= record['N1']
            payload *= record['calls'] * (1 + record['fused'])
            record['selected_weight_payload_GB'] = payload / 1e9
            record['payload_per_interval_GB_s'] = payload / (record['total_us'] * 1000)
        rows.append(record)
    total = sum(row['total_us'] for row in rows)
    for row in rows:
        row['graph_interval_pct'] = row['total_us'] / total * 100
    rows.sort(key=lambda r: -r['total_us'])
    report = {'label': args.label, 'interval_seconds': total / 1e6, 'rows': rows,
              'note': 'Worker-zero wall-clock intervals include internal operation barriers and the following graph barrier. Instrumented diagnostic timing, not acceptance speed. Payload rate is selected tensor bytes / interval, not a memory-controller measurement. Other workers may begin their next operation slightly before worker zero.'}
    (OUT / (args.label + '-analysis.json')).write_text(json.dumps(report, indent=2) + '\n')
    for row in rows[:20]:
        print(row['operation'], row['weight'], 'share_pct', round(row['graph_interval_pct'], 2),
              'payload_GB_s', round(row.get('payload_per_interval_GB_s', 0), 2),
              'worker0_outer_wait_pct', round(row['worker0_outer_barrier_us'] / row['total_us'] * 100, 2))

if __name__ == '__main__':
    main()
