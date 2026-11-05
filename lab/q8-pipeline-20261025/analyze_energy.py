#!/usr/bin/env python3
"""Summarize busy CPU intervals from existing RAPL telemetry, not wall power."""
import argparse
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent.parent / 'results/q8-pipeline-20261025'

def summarize(label):
    report = json.loads((OUT / label / 'run.json').read_text())
    threads = report['benchmarks'][0]['n_threads']
    intervals = []
    for first, last in zip(report['samples'], report['samples'][1:]):
        if not first.get('process') or not last.get('process'):
            continue
        seconds = last['monotonic'] - first['monotonic']
        cpu_seconds = last['process']['cpu_seconds'] - first['process']['cpu_seconds']
        if seconds <= 0 or cpu_seconds < 0.85 * threads * seconds:
            continue
        watts = {'package': 0.0, 'dram': 0.0}
        for key, initial in first['energy'].items():
            final = last['energy'][key]
            joules = ((final['uj'] - initial['uj']) % initial['range']) / 1e6
            kind = 'dram' if initial['name'] == 'dram' else 'package'
            watts[kind] += joules / seconds
        intervals.append({'seconds': seconds, **watts})
    assert intervals
    duration = sum(x['seconds'] for x in intervals)
    return {'label': label, 'threads': threads, 'busy_sample_seconds': duration,
            'package_W_mean': sum(x['package'] * x['seconds'] for x in intervals) / duration,
            'dram_W_mean': sum(x['dram'] * x['seconds'] for x in intervals) / duration,
            'package_plus_dram_W_median': statistics.median(x['package'] + x['dram'] for x in intervals),
            'max_cpu_C': max(max(x['temperatures_C'].values()) for x in report['samples']),
            'note': 'Intervals selected by observed process CPU time >=85% of requested workers. CPU package and DRAM RAPL estimates only; excludes motherboard, drives, fans, GPU, PSU losses.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary')
    args = parser.parse_args()
    cases = json.loads((OUT / args.summary).read_text())
    results = [summarize(row['label']) for row in cases]
    destination = OUT / (Path(args.summary).stem + '-energy.json')
    destination.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
