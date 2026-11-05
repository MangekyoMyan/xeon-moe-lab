#!/usr/bin/env python3
"""Reaggregate measured data without loading a model or requiring root."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def benchmark(out, tag):
    runs = []
    for folder in sorted(out.glob(tag + '-[0-9]-[AB]')):
        report = json.loads((folder / 'run.json').read_text())
        assert report['exit_code'] == 0
        for row in json.loads((folder / 'stdout.txt').read_text()):
            phase = 'pp' + str(row['n_prompt']) if row['n_prompt'] else 'tg' + str(row['n_gen'])
            runs.append({'run': folder.name, 'variant': folder.name[-1], 'phase': phase, 'tokens': row['n_prompt'] or row['n_gen'], 'samples_ns': row['samples_ns'], 'avg_ts': row['avg_ts']})
    phases = {}
    for phase in sorted({r['phase'] for r in runs}):
        data = {}
        for label in ('A', 'B'):
            selected = [r for r in runs if r['phase'] == phase and r['variant'] == label]
            ns = [n for r in selected for n in r['samples_ns']]
            assert ns
            tokens = selected[0]['tokens']
            data[label] = {'runs': len(selected), 'samples': len(ns), 'tokens_per_sample': tokens, 'pooled_tokens_s': tokens * len(ns) * 1e9 / sum(ns), 'mean_run_tokens_s': statistics.mean(r['avg_ts'] for r in selected), 'min_run_tokens_s': min(r['avg_ts'] for r in selected), 'max_run_tokens_s': max(r['avg_ts'] for r in selected)}
        data['speed_gain_percent'] = (data['B']['pooled_tokens_s'] / data['A']['pooled_tokens_s'] - 1) * 100
        data['time_reduction_percent'] = (1 - data['A']['pooled_tokens_s'] / data['B']['pooled_tokens_s']) * 100
        phases[phase] = data
    return {'tag': tag, 'runs': runs, 'phases': phases}


def kernels(out):
    records = []
    for folder in sorted(out.iterdir()):
        match = re.fullmatch(r'(screen|rows|prefetch|packed)-k(\d+)-m(\d+)-t(\d+)', folder.name)
        if not match:
            continue
        data = {}
        for line in (folder / 'stdout.txt').read_text().splitlines():
            row = json.loads(line)
            if 'variant' in row:
                data.setdefault(row['variant'], []).append(row)
        reference = statistics.median(x['ms_per_matvec'] for x in data['baseline'])
        for variant, rows in data.items():
            median = statistics.median(x['ms_per_matvec'] for x in rows)
            records.append({'stage': match[1], 'k': int(match[2]), 'm': int(match[3]), 'threads': int(match[4]), 'variant': variant, 'samples': len(rows), 'median_ms': median, 'speed_ratio': reference / median, 'median_weight_GBs': statistics.median(x['weight_GBs'] for x in rows), 'median_GOPS': statistics.median(x['GOPS'] for x in rows)})
    return records


def response_perf(folder):
    text = (folder / 'stderr.txt').read_text()
    patterns = {'load_ms': r'\bload time\s*=\s*([\d.]+) ms', 'prompt_ms': r'prompt eval time\s*=\s*([\d.]+) ms', 'eval_ms': r'\beval time\s*=\s*([\d.]+) ms /\s*\d+ runs', 'eval_runs': r'\beval time\s*=.*?/\s*(\d+) runs'}
    result = {'run': folder.name}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            result[key] = float(matches[-1])
    result['warm_response_ms'] = result.get('prompt_ms', 0) + result.get('eval_ms', 0)
    report = json.loads((folder / 'run.json').read_text())
    assert report['exit_code'] == 0
    observations = report['observations']
    result['minor_faults_observed'] = observations[-1]['minor_faults']
    result['disk_read_bytes_observed'] = observations[-1]['disk_read_bytes']
    result['process_wall_s'] = report['elapsed_s']
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'results/mxfp4-opt-20261023')
    args = parser.parse_args()
    out = args.output
    primary = benchmark(out, 'accepted-final24')
    secondary = benchmark(out, 'accepted-series128')
    kernel_rows = kernels(out)
    startup = [response_perf(p) for p in sorted(out.glob('startup-reuse-*-r[01]'))]
    startup_summary = {}
    for mode in ('baseline', 'parallel', 'reuse'):
        rows = [r for r in startup if r['run'].startswith('startup-reuse-' + mode + '-')]
        startup_summary[mode] = {key: statistics.mean(r[key] for r in rows) for key in ('load_ms', 'minor_faults_observed', 'warm_response_ms')}
    responses = [response_perf(p) for p in sorted(out.glob('launcher-verified-*')) if p.is_dir()]
    temperatures = []
    audits = []
    for path in sorted(out.glob('*/run.json')):
        record = json.loads(path.read_text())
        for row in record.get('observations', []):
            temperatures.extend(row.get('cpu_temperatures_C', {}).values())
        if path.parent.name.startswith(('accepted-', 'launcher-verified-')):
            audits.append({'run': path.parent.name, 'exit_code': record['exit_code'], 'boot_id': record['boot_id']})
    summary = {'primary': primary, 'secondary': secondary, 'startup': startup_summary, 'startup_runs': startup, 'launcher_responses': responses, 'kernel_comparison_sets': len({(r['stage'], r['k'], r['m'], r['threads']) for r in kernel_rows}), 'kernel_timed_samples': sum(r['samples'] for r in kernel_rows), 'max_observed_cpu_temperature_C': max(temperatures), 'accepted_run_audit': audits, 'notes': ['Only compare identical benchmark command sequences. llama-bench uses process-global std::rand(); preceding pp tests change the later decode token series.', 'Primary: pp128,pp512,tg256; eight ABBA/BAAB runs, two repetitions each. Secondary: pp128,tg256; four ABBA runs.', 'Profiling runs and the excluded original truncated Japanese batch are not included in acceptance throughput.']}
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    write_csv(out / 'KERNEL_SUMMARY.csv', kernel_rows)
    write_csv(out / 'STARTUP_SUMMARY.csv', startup)
    flat = []
    for group in (primary, secondary):
        for row in group['runs']:
            for index, elapsed in enumerate(row['samples_ns']):
                flat.append({'suite': group['tag'], 'run': row['run'], 'variant': row['variant'], 'phase': row['phase'], 'sample': index, 'ns': elapsed, 'tokens_s': row['tokens'] * 1e9 / elapsed})
    write_csv(out / 'MODEL_BENCHMARKS.csv', flat)
    print(json.dumps({'primary': primary['phases'], 'secondary': secondary['phases'], 'startup': startup_summary}, indent=2))


if __name__ == '__main__':
    main()
