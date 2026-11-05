#!/usr/bin/env python3
"""Join benchmark phase timestamps with independent system-wide perf counters."""
import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

RESULTS = Path(__file__).resolve().parent / 'results'


def perf_intervals(path):
    lines = path.read_text().splitlines()
    header = next((line for line in lines if line.startswith('# started on ')), None)
    if not header:
        return []
    origin = datetime.datetime.strptime(header.removeprefix('# started on '), '%a %b %d %H:%M:%S %Y').replace(tzinfo=ZoneInfo('Asia/Tokyo')).timestamp()
    intervals = {}
    for line in lines:
        fields = line.split(';')
        if len(fields) < 6 or not fields[0] or line.startswith('#'):
            continue
        try:
            t, count = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        row = intervals.setdefault(t, {'end_unix': origin + t, 'events': {}})
        row['events'][fields[3]] = count * (2**20 if fields[2] == 'MiB' else 1)
    previous = origin
    for row in intervals.values():
        row['start_unix'] = previous
        row['seconds'] = row['end_unix'] - previous
        previous = row['end_unix']
    return list(intervals.values())


def main():
    counter_sources = [(str(p), perf_intervals(p)) for p in RESULTS.glob('deepseek-*/perf-*.csv')]
    report = {'method': 'llama-bench random-token tests; timestamps sampled at 1 s; counter windows trim 2 s at either end; IMC counts include the whole host; GB uses 10^9 bytes.', 'runs': []}
    for path in sorted(RESULTS.glob('deepseek-*/summary.json')):
        run = json.loads(path.read_text())
        if 'benchmarks' not in run:
            continue
        row = {'source': str(path), 'arguments': run['arguments'], 'wall_seconds': run['wall_seconds'], 'tests': []}
        intervention = path.parent / 'blas-affinity-intervention.json'
        if intervention.exists():
            row['diagnostic_intervention'] = json.loads(intervention.read_text())
            row['caveat'] = 'The first depth-256 prefill was modified while running. Its wall time is not a benchmark of a fixed configuration. Later generation samples follow the recorded BLAS worker affinity change.'
        phases = run['phases']
        for idx, bench in enumerate(run['benchmarks'], 1):
            starts = [p for p in phases if re.match(rf'llama-bench: benchmark {idx}/\d+: (generation|prompt) run 1/', p['message'])]
            item = {k: bench[k] for k in ['n_threads', 'n_prompt', 'n_gen', 'n_depth', 'n_batch', 'n_ubatch', 'load_mode', 'avg_ts', 'stddev_ts', 'samples_ts'] if k in bench}
            if starts:
                start = starts[0]['unix_time']
                duration = sum(bench['samples_ns']) / 1e9
                end = start + duration
                samples = [s for s in run['observations'] if start + 2 <= s['unix_time'] <= end - 2]
                item['timed_window_unix'] = [start, end]
                if len(samples) >= 2:
                    a, b = samples[0], samples[-1]
                    item['observed_disk_read_bytes'] = b['disk_read_bytes'] - a['disk_read_bytes']
                    item['cpu_core_equivalents'] = (b['cpu_seconds'] - a['cpu_seconds']) / (b['unix_time'] - a['unix_time'])
                    item['resident_gib_by_node'] = {n: k / 2**20 for n, k in b['resident_kib_by_node'].items()}
                    item['rss_gib'] = b['rss_kib'] / 2**20
                for source, intervals in counter_sources:
                    selected = [v for v in intervals if v['start_unix'] >= start + 2 and v['end_unix'] <= end - 2]
                    if not selected:
                        continue
                    seconds = sum(v['seconds'] for v in selected)
                    sums = {}
                    for interval in selected:
                        for event, count in interval['events'].items():
                            sums[event] = sums.get(event, 0) + count
                    metrics = {'source': source, 'sampled_seconds': seconds}
                    for event, key in [('uncore_imc_*/cas_count_read/', 'dram_read_GB_per_s'), ('uncore_imc_*/cas_count_write/', 'dram_write_GB_per_s')]:
                        if event in sums:
                            metrics[key] = sums[event] / seconds / 1e9
                    if sums.get('cycles:u'):
                        metrics['user_instructions_per_cycle'] = sums.get('instructions:u', 0) / sums['cycles:u']
                    item.setdefault('counters', []).append(metrics)
            row['tests'].append(item)
        report['runs'].append(row)
    destination = RESULTS / 'deepseek-comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
