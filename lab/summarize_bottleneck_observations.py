#!/usr/bin/env python3
"""Summarize observational evidence, rejecting multiplexed core counters."""
import collections
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / 'results/bottleneck-observation-20261019'


def csvrows(path):
    return [r for r in csv.reader(path.read_text().splitlines()) if r and not r[0].startswith('#')]


def io(text):
    return {k: int(v.strip()) for line in text.splitlines() for k, v in [line.split(':')]}


def summarize(folder):
    obs = json.loads((folder / 'observations.json').read_text())
    cap = json.loads((folder / 'capture.json').read_text())
    if not cap.get('valid_sample') or len(obs.get('telemetry', [])) != obs['counter_seconds'] + 1 or obs.get('helper_exit_code') != 0 or any(c.get('exit_code') != 0 for c in obs['counters']):
        return {'name': folder.name, 'valid': False, 'reason': 'Capture not marked valid'}
    rows = csvrows(folder / 'core.csv')
    excluded = [','.join(r[2:-4]) for r in rows if float(r[-3]) < 99.9 or '<not' in r[0]]
    # Pending-load occupancy is supplemental, not needed for the stop/stall result.
    # Preserve the original capture's failure flag; report only fully-running events.
    optional = {'cycle_activity.cycles_l2_pending:u'}
    if any(e not in optional for e in excluded):
        return {'name': folder.name, 'valid': False, 'reason': 'Multiplexed or unavailable core counters'}
    events = {','.join(r[2:-4]): float(r[0]) for r in rows if ','.join(r[2:-4]) not in excluded}
    seconds = events['duration_time'] / 1e9
    metrics = {'seconds': seconds, 'cpu_equivalent': events['task-clock'] / 1000 / seconds,
               'ipc': events['instructions:u'] / events['cycles:u'] if 'cycles:u' in events else None,
               'user_cycles_per_cpu_second_GHz': events['cycles:u'] / (events['task-clock'] / 1000) / 1e9 if 'cycles:u' in events else None,
               'major_faults': events['major-faults'], 'minor_faults': events['minor-faults'],
               'migrations': events['cpu-migrations'], 'context_switches': events['context-switches']}
    for event in ['cycle_activity.stalls_total:u', 'cycle_activity.stalls_l2_pending:u', 'cycle_activity.cycles_l2_pending:u']:
        if event in events:
            metrics[event + '_fraction_of_cycles'] = events[event] / events['cycles:u']
    local = events.get('mem_load_uops_llc_miss_retired.local_dram:u')
    remote = events.get('mem_load_uops_llc_miss_retired.remote_dram:u')
    if local is not None and remote is not None:
        metrics['remote_share_of_local_plus_remote_retired_dram_load_uops'] = remote / (remote + local) if local + remote else None
        metrics['local_dram_retired_load_uops_per_second'] = local / seconds
        metrics['remote_dram_retired_load_uops_per_second'] = remote / seconds
    channel = collections.defaultdict(lambda: [0, 0])
    intervals = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in csvrows(folder / 'imc.csv'):
        if float(r[-3]) < 99.9 or '<not' in r[2]:
            raise ValueError('IMC event unavailable/multiplexed')
        timestamp, cpu, count = float(r[0]), r[1], float(r[2])
        event, runtime = ','.join(r[4:-4]), float(r[-4]) / 1e9
        kind = 'read' if 'umask=0x03' in event else 'write'
        key = f'{cpu}/{event.split("/")[0]}/{kind}'
        channel[key][0] += count
        channel[key][1] += runtime
        intervals[timestamp][cpu + '/' + kind] += count * 64 / runtime / 1e9
    rates = {key: n * 64 / sec / 1e9 for key, (n, sec) in channel.items()}
    metrics['dram_read_GB_s'] = sum(v for k, v in rates.items() if k.endswith('/read'))
    metrics['dram_write_GB_s'] = sum(v for k, v in rates.items() if k.endswith('/write'))
    metrics['socket0_read_GB_s'] = sum(v for k, v in rates.items() if k.startswith('CPU0/') and k.endswith('/read'))
    metrics['socket1_read_GB_s'] = sum(v for k, v in rates.items() if k.startswith('CPU6/') and k.endswith('/read'))
    telemetry = obs['telemetry']; a, b = telemetry[0], telemetry[-1]
    elapsed = b['monotonic'] - a['monotonic']
    threads = []
    for tid, first in a['threads'].items():
        last = b['threads'].get(tid)
        if last is None:
            continue
        s0, s1 = [int(x) for x in first['schedstat'].split()], [int(x) for x in last['schedstat'].split()]
        threads.append({'tid': tid, 'affinity_start': first['status']['Cpus_allowed_list'],
                        'affinity_end': last['status']['Cpus_allowed_list'],
                        'cpu_equivalent': (s1[0] - s0[0]) / 1e9 / elapsed,
                        'runqueue_wait_ms': (s1[1] - s0[1]) / 1e6,
                        'system_cpu_seconds': (last['stime_ticks'] - first['stime_ticks']) / obs['clk_tck']})
    metrics['thread_cpu_min'] = min(t['cpu_equivalent'] for t in threads)
    metrics['thread_cpu_max'] = max(t['cpu_equivalent'] for t in threads)
    metrics['thread_count'] = len(threads)
    metrics['thread_runqueue_wait_total_ms'] = sum(t['runqueue_wait_ms'] for t in threads)
    metrics['system_cpu_seconds'] = sum(t['system_cpu_seconds'] for t in threads)
    metrics['window_disk_read_bytes'] = io(b['io'])['read_bytes'] - io(a['io'])['read_bytes']
    metrics['window_disk_write_bytes'] = io(b['io'])['write_bytes'] - io(a['io'])['write_bytes']
    metrics['cpu_temperature_max_C'] = max(v for s in telemetry for v in s['cpu_temperatures_mC'].values()) / 1000
    metrics['throttle_delta_sum_including_smt_duplicates'] = sum(b['throttle_counts'][k] - v for k, v in a['throttle_counts'].items())
    numa = collections.Counter()
    for line in (folder / 'numa-start.txt').read_text().splitlines():
        if '.gguf' not in line:
            continue
        fields = dict(piece.split('=', 1) for piece in line.split() if '=' in piece)
        for node in ('N0', 'N1'):
            numa[node] += int(fields.get(node, 0)) * int(fields.get('kernelpagesize_kB', 4)) * 1024
    return {'name': folder.name, 'valid': True, 'excluded_supplemental_events': excluded,
            'original_capture_all_event_gate_passed': obs.get('valid_approximate_sample'),
            'phase': cap['arguments']['phase'],
            'depth': cap['arguments']['depth'], 'metrics': metrics, 'core_events': events,
            'channel_GB_s': rates, 'imc_intervals': dict(intervals), 'threads': threads,
            'model_resident_bytes_by_node': dict(numa), 'window_margin_seconds': cap['nominal_margin_seconds'],
            'caveat': 'Cycles are overlapping diagnostics, not exclusive wall-time fractions. DRAM-source retired-load uops are not DRAM bytes; prefetch traffic is not equivalent. No cold-load claim.'}


if __name__ == '__main__':
    results = []
    for path in sorted(ROOT.glob('*/observations.json')):
        result = summarize(path.parent)
        results.append(result)
        print(json.dumps({'name': result['name'], 'valid': result['valid'], 'metrics': result.get('metrics'), 'reason': result.get('reason')}, ensure_ascii=False))
    (ROOT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
