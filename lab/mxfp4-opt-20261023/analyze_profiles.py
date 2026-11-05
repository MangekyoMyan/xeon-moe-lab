#!/usr/bin/env python3
"""Summarize approximate steady decode diagnostics; never use them as A/B timing."""
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent.parent / 'results/mxfp4-opt-20261023'
result = {}
for label in ('baseline', 'optimized'):
    folder = OUT / ('model-profile-' + label + '-stat')
    capture = json.loads((folder / 'capture.json').read_text())
    assert capture['valid_sample']
    core = {}
    running = []
    for line in (folder / 'core.csv').read_text().splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split(',')
        core[parts[2]] = float(parts[0])
        running.append(float(parts[4]))
    total = {'read': 0, 'write': 0}
    duration = 0
    for line in (folder / 'imc.csv').read_text().splitlines():
        if not line or line.startswith('#'): continue
        parts = line.split(',')
        duration = max(duration, float(parts[0]))
        total['read' if parts[5] == 'umask=0x03/' else 'write'] += int(parts[2])
        running.append(float(parts[7]))
    observation = json.loads((folder / 'observations.json').read_text())
    telemetry = observation['telemetry']
    def fields(text): return {a: int(b) for a, b in (line.split() for line in text.splitlines())}
    a, b = fields(telemetry[0]['vmstat']), fields(telemetry[-1]['vmstat'])
    before_io, after_io = fields(telemetry[0]['io'].replace(':', '')), fields(telemetry[-1]['io'].replace(':', ''))
    symbols = []
    text = (OUT / ('model-profile-' + label + '-record') / 'symbols.txt').read_text()
    for line in text.splitlines():
        match = re.match(r'^\s*(\d+\.\d+)%\s+(\S+)\s+\[\.\]\s+(.+)', line)
        if match:
            symbols.append({'percent': float(match[1]), 'object': match[2], 'symbol': re.split(r'\s{2,}', match[3].strip())[0]})
    result[label] = {'read_GBs': total['read'] * 64 / duration / 1e9, 'write_GBs': total['write'] * 64 / duration / 1e9, 'IMC_window_s': duration, 'IPC': core['instructions:u'] / core['cycles:u'], 'user_cycles_per_task_clock_GHz': core['cycles:u'] / core['task-clock'] / 1e6, 'L2_pending_stall_cycle_fraction': core['cycle_activity.stalls_l2_pending:u'] / core['cycles:u'], 'minor_faults': core['minor-faults'], 'major_faults': core['major-faults'], 'disk_read_bytes_delta': after_io['read_bytes'] - before_io['read_bytes'], 'numa_hint_faults_delta': b['numa_hint_faults'] - a['numa_hint_faults'], 'numa_pages_migrated_delta': b['numa_pages_migrated'] - a['numa_pages_migrated'], 'min_counter_running_percent': min(running), 'symbols': symbols, 'observed_CPU_max_C': max(v for t in telemetry for v in t['cpu_temperatures_mC'].values()) / 1000}
result['limitations'] = ['Approximate windows inside real decode after model loading and warmup; marker receipt is not an acknowledged exact phase boundary.', 'IMC and NUMA counters include the host; other workloads were not deliberately run during these captures.', 'Stall cycles and sampled symbols overlap and are not additive wall-time fractions.', 'Frame-pointer callchains can be incomplete; use flat sampled symbols.', 'The optimized stat window had minor faults but no observed NUMA hint faults/migrations, major faults, or disk reads; their cause was not isolated.']
(OUT / 'PROFILE_SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
