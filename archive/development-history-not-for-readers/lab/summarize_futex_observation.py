#!/usr/bin/env python3
"""Pair futex entry/exit events; report summed syscall time, not wall-time gain."""
import collections
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent / 'results/bottleneck-observation-20261019'
obs = json.loads((root / 'prefill-full-stalls-futex-1/observations.json').read_text())
first, last = obs['telemetry'][0], obs['telemetry'][-1]
lo, hi = first['monotonic'], last['monotonic']
pending, pairs, unmatched_exits = {}, [], []
times = []
for line in (root / 'prefill-futex-events.txt').read_text().splitlines():
    match = re.match(r'^\s*(\d+)\s+(\d+\.\d+):\s+syscalls:sys_(enter|exit)_futex:\s+(.*)$', line)
    if not match:
        raise ValueError('Unparsed trace line: ' + line[:120])
    tid, timestamp, kind, details = match.groups()
    timestamp = float(timestamp); times.append(timestamp)
    if kind == 'enter':
        if tid in pending:
            raise ValueError('Unpaired repeated entry')
        op = int(re.search(r'\bop: (0x[0-9a-f]+)', details).group(1), 16)
        pending[tid] = (timestamp, op)
    else:
        if tid not in pending:
            unmatched_exits.append({'tid': tid, 'time': timestamp}); continue
        started, op = pending.pop(tid)
        ret = int(details.strip(), 16)
        if ret >= 2**63:
            ret -= 2**64
        if timestamp < started:
            raise ValueError('Negative syscall duration')
        pairs.append({'tid': tid, 'op': op, 'command': op & 127, 'ret': ret,
                      'seconds': timestamp - started,
                      'clipped_seconds': max(0, min(timestamp, hi) - max(started, lo))})
if min(times) < lo - 5 or max(times) < hi - 5 or max(times) > hi + 5:
    raise ValueError('Trace/telemetry clocks do not align')
waits = [p for p in pairs if p['command'] in (0, 9)]
success = [p for p in waits if p['ret'] == 0]
def describe(items):
    durations = sorted(p['seconds'] * 1000 for p in items)
    return {'count': len(items), 'sum_syscall_seconds': sum(p['seconds'] for p in items),
            'sum_syscall_seconds_clipped_to_telemetry': sum(p['clipped_seconds'] for p in items),
            'median_ms': statistics.median(durations) if durations else None,
            'p95_ms': durations[int((len(durations)-1)*.95)] if durations else None,
            'return_values': dict(collections.Counter(str(p['ret']) for p in items))}
elapsed = hi - lo
running_seconds = sum((int(last['threads'][tid]['schedstat'].split()[0]) - int(t['schedstat'].split()[0])) / 1e9 for tid, t in first['threads'].items())
runqueue_seconds = sum((int(last['threads'][tid]['schedstat'].split()[1]) - int(t['schedstat'].split()[1])) / 1e9 for tid, t in first['threads'].items())
result = {'paired_calls': len(pairs), 'unmatched_exits': unmatched_exits,
          'unmatched_entries': pending, 'trace_first_last_seconds': [min(times), max(times)],
          'telemetry_first_last_seconds': [lo, hi], 'telemetry_seconds': elapsed,
          'thread_count': len(first['threads']), 'thread_running_seconds': running_seconds,
          'thread_offcpu_seconds': elapsed * len(first['threads']) - running_seconds,
          'thread_runqueue_seconds': runqueue_seconds,
          'all_wait_calls': describe(waits), 'successful_wait_calls': describe(success),
          'command_counts': dict(collections.Counter(str(p['command']) for p in pairs)),
          'caveat': 'Futex syscall duration includes kernel work and wakeup scheduling. Threads overlap; summed thread-seconds are not elapsed wall seconds or a removable speedup. Boundary pairs are clipped to the telemetry window. No optimization was applied.'}
(root / 'futex-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
