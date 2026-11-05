#!/usr/bin/env python3
"""Repeat existing streaming-read diagnostic; no model/configuration changes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

lab = Path(__file__).resolve().parent
out = lab / 'results/bottleneck-observation-20261019'
env = dict(os.environ, OMP_PROC_BIND='true', OMP_PLACES='threads', OMP_DYNAMIC='FALSE', OMP_THREAD_LIMIT='12')
result = {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (lab / 'read-bandwidth', lab / 'read_bandwidth.c')},
          'environment': {k: v for k, v in env.items() if k.startswith('OMP_')}, 'runs': [],
          'caveat': '2 GiB sequential AVX read reduction, 9 passes after first touch/warmup. This is a different workload, not an LLM roofline or a speedup prediction.'}
for placement in ('local', 'interleave', 'interleave', 'local', 'local', 'interleave'):
    command = ['numactl', '--physcpubind=0-11', '--localalloc' if placement == 'local' else '--interleave=0,1', str(lab / 'read-bandwidth'), '12']
    p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=True)
    j = json.loads(p.stdout)
    if sorted(j['cpus']) != list(range(12)) or not j['checksum_passed']:
        raise RuntimeError('Affinity/checksum mismatch')
    result['runs'].append({'placement': placement, 'command': command, 'stderr': p.stderr,
                           'result': j, 'median_GB_s': statistics.median(j['gb_s'])})
result['median_of_run_medians_GB_s'] = {p: statistics.median(r['median_GB_s'] for r in result['runs'] if r['placement'] == p) for p in ('local', 'interleave')}
(out / 'streaming-reference.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['median_of_run_medians_GB_s']))
