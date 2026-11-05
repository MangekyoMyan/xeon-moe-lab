#!/usr/bin/env python3
"""Read existing model shards once; run under numactl --interleave=0,1."""
import datetime
import json
import time
from pathlib import Path

lab = Path(__file__).resolve().parent
files = sorted((lab / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL').glob('*.gguf'))
assert len(files) == 2, files
tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
report_path = lab / 'results' / f'model-prewarm-{tag}.json'
report = {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'files': [str(p) for p in files], 'total_bytes': sum(p.stat().st_size for p in files),
          'read_bytes': 0, 'completed': False, 'observations': []}
start = last = time.monotonic()
buffer = bytearray(4 * 1024 * 1024)
for path in files:
    with path.open('rb', buffering=0) as stream:
        while count := stream.readinto(buffer):
            report['read_bytes'] += count
            now = time.monotonic()
            if now - last >= 30:
                sample = {'elapsed_seconds': round(now - start, 2),
                          'read_gib': round(report['read_bytes'] / 2**30, 3)}
                report['observations'].append(sample)
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(sample), flush=True)
                last = now
report['completed'] = True
report['elapsed_seconds'] = time.monotonic() - start
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'report': str(report_path), 'completed': True,
                  'elapsed_seconds': report['elapsed_seconds']}), flush=True)
