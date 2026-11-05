#!/usr/bin/env python3
"""Fetch the NVIDIA-published redistribution archive and verify its SHA-256."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results' / ROOT.name
manifest = json.loads((ROOT / 'references/redistrib_580.196.07.json').read_text())
entry = manifest['nvidia_driver']['linux-x86_64']
url = 'https://developer.download.nvidia.com/compute/nvidia-driver/redist/' + entry['relative_path']
target = ROOT / Path(entry['relative_path']).name
temporary = target.with_suffix(target.suffix + '.partial')
digest = hashlib.sha256()
count = 0
started = last = time.monotonic()
with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
    while block := response.read(1 << 20):
        stream.write(block); digest.update(block); count += len(block)
        now = time.monotonic()
        if now - last >= 20:
            print('Downloaded', round(count / 1e6), 'MB', flush=True)
            last = now
    stream.flush(); os.fsync(stream.fileno())
assert count == int(entry['size'])
assert digest.hexdigest() == entry['sha256']
temporary.rename(target)
with tarfile.open(target, 'r:xz') as archive:
    names = archive.getnames()
    (OUT / 'driver-archive-files.json').write_text(json.dumps(names, indent=2) + '\n')
    archive.extractall(ROOT / 'driver-archive', filter='data')
record = {'url': url, 'bytes': count, 'sha256': digest.hexdigest(), 'verified': True,
          'elapsed_s': time.monotonic() - started, 'extracted': str(ROOT / 'driver-archive')}
(OUT / 'driver-archive-verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
