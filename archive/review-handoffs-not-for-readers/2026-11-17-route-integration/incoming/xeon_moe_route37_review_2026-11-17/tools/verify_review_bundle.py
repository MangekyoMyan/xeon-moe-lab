#!/usr/bin/env python3
"""Check an unpacked review bundle. Never execute included code or access network."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('bundle', type=Path)
args = p.parse_args()
root = args.bundle.resolve()
manifest = json.loads((root / 'FILES_SHA256.json').read_text())
assert manifest['schema'] == 1
files = manifest['files']
actual = {x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file()}
assert actual == set(files) | {'FILES_SHA256.json'}, 'file set mismatch'
for name, expected in files.items():
    path = root / name
    assert not path.is_symlink() and path.resolve().is_relative_to(root), name
    data = path.read_bytes()
    assert len(data) == expected['bytes'], 'size mismatch: ' + name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], 'hash mismatch: ' + name
    assert not data.startswith(b'\x7fELF'), 'unexpected ELF: ' + name
print(json.dumps({'passed': True, 'manifest_entries': len(files), 'total_files': len(actual),
                  'pinned_commit': manifest['pinned_commit']}, ensure_ascii=False))
