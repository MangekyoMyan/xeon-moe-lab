#!/usr/bin/env python3
"""Stage a versioned private source and build script; do not activate it."""
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
candidate = lab / 'review-q8-prefetch8-20261019'
source = lab / 'llama-private-prefetch'
build = lab / 'build-avx-prefetch'
manifest = json.loads((candidate / 'prepare.json').read_text())
for relative, digest in manifest['files'].items():
    for path, key in [(lab / 'llama-private' / relative, 'before'), (candidate / 'source' / relative, 'after')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest[key]:
            raise SystemExit('Source changed: ' + str(path))
if source.exists() or build.exists(): raise SystemExit('Preserve existing source/build')
shutil.copytree(candidate / 'source', source)
staged = {}
for original, destination in [('run_deepseek_flash.sh', 'run_deepseek_flash_prefetch_staged.sh'),
                              ('rebuild_deepseek.sh', 'rebuild_deepseek_prefetch_staged.sh')]:
    path = lab / destination
    if path.exists(): raise SystemExit('Preserve existing staged script')
    text = (lab / original).read_text().replace('build-avx-deep', 'build-avx-prefetch')
    if original.startswith('rebuild_'):
        text = text.replace('/llama-private ', '/llama-private-prefetch ')
    else:
        text = text.replace('paired MoE and Q8 rows4.', 'paired MoE, Q8 rows4 and bounded Q8 prefetch.')
    path.write_text(text)
    path.chmod(0o755)
    staged[original] = {'staged': str(path), 'original_sha256': hashlib.sha256((lab / original).read_bytes()).hexdigest(),
                        'staged_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(out / 'staging.json').write_text(json.dumps({'source': str(source), 'build': str(build),
    'files': manifest['files'], 'scripts': staged, 'activated': False}, indent=2) + '\n')
print(json.dumps({'staged_source': str(source), 'active_launcher_changed': False}))
