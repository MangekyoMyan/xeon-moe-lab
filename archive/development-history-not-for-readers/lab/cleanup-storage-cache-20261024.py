#!/usr/bin/env python3
"""Remove only an obsolete CUDA download and an abandoned extraction copy."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

LAB = Path('/home/codex/projects/llm-lab')
WORK = LAB / 'v100-prep-20261024'
OUT = LAB / 'results/storage-review-20261024'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def model_state():
    result = {}
    for path in (LAB / 'models').rglob('*.gguf'):
        s = path.stat()
        result[str(path)] = [s.st_dev, s.st_ino, s.st_size, s.st_blocks, s.st_nlink, s.st_mtime_ns]
    return result


def main():
    old_manifest = json.loads((WORK / 'redistrib_12.9.1.json').read_text())
    current_manifest = json.loads((WORK / 'redistrib_12.9.2.json').read_text())
    installed = json.loads((WORK / 'toolkit-installed.json').read_text())
    assert next(x for x in installed if x['component'] == 'libcublas')['version'] == '12.9.2.10'
    old = old_manifest['libcublas']['linux-x86_64']
    old_archive = WORK / 'downloads' / Path(old['relative_path']).name
    stage = WORK / 'downloads/cuda_nvcc-extracted'
    assert old_archive.is_file() and not old_archive.is_symlink()
    assert stage.is_dir() and not stage.is_symlink()
    checks = []
    for component in ('libcublas', 'cuda_nvcc'):
        item = current_manifest[component]['linux-x86_64']
        path = WORK / 'downloads' / Path(item['relative_path']).name
        assert path.stat().st_size == int(item['size']) and digest(path) == item['sha256']
        checks.append({'retained_archive': str(path), 'sha256': item['sha256']})
    assert old_archive.stat().st_size == int(old['size']) and digest(old_archive) == old['sha256']
    runtime = {name: digest(WORK / name) for name in (
        'build-cuda/bin/libggml-cuda.so', 'build-cuda/bin/llama-completion',
        'build-cpu/bin/llama-completion', 'cuda-12.9.2/bin/nvcc')}
    before_models = model_state()
    assert len(before_models) == 11
    removed = []
    for path in (old_archive, stage):
        size = int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True).split()[0])
        removed.append({'path': str(path), 'allocated_bytes': size})
    report = {'started_unix': time.time(), 'removed': removed, 'retained_archives_verified': checks,
              'obsolete_archive_sha256': old['sha256'], 'model_files_before': before_models,
              'runtime_hashes_before': runtime, 'phase': 'verified_before_removal'}
    (OUT / 'cache-cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
    old_archive.unlink()
    shutil.rmtree(stage)
    assert model_state() == before_models
    assert all(digest(WORK / name) == value for name, value in runtime.items())
    linked = subprocess.check_output(['ldd', str(WORK / 'build-cuda/bin/libggml-cuda.so')], text=True)
    assert 'not found' not in linked
    (OUT / 'cuda-linkage-after-cleanup.txt').write_text(linked)
    report.update({'phase': 'complete', 'model_files_unchanged': True, 'runtime_hashes_unchanged': True,
                   'removed_allocated_bytes': sum(x['allocated_bytes'] for x in removed),
                   'finished_unix': time.time()})
    (OUT / 'cache-cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('phase', 'removed_allocated_bytes', 'model_files_unchanged', 'runtime_hashes_unchanged')}, indent=2))


if __name__ == '__main__':
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
