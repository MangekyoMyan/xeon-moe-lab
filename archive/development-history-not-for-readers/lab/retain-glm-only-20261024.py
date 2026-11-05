#!/usr/bin/env python3
"""Remove the user-selected models and their cache links, preserving GLM and all experiment records."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

LAB = Path('/home/codex/projects/llm-lab')
OUT = LAB / 'results/storage-review-20261024'
MODEL_NAMES = (
    'ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF',
    'ggml-org--gemma-3-1b-it-GGUF',
    'ggml-org--gpt-oss-20b-GGUF',
    'mradermacher--Melody1437-26B-A4B-i1-GGUF',
    'teamblobfish--DeepSeek-V4-Flash-GGUF',
)
CACHE_NAMES = (
    'models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF',
    'models--ggml-org--gemma-3-1b-it-GGUF',
    'models--mradermacher--Melody1437-26B-A4B-i1-GGUF',
    'models--teamblobfish--DeepSeek-V4-Flash-GGUF',
)


def filesystem():
    s = os.statvfs(LAB)
    return {'capacity_bytes': s.f_blocks * s.f_frsize,
            'used_bytes': (s.f_blocks - s.f_bfree) * s.f_frsize,
            'available_bytes': s.f_bavail * s.f_frsize}


def identity(path):
    s = path.stat()
    with path.open('rb') as f:
        prefix = f.read(1024 * 1024)
    assert prefix[:4] == b'GGUF'
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size,
            'blocks': s.st_blocks, 'links': s.st_nlink, 'mtime_ns': s.st_mtime_ns,
            'first_MiB_sha256': hashlib.sha256(prefix).hexdigest()}


def main():
    plan = json.loads((OUT / 'glm-only-deletion-plan.json').read_text())
    roots = [LAB / 'models' / name for name in MODEL_NAMES]
    roots += [Path('/root/.cache/huggingface/hub') / name for name in CACHE_NAMES]
    assert {str(p) for p in roots} == set(plan['model_roots'] + plan['cache_roots'])
    keep_dir = LAB / 'models/ggml-org--GLM-4.5-Air-GGUF'
    keep = {str(p): identity(p) for p in sorted(keep_dir.glob('*.gguf'))}
    assert len(keep) == 2 and sum(x['size'] for x in keep.values()) == 113612251680
    keep_keys = {(r['device'], r['inode']) for r in keep.values()}
    all_entries = []
    inodes = {}
    cache_revisions = {}
    for root in roots:
        assert root.is_dir() and not root.is_symlink()
        ref = root / 'refs/main'
        if ref.is_file():
            cache_revisions[str(root)] = ref.read_text().strip()
        for path in [root, *root.rglob('*')]:
            s = path.lstat()
            key = (s.st_dev, s.st_ino)
            assert key not in keep_keys
            row = {'path': str(path), 'size': s.st_size, 'allocated_bytes': s.st_blocks * 512,
                   'device': s.st_dev, 'inode': s.st_ino, 'mode': s.st_mode, 'links': s.st_nlink}
            if path.is_symlink():
                row['symlink_target'] = str(path.readlink())
            if stat.S_ISREG(s.st_mode):
                inodes.setdefault(key, []).append(row)
            all_entries.append(row)
    selected_keys = set()
    for row in plan['selected_model_files']:
        path = Path(row['path'])
        s = path.stat()
        assert s.st_size == row['logical_bytes'] and s.st_nlink == row['hardlinks']
        key = (s.st_dev, s.st_ino)
        selected_keys.add(key)
        assert len(inodes[key]) == s.st_nlink
        assert set(row['same_inode_paths']) == {r['path'] for r in inodes[key]}
    assert len(selected_keys) == 9
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        for fd in (proc / 'fd').glob('*'):
            try:
                s = fd.stat()
            except OSError:
                continue
            assert (s.st_dev, s.st_ino) not in selected_keys, f'Model still open by PID {proc.name}'
        try:
            maps = (proc / 'maps').read_text()
        except OSError:
            continue
        assert not any(str(root) + '/' in maps for root in roots), f'Model still mapped by PID {proc.name}'
    unique = {(r['device'], r['inode']): r['allocated_bytes'] for r in all_entries}
    report = {'phase': 'verified_before_deletion', 'started_unix': time.time(),
              'authorization': 'User requested retaining only GLM-4.5-Air and deleting the other models to prepare for a larger model.',
              'filesystem_before': filesystem(), 'kept_glm_before': keep,
              'roots_to_remove': [str(p) for p in roots], 'entries_to_remove': all_entries,
              'cache_revisions': cache_revisions, 'unique_allocated_bytes_to_remove': sum(unique.values()),
              'selected_model_files': 9}
    log = OUT / 'glm-only-deletion.json'
    log.write_text(json.dumps(report, indent=2) + '\n')
    for root in roots:
        shutil.rmtree(root)
        report.setdefault('removed_roots', []).append(str(root))
        log.write_text(json.dumps(report, indent=2) + '\n')
    assert all(not p.exists() for p in roots)
    assert {str(p): identity(p) for p in sorted(keep_dir.glob('*.gguf'))} == keep
    assert sorted(p.name for p in (LAB / 'models').iterdir()) == ['ggml-org--GLM-4.5-Air-GGUF']
    report.update({'phase': 'complete', 'finished_unix': time.time(),
                   'kept_glm_verified_unchanged': True, 'filesystem_after': filesystem(),
                   'remaining_model_files': [str(p) for p in sorted(keep_dir.glob('*.gguf'))]})
    report['available_bytes_gained'] = report['filesystem_after']['available_bytes'] - report['filesystem_before']['available_bytes']
    log.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('phase', 'unique_allocated_bytes_to_remove', 'available_bytes_gained',
                     'filesystem_after', 'kept_glm_verified_unchanged')}, indent=2))


if __name__ == '__main__':
    try:
        with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    finally:
        owner = LAB.stat()
        for path in OUT.glob('*.json'):
            os.chown(path, owner.st_uid, owner.st_gid)
