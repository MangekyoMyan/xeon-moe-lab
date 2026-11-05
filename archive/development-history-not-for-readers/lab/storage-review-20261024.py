#!/usr/bin/env python3
"""Inventory allocated space using metadata only, without following filesystem boundaries."""
import datetime
import json
import os
from pathlib import Path
import stat

LAB = Path('/home/codex/projects/llm-lab')
OUT = LAB / 'results/storage-review-20261024'


def main():
    OUT.mkdir(exist_ok=True)
    device = LAB.stat().st_dev
    large = []
    models = []
    errors = []
    scanned = 0
    pending = [Path('/')]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if path in (Path('/proc'), Path('/sys'), Path('/dev'), Path('/run')):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        errors.append({'path': str(path), 'error': str(exc)})
                        continue
                    if info.st_dev != device:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        scanned += 1
                        is_model = path.suffix.lower() in ('.gguf', '.safetensors', '.pt', '.pth', '.onnx', '.ckpt') and info.st_size >= 64 * 1024**2
                        if is_model or info.st_size >= 512 * 1024**2:
                            row = {'path': str(path), 'logical_bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
                                   'inode': info.st_ino, 'device': info.st_dev, 'hardlinks': info.st_nlink}
                            if info.st_size >= 512 * 1024**2:
                                large.append(row)
                            if is_model:
                                models.append(row)
        except OSError as exc:
            errors.append({'path': str(directory), 'error': str(exc)})
    fs = os.statvfs(LAB)
    report = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'regular_files_scanned': scanned, 'scope': 'Same filesystem as llm-lab; no symlinks followed; metadata only',
              'filesystem': {'capacity_bytes': fs.f_blocks * fs.f_frsize,
                             'used_bytes': (fs.f_blocks - fs.f_bfree) * fs.f_frsize,
                             'available_bytes': fs.f_bavail * fs.f_frsize,
                             'free_not_available_to_unprivileged_bytes': (fs.f_bfree - fs.f_bavail) * fs.f_frsize},
              'model_candidates': sorted(models, key=lambda x: x['path']),
              'large_files': sorted(large, key=lambda x: -x['allocated_bytes']), 'scan_errors': errors}
    output = OUT / 'filesystem-inventory-before.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    owner = LAB.stat()
    if os.geteuid() == 0:
        os.chown(OUT, owner.st_uid, owner.st_gid)
        os.chown(output, owner.st_uid, owner.st_gid)
    print(json.dumps({'scanned': scanned, 'models': len(models), 'large_files': len(large), 'errors': len(errors),
                      'model_candidates_outside_main_directory': [r for r in models if not r['path'].startswith(str(LAB / 'models') + '/')],
                      'large_files_outside_lab': [r for r in large if not r['path'].startswith(str(LAB) + '/')]}, indent=2))


if __name__ == '__main__':
    main()
