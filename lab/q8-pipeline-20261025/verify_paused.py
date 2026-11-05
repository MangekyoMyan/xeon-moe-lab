#!/usr/bin/env python3
"""Verify this lab's workloads are gone, the lock is free, and persist the checkpoint."""
import datetime
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name

def main():
    active = []
    names = {'llama-bench', 'llama-completion', 'kernel-probe', 'tile-probe', 'reuse-candidate-dense',
             'reuse-candidate-moe', 'candidate-logits-check', 'reuse-logits-check', 'test-backend-ops',
             'test-quantize-fns', 'cc1', 'cc1plus', 'cmake'}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = [x.decode(errors='replace') for x in (entry / 'cmdline').read_bytes().split(b'\0') if x]
            if args and Path(args[0]).name in names and any(str(LAB) in arg for arg in args):
                active.append({'pid': int(entry.name), 'name': Path(args[0]).name})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    assert not active, active
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    tiles = json.loads((OUT / 'tile-micro-summary.json').read_text()) if (OUT / 'tile-micro-summary.json').exists() else []
    state_path = OUT / 'WORK_STATE.json'
    state = json.loads(state_path.read_text())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state.update(status='paused_by_user', phase='Stopped for GPU installation and reboot', paused_utc=now,
                 original_finish_target_suspended=True, active_experiment_processes=active, experiment_lock_free=True,
                 completed_tile_micro_cases=[row['label'] for row in tiles], automatic_resume=False,
                 current_model_page_layout='4KiB interleave restored; reboot will discard page cache',
                 baseline_unchanged=True, no_adoption_yet=True)
    state['resume_order'] = [
        'Wait for user to report GPU installation/reboot complete; do not resume automatically.',
        'Inspect actual new boot, GPU PCI identity, driver availability, PCIe link, NUMA node, cooling and health.',
        'Use existing V100 preparation tools; initial GPU target remains ECC enabled and 150 W.',
        'Resume CPU experiments from saved tile micro cases; original sequence is not restart-idempotent, so skip completed build/validation/case directories and rerun only the interrupted case.',
        'New tiles build passed graph/tile edge tests after fixing dense multi-input range. Full-model tests of this newest build are still pending.',
        'Keep accepted runtime/launcher unchanged until combined optimizations pass model validation and acceptance measurements.'
    ]
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    report = {'paused_utc': now, 'active_experiment_processes': active, 'experiment_lock_free': True,
              'saved_tile_micro_cases': len(tiles), 'baseline_runtime_unchanged': True,
              'model_files_unchanged': True, 'automatic_resume': False}
    (OUT / 'PAUSED_FOR_GPU.json').write_text(json.dumps(report, indent=2) + '\n')
    owner = LAB.stat()
    for path in [OUT, *OUT.rglob('*')]:
        os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
    os.sync()
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
