#!/usr/bin/env python3
"""Stop this experiment coordinator and verify its child workloads have exited."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results' / ROOT.name
COORDINATORS = {str(ROOT / name) for name in ['run_initial_sequence.py', 'run_mid_sequence.py',
                                             'run_prefill_sequence.py', 'run_tiles_sequence.py']}

def processes():
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            argv = folder.joinpath('cmdline').read_bytes().split(b'\0')
            args = [arg.decode(errors='replace') for arg in argv if arg]
            if args and Path(args[0]).name.startswith('python') and len(args) > 1 and args[1] in COORDINATORS:
                found.append({'pid': int(folder.name), 'script': args[1]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found

def main():
    (OUT / 'STOP_REQUESTED').write_text('User requested pause for GPU installation and reboot. Do not resume automatically.\n')
    before = processes()
    for item in before:
        os.kill(item['pid'], signal.SIGINT)
    deadline = time.monotonic() + 20
    while processes() and time.monotonic() < deadline:
        time.sleep(0.25)
    after = processes()
    report = {'reason': 'User requested pause for GPU installation and reboot', 'before': before, 'remaining_coordinators': after}
    (OUT / 'pause-processes.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    assert not after, 'Coordinator did not exit promptly; inspect before reboot clearance'

if __name__ == '__main__':
    main()
