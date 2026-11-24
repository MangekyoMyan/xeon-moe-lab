"""Host-specific identity and maintenance helpers shared by these isolated tests."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
PRIOR = LAB / "normal24-timeline-20261124"
sys.path.insert(0, str(PRIOR))
import study
import prefill

# Route new guard windows/stop markers to the new study. Baseline model and driver
# pins belong to the accepted timeline control and stay at their original paths.
study.R = HERE
read, write, fresh, sha = study.read, study.write, study.fresh, study.sha
adopt = study.adopt


@contextlib.contextmanager
def window(label):
    assert os.geteuid() == 0 and __debug__
    with (PRIOR / ".study.lock").open("r") as shared_lock, (HERE / ".study.lock").open("a") as own_lock:
        for lock in (shared_lock, own_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {}
        def interrupt(sig, frame):
            raise KeyboardInterrupt(f"controller received {sig}")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old[sig] = signal.signal(sig, interrupt)
        try:
            with study.maintenance(label):
                yield
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
