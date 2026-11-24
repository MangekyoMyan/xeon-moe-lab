"""Timestamp append positions of the two model log files over the existing window."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
START = json.loads((ROOT/"WINDOW_START.json").read_text())
MODEL = START["model"]


def identity():
    path = Path(f"/proc/{MODEL['pid']}/stat")
    try:
        stat = path.read_text()
        return (int(stat[stat.rfind(")")+2:].split()[19]) == MODEL["start_ticks"]
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == START["boot"])
    except (OSError, ValueError, IndexError):
        return False


def sample():
    logs = {}
    for name, ref in START["logs"].items():
        try:
            path = os.stat(ref["path"])
            fd = os.stat(f"/proc/{MODEL['pid']}/fd/{ref['fd']}")
            logs[name] = {"dev":path.st_dev,"ino":path.st_ino,"size":path.st_size,
                          "mtime_ns":path.st_mtime_ns,"fd_matches":(fd.st_dev,fd.st_ino)==(path.st_dev,path.st_ino)}
        except OSError as exc:
            logs[name] = {"unavailable_errno":exc.errno}
    return {"monotonic":time.monotonic(),"wall":time.time(),"identity_matches":identity(),"logs":logs}


if __name__=="__main__":
    until=START["monotonic"]+600
    with (ROOT/"LOG_OFFSETS.jsonl").open("x",buffering=1) as out:
        while True:
            row=sample()
            out.write(json.dumps(row,separators=(",",":"))+"\n")
            remaining=until-time.monotonic()
            if remaining<=0 or not row["identity_matches"]:
                break
            time.sleep(min(10,remaining))
