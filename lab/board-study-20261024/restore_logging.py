#!/usr/bin/env python3
"""Archive the full SEL, reclaim its capacity, and align its clock with the synced OS."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
ENV = os.environ.copy()
ENV.update({"TZ": "UTC", "LC_ALL": "C", "LD_LIBRARY_PATH": str(ROOT / "diagnostics/usr/lib/x86_64-linux-gnu")})
IPMI = str(ROOT / "diagnostics/usr/bin/ipmitool")


def call(*args):
    return subprocess.run([IPMI, "-I", "open", *args], env=ENV, capture_output=True, text=True, timeout=45, check=True)


def main():
    verified = json.loads((OUT / "sel-backup-verified.json").read_text())
    before = (OUT / "sel-before.bin").read_bytes()
    assert verified["records"] == 512 and verified["ids_match_text"]
    assert hashlib.sha256(before).hexdigest() == verified["sha256"]
    call("sel", "writeraw", str(OUT / "sel-before-recheck.bin"))
    assert (OUT / "sel-before-recheck.bin").read_bytes() == before
    archive = ROOT.parent / "board-sel-backup-20261024.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("sel-before.bin", "sel-full.txt", "sel-backup-verified.json", "bmc-sel-info.txt", "clock-check.json", "bmc-info.txt"):
            z.write(OUT / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.read("sel-before.bin") == before
    records = [{"backup": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]
    result = call("sel", "clear")
    records.append({"command": "sel clear", "stdout": result.stdout, "stderr": result.stderr})
    print("Archived and cleared SEL", flush=True)
    for attempt in range(10):
        result = call("sel", "info")
        (OUT / "sel-after-clear.txt").write_text(result.stdout)
        entries = int(next(line.split(":", 1)[1] for line in result.stdout.splitlines() if line.startswith("Entries")))
        if entries < 512:
            break
        time.sleep(1)
    else:
        raise RuntimeError("SEL clear has not completed")
    synchronized = subprocess.check_output(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], text=True).strip()
    if synchronized != "yes":
        raise RuntimeError("OS clock is not synchronized; BMC clock was not changed")
    stamp = time.strftime("%m/%d/%Y %H:%M:%S", time.gmtime())
    result = call("sel", "time", "set", stamp)
    records.append({"command": ["sel", "time", "set", stamp], "timezone": "UTC", "stdout": result.stdout, "stderr": result.stderr})
    result = call("sel", "time", "get")
    bmc = datetime.datetime.strptime(result.stdout.strip().removesuffix(" UTC"), "%m/%d/%y %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    difference = bmc - time.time()
    records.append({"clock_after": result.stdout, "bmc_minus_host_seconds": difference})
    (OUT / "logging-maintenance.json").write_text(json.dumps(records, indent=2) + "\n")
    assert abs(difference) < 3
    print("BMC clock synchronized; difference seconds", difference, flush=True)
    for name, command in (("sel-after-maintenance", ("sel", "elist")), ("sel-info-after-maintenance", ("sel", "info"))):
        result = call(*command)
        (OUT / (name + ".txt")).write_text(result.stdout)


if __name__ == "__main__":
    try:
        main()
    finally:
        owner = ROOT.parent.stat()
        for path in [*OUT.glob("*"), ROOT.parent / "board-sel-backup-20261024.zip"]:
            if path.is_file():
                os.chown(path, owner.st_uid, owner.st_gid)
