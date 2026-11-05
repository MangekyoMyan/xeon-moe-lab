#!/usr/bin/env python3
"""Read local BMC sensors, event history and disk diagnostics."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
TOOLS = ROOT / "diagnostics/usr"
ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = str(TOOLS / "lib/x86_64-linux-gnu")


def main():
    ipmi = str(TOOLS / "bin/ipmitool")
    smart = str(TOOLS / "sbin/smartctl")
    sensor = str(TOOLS / "bin/sensors")
    commands = [
        ("bmc-info", [ipmi, "-I", "open", "mc", "info"]),
        ("bmc-sensors", [ipmi, "-I", "open", "sensor", "list"]),
        ("bmc-sel-info", [ipmi, "-I", "open", "sel", "info"]),
        ("bmc-sel", [ipmi, "-I", "open", "sel", "elist", "last", "64"]),
        ("bmc-power", [ipmi, "-I", "open", "dcmi", "power", "reading"]),
        ("linux-sensors", [sensor, "-j"]),
        ("smart-before", [smart, "-x", "-j", "/dev/sda"]),
    ]
    records = []
    for label, command in commands:
        try:
            result = subprocess.run(command, env=ENV, text=True, capture_output=True, timeout=60)
            (OUT / (label + ".txt")).write_text(result.stdout)
            (OUT / (label + ".stderr")).write_text(result.stderr)
            records.append({"label": label, "command": command, "returncode": result.returncode})
            print(label, result.returncode, result.stdout[:700], result.stderr[:300], flush=True)
        except subprocess.TimeoutExpired:
            records.append({"label": label, "command": command, "timeout": True})
            print(label, "timeout", flush=True)
    (OUT / "diagnostics-run.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = ROOT.parent.stat()
            for path in OUT.glob("*"):
                if path.is_file():
                    os.chown(path, owner.st_uid, owner.st_gid)
