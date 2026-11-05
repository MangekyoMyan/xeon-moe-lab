#!/usr/bin/env python3
"""Read-only repeatable platform health snapshot, with OS timestamps."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from hardware_run import ROOT, LAB, ENV, ipmi, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"os_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "platform": {}, "linux": sample()}
    for name in ("board_vendor", "board_name", "board_version", "bios_version", "bios_date"):
        report["platform"][name] = (Path("/sys/class/dmi/id") / name).read_text().strip()
    report["bmc_info"] = ipmi("mc", "info")
    report["bmc_sensors"] = ipmi("sensor", "list")
    report["sel_info"] = ipmi("sel", "info")
    report["sel_events"] = ipmi("sel", "elist")
    raw = bytes.fromhex(ipmi("raw", "0x0a", "0x48").strip())
    assert len(raw) == 4
    report["bmc_minus_os_seconds"] = int.from_bytes(raw, "little") - time.time()
    for name, options in (("smart", ["-a"]), ("disk_settings", ["-g", "all"])):
        result = subprocess.run([str(ROOT / "diagnostics/usr/sbin/smartctl"), *options, "-j", "/dev/sda"],
            env=ENV, capture_output=True, text=True, timeout=30)
        report[name] = json.loads(result.stdout)
        report[name + "_returncode"] = result.returncode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if os.geteuid() == 0:
        owner = LAB.stat()
        os.chown(args.output, owner.st_uid, owner.st_gid)
    print("Saved", args.output)
    print("BMC clock offset seconds:", round(report["bmc_minus_os_seconds"], 1))
    print("Current SEL memory events:", "ECC" in report["sel_events"])
    print("SMART overall health:", report["smart"].get("smart_status"))


if __name__ == "__main__":
    main()
