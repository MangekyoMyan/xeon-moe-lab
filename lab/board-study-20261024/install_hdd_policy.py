#!/usr/bin/env python3
"""Persist APM 254 only for the verified model HDD, with a root-owned utility."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
TARGET = Path("/usr/local/libexec/llm-lab/smartctl")
RULE = Path("/etc/udev/rules.d/99-local-llm-hdd-apm.rules")
COPYRIGHT = Path("/usr/local/share/doc/llm-lab-smartctl/copyright")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.geteuid() == 0
    assert not TARGET.exists() and not RULE.exists() and not COPYRIGHT.exists()
    props = dict(line.split("=", 1) for line in subprocess.check_output(
        ["udevadm", "info", "--query=property", "--name=/dev/sda"], text=True).splitlines() if "=" in line)
    serial = props["ID_SERIAL"]
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", serial)
    assert props["ID_MODEL"] == "WDC_WD10SPZX-75Z10T0"
    package = next((ROOT / "packages").glob("smartmontools_7.4-3_*.deb"))
    expected = next(row["sha256"] for row in json.loads((ROOT / "diagnostic-tools.json").read_text()) if row["package"] == "smartmontools")
    assert sha(package) == expected
    with tempfile.TemporaryDirectory(prefix="llm-smartctl-") as temp:
        subprocess.run(["dpkg-deb", "-x", str(package), temp], check=True)
        for destination, source in ((TARGET, Path(temp) / "usr/sbin/smartctl"),
                                    (COPYRIGHT, Path(temp) / "usr/share/doc/smartmontools/copyright")):
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            shutil.copyfile(source, destination)
            os.chown(destination, 0, 0)
            os.chmod(destination, 0o755 if destination == TARGET else 0o644)
    text = ("# Dedicated LLM model HDD: avoid frequent head parking. Original APM was 128.\n"
            f'ACTION=="add", SUBSYSTEM=="block", ENV{{DEVTYPE}}=="disk", ENV{{ID_SERIAL}}=="{serial}", '
            f'RUN+="{TARGET} -s apm,254 /dev/%k"\n')
    RULE.write_text(text)
    os.chown(RULE, 0, 0)
    os.chmod(RULE, 0o644)
    verify = subprocess.run(["udevadm", "verify", str(RULE)], capture_output=True, text=True, check=True)
    subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LD_")}
    current = subprocess.run([str(TARGET), "-g", "apm", "-j", "/dev/sda"], env=env, capture_output=True, text=True, check=True)
    assert json.loads(current.stdout)["ata_apm"]["level"] == 254
    record = {"target_disk_serial": serial, "original_apm": 128, "selected_apm": 254,
              "rule": text, "udev_verify": verify.stdout + verify.stderr,
              "installed": {str(p): sha(p) for p in (TARGET, RULE, COPYRIGHT)},
              "boot_application_tested": False,
              "rollback": [f"rm {RULE}", "udevadm control --reload-rules", f"{TARGET} -s apm,128 /dev/sda", f"rm {TARGET} {COPYRIGHT}"]}
    f = OUT / "hdd-persistent-policy.json"
    f.write_text(json.dumps(record, indent=2) + "\n")
    owner = ROOT.parent.stat()
    os.chown(f, owner.st_uid, owner.st_gid)
    print("Root-owned disk utility and disk-specific udev rule installed and verified")


if __name__ == "__main__":
    main()
