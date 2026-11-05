#!/usr/bin/env python3
"""Install reviewed boot policy, preserve backups, and rebuild only this initrd."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / "results" / ROOT.name
KERNEL = os.uname().release
assert os.geteuid() == 0
assert KERNEL == "6.12.107+deb13-amd64"
assert subprocess.check_output(["modinfo", "-F", "version", "nvidia"], text=True).strip() == "580.196.07"
assert not Path("/sys/module/nvidia").exists(), "Keep this boot free of new GPU module loads"

BACKUP = OUT / "boot-backup"
BACKUP.mkdir(exist_ok=False)
record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
          "modified_files": [], "reboot_executed": False, "policy_started": False}


def fingerprint(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def install(destination, data, mode):
    path = Path(destination)
    item = {"path": str(path), "previously_existed": path.exists()}
    if path.exists():
        backup = BACKUP / path.relative_to("/")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        item["previous_sha256"] = fingerprint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    os.chown(path, 0, 0)
    item["installed_sha256"] = fingerprint(path)
    record["modified_files"].append(item)


initrd = Path("/boot") / ("initrd.img-" + KERNEL)
shutil.copy2(initrd, BACKUP / initrd.name)
record["initrd_before_sha256"] = fingerprint(initrd)
install("/etc/modprobe.d/llm-gpu-nouveau.conf",
        b"# PG500-216 uses the NVIDIA proprietary R580 driver.\n"
        b"blacklist nouveau\noptions nouveau modeset=0\n", 0o644)
install("/usr/local/sbin/llm-gpu-initial-policy", (ROOT / "initial_gpu_policy.py").read_bytes(), 0o755)
install("/etc/systemd/system/llm-gpu-initial-policy.service",
        (ROOT / "llm-gpu-initial-policy.service").read_bytes(), 0o644)

for command in [
    ["systemd-analyze", "verify", "/etc/systemd/system/llm-gpu-initial-policy.service"],
    ["systemctl", "daemon-reload"],
    ["systemctl", "enable", "llm-gpu-initial-policy.service"],
    ["update-initramfs", "-u", "-k", KERNEL],
]:
    subprocess.run(command, check=True)
record["initrd_after_sha256"] = fingerprint(initrd)
assert not Path("/sys/module/nvidia").exists()
assert record["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
(OUT / "boot-preparation.json").write_text(json.dumps(record, indent=2) + "\n")
owner = LAB.stat()
for path in [OUT, *OUT.rglob("*")]:
    if not path.is_symlink():
        os.chown(path, owner.st_uid, owner.st_gid)
print(json.dumps(record, indent=2))
