#!/usr/bin/env python3
"""Extract Debian diagnostic tools locally; do not install services or packages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    "ipmitool": "1.8.19-9", "lm-sensors": "1:3.6.2-2", "smartmontools": "7.4-3",
    "freeipmi-common": "1.6.15-1", "libfreeipmi17": "1.6.15-1",
    "stressapptest": "1.0.11-2+b1", "libaio1t64": "0.3.113-8+b1",
    "libgcrypt20": "1.11.0-7+deb13u1", "libgpg-error0": "1.51-4",
}


def main():
    cache = ROOT / "packages"
    target = ROOT / "diagnostics"
    cache.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    subprocess.run(["apt-get", "download", *[f"{name}={version}" for name, version in PACKAGES.items()]], cwd=cache, check=True)
    records = []
    for file in sorted(cache.glob("*.deb")):
        info = subprocess.check_output(["dpkg-deb", "-f", str(file), "Package", "Version"], text=True)
        fields = dict(line.split(": ", 1) for line in info.strip().splitlines())
        name, version = fields["Package"], fields["Version"]
        assert PACKAGES[name] == version
        metadata = subprocess.check_output(["apt-cache", "show", f"{name}={version}"], text=True)
        expected = next(line.split(": ", 1)[1] for line in metadata.splitlines() if line.startswith("SHA256: "))
        with file.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        assert digest == expected
        subprocess.run(["dpkg-deb", "-x", str(file), str(target)], check=True)
        records.append({"package": name, "version": version, "sha256": digest})
    assert len(records) == len(PACKAGES)
    (ROOT / "diagnostic-tools.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = ROOT.parent.stat()
            for file in [ROOT, *ROOT.rglob("*")]:
                os.chown(file, owner.st_uid, owner.st_gid, follow_symlinks=False)
