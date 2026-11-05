#!/usr/bin/env python3
"""Record host inventory and verify that unrelated runtime files/settings are intact."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess

from glm_q8_experiment import LAB, OUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher = json.loads((OUT / "launcher-verification.json").read_text())
        assert launcher["exit_code"] == 0 and launcher["response_identical_to_baseline"]
        before = json.loads((OUT / "before.json").read_text())
        settings = {path: Path(path).read_text().strip() for path in before["settings"]}
        hashes = {path: sha(path) for path in before["sha256"]}
        root = LAB / "review-glm-q8-columns4-20261020"
        prepared = json.loads((root / "prepare.json").read_text())
        source_hashes = {}
        for rel, expected in prepared["hashes"].items():
            actual = {"before": sha(LAB / "llama-private-prefetch" / rel),
                      "after": sha(root / "source" / rel)}
            assert actual == expected, rel
            source_hashes[rel] = actual
        active = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                comm = (path / "comm").read_text().strip()
                if comm.startswith(("llama-bench", "llama-complet", "glm-q8")) or comm == "perf":
                    active.append({"pid": int(path.name), "comm": comm})
            except (OSError, ProcessLookupError):
                pass
        inventory = {}
        for name, command in {
            "cpu": ["lscpu"],
            "numa": ["numactl", "--hardware"],
            "board": ["dmidecode", "-t", "2"],
            "dimms": ["dmidecode", "-t", "17"],
        }.items():
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            (OUT / f"host-{name}.txt").write_text(result.stdout)
            inventory[name] = {"command": command, "exit_code": result.returncode}
        (OUT / "host-memory.txt").write_text(Path("/proc/meminfo").read_text())
        disk = os.statvfs(LAB)
        report = {
            "normal_runtime_unchanged": hashes == before["sha256"],
            "system_settings_unchanged": settings == before["settings"],
            "settings": settings, "sha256": hashes, "source_hashes": source_hashes,
            "candidate_cpu_sha256": sha(root / "build/bin/libggml-cpu.so"),
            "launcher_sha256": sha(LAB / "run_glm_air_q8_optimized.sh"),
            "active_llama_or_perf_processes": active,
            "disk_available_GiB": disk.f_bavail * disk.f_frsize / 2**30,
            "inventory": inventory,
        }
        (OUT / "after-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["normal_runtime_unchanged"] and report["system_settings_unchanged"]
        assert not active, active
        account = pwd.getpwnam("codex")
        for path in [OUT, *OUT.rglob("*")]:
            if not path.is_symlink():
                os.chown(path, account.pw_uid, account.pw_gid)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
