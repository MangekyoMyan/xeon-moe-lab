#!/usr/bin/env python3
"""Smoke-test the final launcher against the validated deterministic response."""
import fcntl
import hashlib
import json
import subprocess
import time

from glm_q8_experiment import LAB, OUT, environment
from run_glm_q8_observations import QUESTION


def main():
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert json.loads((OUT / "column-confirmation.json").read_text())["complete"]
        assert json.loads((OUT / "observation-B-light/summary.json").read_text())["valid"]
        launcher = LAB / "run_glm_air_q8_optimized.sh"
        command = ["bash", str(launcher), "-p", QUESTION, "-n", "192"]
        started = time.time()
        result = subprocess.run(command, env=environment(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        (OUT / "launcher.stdout.txt").write_bytes(result.stdout)
        (OUT / "launcher.stderr.txt").write_bytes(result.stderr)
        reference = (OUT / "columns-response-0-B/stdout.txt").read_bytes()
        report = {
            "command": command, "exit_code": result.returncode,
            "elapsed_seconds": time.time() - started,
            "response_identical_to_baseline": result.stdout == reference,
            "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
            "response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        }
        (OUT / "launcher-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        assert result.returncode == 0 and result.stdout == reference, report
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
