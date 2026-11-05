#!/usr/bin/env python3
"""Resume completed confirmations, then collect counters without a NUMA map walk."""
import fcntl
import json
import subprocess
import sys

from glm_q8_experiment import LAB, OUT, MODEL, BUILDS, environment
from summarize_bottleneck_observations import summarize


def main():
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run([sys.executable, str(LAB / "run_glm_q8_column_confirmation.py")],
                   check=True, env=environment())
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = OUT / "observation-B-light"
        command = [
            sys.executable, str(LAB / "observe_glm_counters_light.py"),
            "decode", "stalls", "--bench", str(BUILDS["B"] / "bin/llama-bench"),
            "--model", str(MODEL), "--threads", "24", "--tokens", "128",
            "--window", "18", "--output", str(target),
        ]
        subprocess.run(command, check=True, env=environment())
        result = summarize(target)
        result["placement_not_measured"] = True
        assert result["valid"]
        (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"light_counter_summary": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
