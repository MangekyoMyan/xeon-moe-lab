#!/usr/bin/env python3
"""Serial row-sharing kernel measurements, separate from full model tests."""
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
from glm_q8_experiment import LAB, OUT, environment


def main():
    target = OUT / "row-micro"
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target.mkdir(exist_ok=False)
        report = {"note": "Synthetic matrices, not end-to-end model speed.",
                  "variants": ["single_pf8", "rows2_pf0", "rows4_pf0", "rows2_pf8", "rows4_pf8"],
                  "runs": []}
        for threads in (1, 12, 24):
            for k, m in ((4096, 1408), (4096, 11264), (1408, 32768), (12288, 4096), (4096, 151552)):
                name = f"k{k}-m{m}-t{threads}"
                command = ["numactl", "--physcpubind=0-23", "--interleave=0,1",
                           str(LAB / "glm-q8-micro-20261020/probe"),
                           str(LAB / "glm-q8-micro-20261020/kernels.so"),
                           str(k), str(m), str(threads), "16", "3"]
                with (target / (name + ".json")).open("w") as stdout, (target / (name + ".stderr")).open("w") as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, env=environment(), check=True, timeout=300)
                data = json.loads((target / (name + ".json")).read_text())
                medians = {v: statistics.median(x["seconds"] for x in data["runs"] if x["variant"] == v)
                           for v in range(5)}
                row = {"name": name, "command": command, "median_seconds": medians,
                       "gain_percent": {v: (medians[0] / s - 1) * 100 for v, s in medians.items()},
                       "guarded_exact_outputs": data["guarded_exact_outputs"]}
                report["runs"].append(row)
                (target / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
