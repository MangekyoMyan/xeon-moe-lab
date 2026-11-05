#!/usr/bin/env python3
"""Sequential Q6 screenings; isolated-kernel timings are not model speedups."""
import json
from pathlib import Path
import statistics
import subprocess

LAB = Path(__file__).resolve().parent
ROOT = LAB / "q6-xeon-20261020"
OUT = LAB / "results/optimization-q6-xeon-20261020"

def main():
    OUT.mkdir(exist_ok=False)
    rows = []
    for k, m, threads in ((2816, 1408, 1), (2816, 65536, 1),
                          (2816, 11264, 12), (2816, 11264, 24),
                          (2816, 262144, 12), (2816, 262144, 24)):
        label = f"k{k}-m{m}-t{threads}"
        command = ["numactl", "--physcpubind=0-23", "--interleave=0,1",
                   str(ROOT / "probe"), str(ROOT), str(k), str(m), str(threads), "20"]
        print(json.dumps({"start": label}), flush=True)
        with (OUT / (label + ".json")).open("w") as out, (OUT / (label + ".stderr")).open("w") as err:
            subprocess.run(command, stdout=out, stderr=err, check=True, timeout=600)
        data = json.loads((OUT / (label + ".json")).read_text())
        names = sorted({r["variant"] for r in data["runs"]})
        times = {n: statistics.median(r["seconds"] for r in data["runs"] if r["variant"] == n)
                 for n in names}
        row = {"name": label, "command": command, "median_seconds": times,
               "gain_percent": {n: 100*(times["original"]/t-1) for n,t in times.items()},
               "guarded_exact_cases": data["guarded_exact_cases"],
               "finite_half_exact_cases": data["finite_half_exact_cases"]}
        rows.append(row)
        (OUT / "micro-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)

if __name__ == "__main__":
    main()
