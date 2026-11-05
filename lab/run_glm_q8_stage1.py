#!/usr/bin/env python3
"""Measure thread count and the existing prefetch change before new kernels."""
import fcntl
import json
import statistics
from glm_q8_experiment import OUT, run


def main():
    OUT.mkdir(exist_ok=True)
    report = {"complete": False, "thread_runs": [], "prefetch_runs": []}
    with (OUT / ".experiment.lock").open("a") as lock:
        print("Waiting for the initial load/smoke run to finish.", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        smoke = json.loads((OUT / "smoke/run.json").read_text())
        assert smoke.get("exit_code") == 0 and smoke.get("benchmarks")
        for i, threads in enumerate((24, 12, 12, 24)):
            r = run(f"threads-{i}-t{threads}", threads=str(threads), tokens=64, repetitions=2)
            report["thread_runs"].append({
                "label": r["label"], "threads": threads, "avg_ts": r["benchmarks"][0]["avg_ts"]})
        medians = {t: statistics.median(x["avg_ts"] for x in report["thread_runs"] if x["threads"] == t)
                   for t in (12, 24)}
        selected = max(medians, key=medians.get)
        report.update(thread_medians=medians, selected_threads=selected)
        (OUT / "stage1.json").write_text(json.dumps(report, indent=2) + "\n")
        for i, build in enumerate("ABBA"):
            r = run(f"prefetch-{i}-{build}", build=build, threads=str(selected), tokens=128, repetitions=2)
            report["prefetch_runs"].append({
                "label": r["label"], "build": build, "avg_ts": r["benchmarks"][0]["avg_ts"]})
            (OUT / "stage1.json").write_text(json.dumps(report, indent=2) + "\n")
        report["prefetch_medians"] = {
            b: statistics.median(x["avg_ts"] for x in report["prefetch_runs"] if x["build"] == b)
            for b in ("A", "B")}
        report["prefetch_gain_percent"] = (
            report["prefetch_medians"]["B"] / report["prefetch_medians"]["A"] - 1) * 100
        report["complete"] = True
        (OUT / "stage1.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
