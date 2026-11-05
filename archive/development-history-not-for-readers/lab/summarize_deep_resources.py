#!/usr/bin/env python3
import json
from pathlib import Path
import statistics

lab = Path(__file__).resolve().parent
reports = []
for path in sorted((lab / "results").glob("cpu-abba-20261019T*/comparison.json")):
    comparison = json.loads(path.read_text())
    if "candidate_gain_percent" not in comparison or "numa" not in comparison["experiment_note"].lower():
        continue
    variants = {"A": [], "B": []}
    for run in comparison["runs"]:
        result = json.loads(Path(run["summary"]).read_text())
        first_timed = next((p["elapsed_seconds"] for p in result["phases"]
                            if "generation run 1/" in p["message"]), None)
        samples = result["observations"]
        variants[run["variant"]].append({
            "first_timed_generation_seconds_approx": first_timed,
            "max_observed_rss_gib": max(s.get("rss_kib", 0) for s in samples) / 1048576,
            "max_observed_hwm_gib": max(s.get("hwm_kib", 0) for s in samples) / 1048576,
            "max_observed_read_bytes": max(s.get("disk_read_bytes", 0) for s in samples),
            "summary": run["summary"],
        })
    medians = {key: {metric: statistics.median(row[metric] for row in values)
                     for metric in ("first_timed_generation_seconds_approx", "max_observed_rss_gib", "max_observed_hwm_gib")}
               for key, values in variants.items()}
    reports.append({"comparison": str(path), "note": comparison["experiment_note"], "runs": variants,
                    "medians": medians, "candidate_minus_baseline": {
                        key: medians["B"][key] - medians["A"][key] for key in medians["A"]}})
(lab / "results/deep-review-resource-costs.json").write_text(json.dumps({
    "note": "First timed-generation markers include process setup, model loading and warmup; timestamps are sampled approximately once per second. They are not pure model-load durations. RSS includes file-backed resident pages.",
    "experiments": reports}, indent=2) + "\n")
for r in reports:
    print(r["note"], r["candidate_minus_baseline"])
