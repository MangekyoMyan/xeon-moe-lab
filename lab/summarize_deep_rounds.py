#!/usr/bin/env python3
"""Summarize this review round without treating within-process samples as independent."""
import json
from pathlib import Path
import statistics

lab = Path(__file__).resolve().parent
rows = []
for path in sorted((lab / "results").glob("cpu-abba-20261019T*/comparison.json")):
    r = json.loads(path.read_text())
    if "candidate_gain_percent" not in r:
        continue
    variants = r["process_mean_tok_s"]
    rows.append({"path": str(path), "note": r["experiment_note"], "order": r["order"],
                 "baseline_process_means": variants["A"], "candidate_process_means": variants["B"],
                 "baseline_median": statistics.median(variants["A"]),
                 "candidate_median": statistics.median(variants["B"]),
                 "gain_percent": r["candidate_gain_percent"], "same_boot": r["same_boot"],
                 "warm_cache_observed": r["warm_cache_observed"]})
adoption_path = lab / "results/deep-review-source-adoption.json"
adopted = adoption_path.exists() and json.loads(adoption_path.read_text()).get("launcher_switched", False)
result = {"adopted": adopted, "active_default": "Combined MoE+Q8 with NUMA replicas disabled" if adopted else None,
          "comparison_units": "Fresh-process means; within-process repetitions are not independent processes", "rounds": rows}
(lab / "results/deep-review-round-summary.json").write_text(json.dumps(result, indent=2) + "\n")
for r in rows:
    print(f'{r["note"]}: {r["baseline_median"]:.6f} -> {r["candidate_median"]:.6f} tok/s ({r["gain_percent"]:+.3f}%), {r["order"]}')
