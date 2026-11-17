#!/usr/bin/env python3
"""Independent, read-only recomputation of the pinned calibration raw files.

Only Python's standard library is used. No project code, model, or library runs.
Output goes outside the reviewed repository. Times are descriptive, not causal.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PIN = "fa6671f7b5281bad28947fe5324e2495e792a86f"
REL = Path("archive/review-handoffs-not-for-readers/2026-11-17-route-calibration")
KEYS = ["step_begin_ns", "decode_begin_ns", "decode_return_ns", "output_ready_ns",
        "validation_end_ns", "finalize_end_ns", "step_end_ns"]


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def stats(x):
    v = sorted(x)
    at = (len(v) - 1) * .95
    lo, hi = math.floor(at), math.ceil(at)
    return {"n": len(v), "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "p95_linear_ms": v[lo] + (v[hi] - v[lo]) * (at - lo),
            "min_ms": v[0], "max_ms": v[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    if repo == out or repo in out.parents:
        raise SystemExit("Output must be outside the reviewed repository")
    c = repo / REL
    hashes = {}
    report = {"pin": PIN, "scope": "independent raw read and arithmetic only",
              "float_files": [], "runs": [], "blocks": [], "timing_arms": {}}
    float_payloads = []
    for p in sorted(c.glob("runs/*/correct*.f32")):
        b = p.read_bytes()
        assert len(b) == 4 * 129280 * 4
        values = array.array("f")
        assert values.itemsize == 4
        values.frombytes(b)
        if sys.byteorder != "little":
            values.byteswap()
        assert all(math.isfinite(x) for x in values)
        sha = hashlib.sha256(b).hexdigest()
        hashes[str(p.relative_to(repo))] = sha
        report["float_files"].append({"file": str(p.relative_to(c)), "sha256": sha,
                                      "bytes": len(b), "floats": len(values), "finite": True})
        float_payloads.append(b)
    assert len(float_payloads) == 5
    assert all(b == float_payloads[0] for b in float_payloads)
    report["all_five_float_files_byte_equal"] = True
    grouped = {}
    t1_count = 0
    trace_records = 0
    screen_records = 0
    for p in sorted(c.glob("runs/*/RUN.json")):
        run = read_json(p)
        assert run["passed"] is True
        hashes[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
        report["runs"].append({"directory": p.parent.name, "arm": run["arm"],
                               "cparams": run["cparams"], "memory_after_warmup": run["memory_after_warmup"]})
        assert run["cparams"]["n_outputs_max"] == 1
        for q in sorted(p.parent.glob("*.json")):
            o = read_json(q)
            if "steps" not in o:
                continue
            hashes[str(q.relative_to(repo))] = hashlib.sha256(q.read_bytes()).hexdigest()
            steps = o["steps"]
            assert len(steps) == o["calls"]
            last = o["session_setup_end_ns"]
            core = []
            outer = []
            for s in steps:
                ts = [s[k] for k in KEYS]
                assert all(isinstance(x, int) for x in ts)
                assert ts == sorted(ts) and ts[0] >= last
                assert s["decode_status"] == 0 and s["synchronized"] and s["ok"]
                core.append((ts[3] - ts[1]) / 1e6)
                # Out-of-engine stamp interval, not all collector overhead.
                outer.append(((ts[1] - ts[0]) + (ts[6] - ts[3])) / 1000)
                last = ts[-1]
            assert o["block_end_ns"] >= last
            recs = o.get("trace", {}).get("records", [])
            nr = len(recs)
            trace_records += nr
            if o["name"].startswith("screen_"):
                screen_records += nr
            t1_count += len(steps)
            row = {"file": str(q.relative_to(c)), "mode": o["mode"],
                   "name": o["name"], "case_id": o["case_id"],
                   "core": stats(core), "first_ms": core[0],
                   "after_first_mean_ms": statistics.mean(core[1:]),
                   "outer_stamped_mean_us": statistics.mean(outer), "route_records": nr}
            report["blocks"].append(row)
            if o["name"].startswith("timing_"):
                grouped.setdefault(o["mode"], []).extend(core)
    assert t1_count == 1044 and trace_records == 19092 and screen_records == 14208
    report.update(t1_calls=t1_count, route_records=trace_records, screen_route_records=screen_records)
    report["timing_arms"] = {k: stats(v) for k, v in grouped.items()}
    means = {k: v["mean_ms"] for k, v in report["timing_arms"].items()}
    report["descriptive_ratios_percent"] = {
        "D0_over_F_minus_1": 100 * (means["D0"] / means["F"] - 1),
        "D1_over_D0_minus_1": 100 * (means["D1"] / means["D0"] - 1),
        "D2_over_D1_minus_1": 100 * (means["D2"] / means["D1"] - 1),
    }
    f_means = [b["core"]["mean_ms"] for b in report["blocks"]
               if b["mode"] == "F" and b["name"].startswith("timing_")]
    report["F_block_max_over_min_minus_1_percent"] = 100 * (max(f_means) / min(f_means) - 1)
    free = {m["free_bytes"] for r in report["runs"] for m in r["memory_after_warmup"]}
    assert free == {2225340416}
    one_projection = 5120 * 2304 * 384 * 17 // 32
    shapes = []
    for u in (1024, 512, 256):
        swa_rows = ((128 + u + 255) // 256) * 256
        shapes.append({"ubatch": u, "raw_SWA_rows": swa_rows,
                       "raw_SWA_payload_MiB": 40 * 512 * swa_rows * 2 / 2**20,
                       "HCA_mask_payload_MiB": 131072 * u * 2 / 2**20,
                       "HCA_indexer_score_payload_MiB": 131072 * u * 4 / 2**20,
                       "MXFP4_full384_single_projection_MiB": one_projection / 2**20,
                       "summation_to_arena_peak_not_valid": True})
    report["source_shape_arithmetic_not_measured_free"] = shapes
    report["guard_phase_analysis"] = {
        "observed_free_MiB": next(iter(free)) / 2**20,
        "observed_phase": "M2: after vision plus 64-token text warmup, no bank",
        "legacy_guard_phase": "After context sched_reserve, in hot initialize before vision; both pre and post are before vision",
        "legacy_pre_rule": "free_bytes > native_payload_bytes + 3072 * 2**20",
        "legacy_post_rule": "free_bytes > 2560 * 2**20 after private bank reachable-H warmup and sync, still before vision",
        "legacy_pre_guard_free_measured_here": None,
        "legacy_post_guard_free_measured_here": None,
        "actual_required_improvement_at_legacy_phase_MiB": None,
        "legacy_guard_capacity_failure_observed": False,
        "hypothetical_postwarmup_3072_screen": {
            "adopted_as_new_required_gate": False,
            "strict_deficit_before_payload_MiB": 3072 - next(iter(free))/2**20,
            "example_small_triplet_bytes": 12072960,
            "strict_deficit_with_example_native_triplet_MiB": 3072 - next(iter(free))/2**20 + 12072960/2**20,
            "scope": "Arithmetic only if one additionally asks to retain 3072 MiB at M2; not the existing guard at its actual phase",
        },
        "extra_allocation_costs": "unknown; metadata, alignment, allocator/pool and runtime demands not included",
    }
    for rel in [
        "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/exploration-20261112/hot-ffn/hot-backend.cpp",
        "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/exploration-20261112/hot-ffn/llama-context.cpp",
    ]:
        hashes[rel] = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
    report["source_raw_sha256"] = hashes
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "t1_calls": t1_count, "all_f32_equal": True,
                      "timing_means_ms": means, "guard": report["guard_phase_analysis"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
