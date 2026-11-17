#!/usr/bin/env python3
"""Read saved JSON only; do not import or execute repository code or binaries."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arch = Path("archive/review-handoffs-not-for-readers")
    hist = arch / "2026-11-16-current"
    snap = hist / "snapshot/llm-lab"
    old = snap / "exploration-20261112"
    new = arch / "2026-11-17-parallel-pilot"
    sources = {}

    def read(rel):
        rel = Path(rel)
        data = (repo / rel).read_bytes()
        sources[str(rel)] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        return json.loads(data)

    def local_old_path(absolute):
        return snap / Path(absolute).relative_to("/home/codex/projects/llm-lab")

    def check_summary(summary_rel):
        summary = read(summary_rel)
        result = {}
        all_runs = {}
        for arm, absolute_paths in summary["sources"].items():
            runs = [read(local_old_path(p)) for p in absolute_paths]
            all_runs[arm] = runs
            rows = [c for run in runs for c in run["cases"]]
            assert all(run["complete"] and run["error"] is None for run in runs)
            assert len({run["input_sha256"] for run in runs}) == 1
            steps = sum(c["timings"]["predicted_n"] - 1 for c in rows)
            ms = sum(c["timings"]["predicted_ms"] for c in rows)
            tps = steps * 1000 / ms
            assert abs(tps - summary["aggregate"][arm]["decode_tok_s"]) < 1e-11
            result[arm] = {"cases": len(rows), "decode_steps": steps, "total_ms": ms,
                           "aggregate_decode_tok_s": tps}
        names = [c["name"] for c in all_runs["control"][0]["cases"]]
        comparisons = []
        for name in names:
            cases = [next(c for c in run["cases"] if c["name"] == name)
                     for runs in all_runs.values() for run in runs]
            comparisons.append({"name": name,
                                "recorded_token_hashes_all_equal": len({c["token_sha256"] for c in cases}) == 1,
                                "recorded_text_hashes_all_equal": len({c["output_sha256"] for c in cases}) == 1})
        result["recorded_hash_comparisons"] = comparisons
        result["all_recorded_token_and_text_hashes_equal"] = all(
            c["recorded_token_hashes_all_equal"] and c["recorded_text_hashes_all_equal"]
            for c in comparisons)
        assert result["all_recorded_token_and_text_hashes_equal"] == summary["all_outputs_identical"]
        result["gain_percent"] = (result["candidate"]["aggregate_decode_tok_s"] /
                                   result["control"]["aggregate_decode_tok_s"] - 1) * 100
        return result, all_runs

    final, _ = check_summary(old / "FINAL_ABBA.json")
    miss, miss_runs = check_summary(old / "hot-ffn/MISS_ABBA.json")
    pilot = read(old / "hot-ffn/MODEL_PILOT_RESULTS.json")
    hot = read(old / "heldout/20261112-180359-hot14-r1/RESULT.json")
    cases = []
    for c in hot["cases"]:
        tm = c["timings"]
        assert tm["predicted_n"] == c["output_token_count"] == 128
        rate = (tm["predicted_n"] - 1) * 1000 / tm["predicted_ms"]
        assert abs(rate - tm["predicted_per_second"]) < 1e-11
        ctrl = [next(x for x in run["cases"] if x["name"] == c["name"])
                for run in miss_runs["control"]]
        ctrl_rate = sum(x["timings"]["predicted_n"] - 1 for x in ctrl) * 1000 / sum(
            x["timings"]["predicted_ms"] for x in ctrl)
        same_ids = all(x["token_sha256"] == c["token_sha256"] for x in ctrl)
        same_text = all(x["output_sha256"] == c["output_sha256"] for x in ctrl)
        pub = next(x for x in pilot["cases"] if x["name"] == c["name"])
        assert abs(ctrl_rate - pub["control_tok_s"]) < 1e-11
        assert same_ids == pub["same128tokens"]
        cases.append({"name": c["name"], "prompt_tokens": c["prompt_tokens"],
                      "generated_tokens": c["output_token_count"], "decode_steps": 127,
                      "predicted_ms": tm["predicted_ms"], "control_decode_tok_s": ctrl_rate,
                      "candidate_decode_tok_s": rate, "same_recorded_token_hash": same_ids,
                      "same_recorded_text_hash": same_text})
    hot_ms = sum(c["predicted_ms"] for c in cases)
    hot_rate = sum(c["decode_steps"] for c in cases) * 1000 / hot_ms
    assert abs(hot_rate - pilot["candidate_tok_s"]) < 1e-11
    capacity = read(old / "hot-ffn/CAPACITY.json")
    b14 = next(x for x in capacity["budgets"] if x["budget_GiB"] == 14)
    bank = read(new / "replay/BANK_DESIGN.json")

    # The current timing is independently recomputed from integer nanosecond pairs.
    current_summary = read(new / "INTEGRATED_ANALYSIS.json")
    runs = {}
    for p in sorted((repo / new / "integrated/runs").glob("*/RUN.json")):
        run = read(p.relative_to(repo))
        assert run["passed"]
        for b in run["blocks"]:
            if b["quality"]:
                continue
            label = run["arm"] if b["mode"] == "F" else b["mode"]
            assert b["prefix_tokens"] == 272
            assert len(b["start_ns"]) == len(b["output_ready_ns"]) == len(b["status"]) == 64
            assert all(x == 0 for x in b["status"])
            ns = [end - start for start, end in zip(b["start_ns"], b["output_ready_ns"])]
            assert all(n > 0 for n in ns)
            runs.setdefault(label, []).append(ns)
    current = {}
    for arm, blocks in runs.items():
        vals = [n for b in blocks for n in b]
        total_ns = sum(vals)
        rate = len(vals) * 1e9 / total_ns
        mean = total_ns / len(vals) / 1e6
        assert abs(rate - current_summary["timing_summary"][arm]["tokens_per_second"]) < 1e-11
        current[arm] = {"blocks": len(blocks), "calls": len(vals), "total_ns": total_ns,
                        "mean_ms": mean, "T1_per_second": rate,
                        "block_mean_ms": [sum(b) / len(b) / 1e6 for b in blocks]}
    f_calls = current["F-before"]["calls"] + current["F-after"]["calls"]
    f_ns = current["F-before"]["total_ns"] + current["F-after"]["total_ns"]
    fmean = f_ns / f_calls / 1e6
    current["F-bracket"] = {"calls": f_calls, "mean_ms": fmean,
                             "T1_per_second": f_calls * 1e9 / f_ns}
    current["conditional_9p8_target_arithmetic"] = {
        "warning": "Not an old-vs-new speed comparison; assumes 9.8 on this same current trajectory/boundary.",
        "target_mean_ms": 1000 / 9.8,
        "required_time_reduction_ms": current["P"]["mean_ms"] - 1000 / 9.8,
        "required_time_reduction_percent": (1 - (1000 / 9.8) / current["P"]["mean_ms"]) * 100,
        "required_rate_increase_percent": (9.8 / current["P"]["T1_per_second"] - 1) * 100,
    }
    current["C_minus_F_mean_ms"] = current["C"]["mean_ms"] - fmean
    current["S_minus_P_mean_ms"] = current["S"]["mean_ms"] - current["P"]["mean_ms"]

    report = {
        "scope": "Independent saved-data arithmetic only. No engine, supplied scripts, GPU, model, or API run.",
        "pin": "d4d50b651ff7754ee2189228762468fc02971bfa",
        "historical_output_evidence_limit": "Old RESULT.json contains recorded token/text hashes; original full native responses are not in these run folders. Hash equality here compares stored hashes, not an independent rehash of missing output IDs.",
        "exact_user_9p8_origin": "unresolved; fresh-library-ja 9.7521465 is a nearby primary value, not an established identity",
        "final_adopted_ABBA": final,
        "old_dispatcher_all_miss_ABBA": miss,
        "old_hot14": {"cases": cases, "total_ms": hot_ms, "decode_steps": 127 * len(cases),
                      "aggregate_decode_tok_s": hot_rate,
                      "control_decode_tok_s": miss["control"]["aggregate_decode_tok_s"],
                      "gain_percent": (hot_rate / miss["control"]["aggregate_decode_tok_s"] - 1) * 100,
                      "full128_recorded_token_hash_matches": sum(c["same_recorded_token_hash"] for c in cases),
                      "capacity_14_GiB": b14,
                      "GPU_full_layers37_39_native_bytes": capacity["GPU_layers37_39_native_bytes"],
                      "GPU_full_layers37_39_native_GiB": capacity["GPU_layers37_39_native_bytes"] / 2**30,
                      "scope": pilot["scope"],
                      "adopted_as_default": pilot["adopted_as_default"],
                      "server_ttft_note": hot["ttft_measurement"],
                      "selected_model_pilot_quality": pilot["quality"]},
        "bank_comparison": {"old_experts": b14["triplets"], "new_experts": bank["experts"],
                            "old_native_bytes": b14["native_payload_bytes"], "new_native_bytes": bank["native_bytes"],
                            "old_to_new_native_byte_ratio": b14["native_payload_bytes"] / bank["native_bytes"],
                            "warning": "Old hot14 moved original full GPU layers37-39 back to CPU; new plan keeps those layers. Bank-byte ratio is not GPU-usage ratio."},
        "current_pilot": current,
    }
    (out / "HISTORY_RECOMPUTED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # Stable, exact text excerpts with one-based line anchors for the narrative memo.
    anchors = {
        hist / "review/CURRENT_NOTE_JA.txt": [(3, 19), (21, 31)],
        old / "EVENING_RESULTS_JA.txt": [(3, 16), (18, 42), (44, 54)],
        old / "checkpoint-1600/RESULTS_JA.txt": [(1, 200)],
        snap / "optimization-20261112/RESULTS_JA.txt": [(1, 160)],
        old / "fresh_probe.py": [(1, 220)],
        old / "prepare_hot_profiles.py": [(1, 100)],
        snap / "deepseek41-bringup-20261101/source/tools/server/server-common.h": [(414, 505)],
        snap / "deepseek41-bringup-20261101/source/tools/server/server-common.cpp": [(135, 161)],
        arch / "2026-11-16-next-optimization/CONSULT_PRO_JA.txt": [(1, 150)],
        arch / "2026-11-16-abs-operator/RESULTS_JA.txt": [(1, 180)],
        arch / "2026-11-17-mixed-operator/RESULTS_JA.txt": [(1, 180)],
        arch / "2026-11-17-route-observation/CONSULT_PRO_JA.txt": [(1, 150)],
        arch / "2026-11-17-route-capture/ACTUAL_RESULTS_JA.txt": [(1, 150)],
        arch / "2026-11-17-route-calibration/RESULTS_JA.txt": [(1, 180)],
        arch / "2026-11-17-gpu-workspace/RESULTS_JA.txt": [(1, 180)],
        new / "RESULTS_JA.txt": [(1, 69)],
        new / "ROOT_TRIAGE_JA.txt": [(1, 100)],
    }
    all_lines = []
    for rel, ranges in anchors.items():
        data = (repo / rel).read_bytes()
        sources[str(rel)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        lines = data.decode().splitlines()
        all_lines.append("\nFILE " + str(rel) + "\nSHA256 " + sources[str(rel)]["sha256"])
        for a, b in ranges:
            all_lines.extend(f"{i}: {lines[i-1]}" for i in range(a, min(b, len(lines)) + 1))
    (out / "SOURCE_EXCERPTS.txt").write_text("\n".join(all_lines) + "\n")
    (out / "SOURCE_SHA256.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"old_candidate_tps": hot_rate, "old_cases": len(cases),
                      "old_hash_match_count": sum(c["same_recorded_token_hash"] for c in cases),
                      "current_F_T1_per_second": current["F-bracket"]["T1_per_second"],
                      "current_P_T1_per_second": current["P"]["T1_per_second"],
                      "sources": len(sources), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
