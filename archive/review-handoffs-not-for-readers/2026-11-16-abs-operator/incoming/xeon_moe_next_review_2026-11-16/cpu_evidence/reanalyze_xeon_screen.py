#!/usr/bin/env python3
"""Independently inspect saved Xeon screens and earlier CPU experiments.

Reads JSON/source/logs only. Does not import archived code, compile, execute a
benchmark, connect to an API, or load a model. Output must be a new JSON file.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

COMMIT = "0c9eb472d2460048fb44cca949ad3722ed27b93b"
REL = Path("archive/review-handoffs-not-for-readers")
TYPES = {1: "IQ3_XXS", 2: "IQ2_S", 3: "IQ3_S"}
GGML_TYPES = {18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S", 39: "MXFP4"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def key(row):
    return row["type"], row["variant"], row["n"], row["rows"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    repo = args.repository.resolve()
    n = repo / REL / "2026-11-16-next-optimization"
    h = repo / REL / "2026-11-16-current"
    e = h / "snapshot/llm-lab/exploration-20261112"
    earlier = h / "snapshot/llm-lab/optimization-20261112"
    source_files = set()

    def track(path):
        source_files.add(path)
        return read(path)

    expected = {(t, "experimental_input_abs", k, m)
                for t in TYPES for k in (2304, 5120) for m in (256, 8192)}
    expected |= {(1, "stored_iq_rhs_bias62", k, m)
                 for k in (2304, 5120) for m in (256, 8192)}
    receipts = []; recomputed = []; samples = []
    max_ratio_error = max_median_error = 0.0
    for label, cpu, node in (("abs-original-xeon", 3, 0), ("abs-socket1-xeon", 9, 1)):
        d = track(n / label / "RESULT.json")
        receipts.append(d)
        require(d["test_passed"] is True and d["benchmark_passed"] is True, "Run failed")
        require(all(c["exit_code"] == 0 for c in d["commands"]), "Command failed")
        require(d["commands"][-1]["argv"][-1] == "bench", "Last command is not the full bench")
        parsed = lines(d["commands"][-1]["stdout"])
        require(parsed == d["benchmark_rows"], "Parsed rows differ from final raw stdout")
        require(len(parsed) == 16 and {key(r) for r in parsed} == expected, "Case inventory mismatch")
        correctness = lines(d["commands"][1]["stdout"])
        require(correctness == d["correctness_rows"], "Correctness stdout mismatch")
        totals = {field: sum(r[field] for r in correctness)
                  for field in ("bit_exact_outputs", "unsafe_rejections", "stored_bias_variant_bit_checks")}
        require(totals == d["totals"], "Correctness total mismatch")
        require(totals == {"bit_exact_outputs": 61440, "unsafe_rejections": 4800,
                           "stored_bias_variant_bit_checks": 20480}, "Unexpected correctness family")
        require(len(correctness) == 24, "Expected 3 types x 8 MXCSR settings")
        expected_correctness = {(t, flush | rounding) for t in TYPES
                                for flush in (0, 0x8040)
                                for rounding in (0, 0x2000, 0x4000, 0x6000)}
        require({(r["type"], r["mxcsr"] & 0xe040) for r in correctness} == expected_correctness,
                "Missing or duplicated correctness control setting")
        require(all(r["kind"] == "correctness" and r["bit_exact_outputs"] == 2560
                    and r["unsafe_rejections"] == 200
                    and r["stored_bias_variant_bit_checks"] == (2560 if r["type"] == 1 else 0)
                    for r in correctness), "Unexpected correctness per-setting count")
        for path, expected_sha in d["source_sha256"].items():
            actual = n / "incoming/xeon_moe_review_2026-11-16/cpu/iq_abs_rhs_probe.cpp" if Path(path).is_absolute() else h / path
            source_files.add(actual)
            require(digest(actual) == expected_sha, "Source SHA mismatch: " + path)
        for row in parsed:
            require(row["preparation_once_per_matrix_included"] is True, "Preparation absent")
            require(row["weight_payload_bytes"] == row["n"] // 256 * row["rows"] * 144, "Weight bytes")
            require(row["sidecar_bytes"] == row["n"], "Sidecar bytes")
            require(row["traversals_per_arm"] == max(1, 32768 // row["rows"]), "Traversal count")
            record = {"socket": node, "cpu": cpu, "type": TYPES[row["type"]], "variant": row["variant"],
                      "K": row["n"], "rows": row["rows"], "weight_payload_bytes": row["weight_payload_bytes"],
                      "traversals_per_arm": row["traversals_per_arm"], "dots_per_arm": row["rows"] * row["traversals_per_arm"]}
            for clock, prefix in (("wall", ""), ("thread_cpu", "thread_cpu_")):
                akey = "current_" + ("thread_cpu_" if prefix else "") + "ns_per_dot"
                bkey = "candidate_" + ("thread_cpu_" if prefix else "") + "ns_per_dot"
                a = row[akey]; b = row[bkey]
                saved_ratios = row[prefix + "paired_speedups"]
                require(len(a) == len(b) == 8 and len(saved_ratios) == 4, "Timing count")
                require(all(math.isfinite(v) and v > 0 for v in a + b + saved_ratios), "Invalid timing")
                ratios = [sum(a[2*j:2*j+2]) / sum(b[2*j:2*j+2]) for j in range(4)]
                median = statistics.median(ratios)
                ratio_error = max(abs(x-y) for x, y in zip(ratios, saved_ratios))
                median_error = abs(median - row[prefix + "paired_median_speedup"])
                require(ratio_error <= 0.00000051 and median_error <= 0.00000051, "Recomputed ratio disagrees with rounded stdout")
                max_ratio_error = max(max_ratio_error, ratio_error); max_median_error = max(max_median_error, median_error)
                record[clock] = {"paired_ratios_from_arm_times": ratios, "paired_median_ratio": median,
                                 "gain_percent": 100*(median-1), "min_ratio": min(ratios), "max_ratio": max(ratios),
                                 "current_ns_per_dot": a, "candidate_ns_per_dot": b,
                                 "current_median_ns_per_dot": statistics.median(a), "candidate_median_ns_per_dot": statistics.median(b),
                                 "baseline_max_to_min_ratio": max(a)/min(a),
                                 "passes_practical_screen": median >= 1.03 and min(ratios) > 1}
                for j, ratio in enumerate(ratios):
                    order = ["candidate", "current", "current", "candidate"] if j % 2 == 0 else ["current", "candidate", "candidate", "current"]
                    av = iter(a[2*j:2*j+2]); bv = iter(b[2*j:2*j+2])
                    values = [next(av) if arm == "current" else next(bv) for arm in order]
                    samples.append({"socket": node, "clock": clock, "type": record["type"], "variant": row["variant"],
                                    "K": row["n"], "rows": row["rows"], "retained_pair": j,
                                    "arm_order": order, "ns_per_dot": values, "ratio": ratio})
            recomputed.append(record)
    require(receipts[0]["probe_sha256"] == receipts[1]["probe_sha256"], "Reported binary SHA differs")
    require(receipts[1]["commands"][:-1] == receipts[0]["commands"], "Unexpected replication provenance")
    require(receipts[0]["correctness_rows"] == receipts[1]["correctness_rows"], "Expected reused correctness receipt")
    published = track(n / "CPU_SCREEN_RESULTS.json")
    combined = []
    for k in sorted(expected):
        rows = [r for r in recomputed if (next(t for t,name in TYPES.items() if name == r["type"]), r["variant"], r["K"], r["rows"]) == k]
        require(len(rows) == 2, "Missing socket")
        passed = all(r[c]["passes_practical_screen"] for r in rows for c in ("wall", "thread_cpu"))
        exported = [r for r in published["rows"] if (r["type"],r["variant"],r["K"],r["rows"]) == (TYPES[k[0]],k[1],k[2],k[3])]
        require(len(exported) == 1 and exported[0]["passes_practical_screen_both_sockets"] == passed, "Published decision mismatch")
        combined.append({"type": TYPES[k[0]], "variant": k[1], "K": k[2], "rows": k[3],
                         "both_sockets_pass": passed, "wall_gain_percent": [r["wall"]["gain_percent"] for r in rows],
                         "thread_cpu_gain_percent": [r["thread_cpu"]["gain_percent"] for r in rows]})

    # The old bias62 comparison used the actual backend and different evidence.
    old_names = ["20261112-132050-iq-rhs-control-r1", "20261112-132845-iq-rhs-candidate-r1",
                 "20261112-133311-iq-rhs-candidate-r2", "20261112-133953-iq-rhs-control-r2"]
    old_runs = [track(e / "heldout" / label / "RESULT.json") for label in old_names]
    expected_input = track(e / "HELDOUT_INPUTS.json")
    expected_cases = [r["name"] for r in expected_input["cases"]]
    input_sha = digest(e / "HELDOUT_INPUTS.json")
    for run in old_runs:
        require(run["complete"] is True and run["error"] is None, "Old bias run incomplete")
        require([r["name"] for r in run["cases"]] == expected_cases, "Old bias inventory mismatch")
        require(run["input_sha256"] == input_sha, "Old bias input SHA mismatch")
    old_summary = track(e / "iq-rhs/LIVE_ABBA.json")
    old_rates = []
    for indices in ([0,3],[1,2]):
        rows = [r for i in indices for r in old_runs[i]["cases"]]
        require(all(math.isfinite(r["timings"]["predicted_ms"]) and r["timings"]["predicted_ms"] > 0 for r in rows), "Old bias time invalid")
        old_rates.append(1000*sum(r["timings"]["predicted_n"]-1 for r in rows)/sum(r["timings"]["predicted_ms"] for r in rows))
    old_gain = (old_rates[1]/old_rates[0]-1)*100
    require(math.isclose(old_gain, old_summary["aggregate_gain_percent"], abs_tol=1e-10), "Old bias aggregate mismatch")
    old_hash_equal = all(len({(run["cases"][j]["output_sha256"],run["cases"][j]["token_sha256"]) for run in old_runs}) == 1 for j in range(8))

    # Prior row blocking is outside exploration-20261112, but present in H.
    prior_pair = []
    for name in ("compact-pair-synthetic-abba", "compact-pair-unrolled-synthetic-abba"):
        path = earlier / "execution" / name / "stdout.log"; source_files.add(path)
        parsed = lines(path.read_text())
        rows = [r for r in parsed if "speedup" in r]
        require(len(rows) == 12 and parsed[-1] == {"pass": True, "outputs_checked": 886560}, "Old pair inventory")
        prior_pair.append({"experiment": name, "conditions": len(rows), "speedup_min": min(r["speedup"] for r in rows),
                           "speedup_max": max(r["speedup"] for r in rows), "pass_receipt": parsed[-1], "rows": rows,
                           "limits": "Upper medians of arm times; raw per-arm samples not preserved. Different synthetic shape and reduction chunking from the new screen."})
    census = track(e / "training/20261112-075449/TRAINING_COUNTS.json")
    composition = collections.Counter((GGML_TYPES.get(r["type"],str(r["type"])),r["k"],r["rows"]) for r in census["records"])
    controls = []
    for node in (0,1):
        for k in (2304,5120):
            for m in (256,8192):
                rr = [r for r in recomputed if r["socket"] == node and r["type"] == "IQ3_XXS" and r["K"] == k and r["rows"] == m]
                a = next(r for r in rr if r["variant"] == "experimental_input_abs")
                b = next(r for r in rr if r["variant"] == "stored_iq_rhs_bias62")
                controls.append({"socket":node,"K":k,"rows":m,
                                 "abs_block_baseline_median_ns":a["wall"]["current_median_ns_per_dot"],
                                 "bias_block_baseline_median_ns":b["wall"]["current_median_ns_per_dot"],
                                 "baseline_ratio_abs_block_over_bias_block":a["wall"]["current_median_ns_per_dot"]/b["wall"]["current_median_ns_per_dot"],
                                 "scope":"Diagnostic of separate control blocks, not a direct abs-versus-bias comparison."})
    for path in [n/"CONSULT_PRO_JA.txt", n/"analyze_screen.py", n/"EXPERIMENT_PLAN.txt", n/"RUN_CONTEXT.json", n/"BASELINE_IDLE.json", n/"AFTER_SCREEN.json",
                 n/"incoming/xeon_moe_review_2026-11-16/cpu/run_iq_abs_rhs_probe.py",e/"sparse-cache/cost-probe.cpp",e/"COST384_RESULTS.json",e/"build_unroll_probe.py",
                 e/"checkpoint-1600/RESULTS_JA.txt",h/"review/capture/CURRENT_STATE.json",
                 earlier/"RESULTS_JA.txt",earlier/"cpu-bias/biased-iq3.inc",earlier/"cpu-bias/test-bias.cpp",
                 e/"mx-prepack/release-candidate-a/sources/ggml-cpu.c",
                 h/"snapshot/llm-lab/mxfp4-hybrid-20261102/hybrid/engine/ggml/src/ggml-common.h"]:
        source_files.add(path)
    for sub in ("cpu-pair", "cpu-pair-unrolled"):
        for name in ("paired-rows.inc", "test-pair.cpp", "BUILD.json"):
            source_files.add(earlier/sub/name)
    for path in sorted((e/"runs").glob("*/perf-control.json")):
        source_files.add(path)
    for path in sorted((earlier/"runs").glob("*/ANALYSIS.json")):
        source_files.add(path)
    for sub in ("iq-rhs/trace-calls.cpp", "mx-prepack/backend/trace-calls.cpp", "FINAL_ABBA.json"):
        source_files.add(e/sub)
    abs_rows = [r for r in combined if r["variant"] == "experimental_input_abs"]
    result = {
      "source_commit":COMMIT,
      "scope":"Independent read-only reaggregation of saved Xeon runs. No new performance measurement, CPU/GPU/model/API execution, or archived script execution.",
      "checks":{"source_hashes_match":True,"all_raw_stdout_matches_parsed":True,"case_inventory_each_socket":16,
                "raw_wall_arms_each_socket":128,"raw_thread_cpu_arms_each_socket":128,
                "max_ratio_reconstruction_error":max_ratio_error,"max_median_reconstruction_error":max_median_error,
                "reported_binary_sha_equal":True,"reported_binary_sha256":receipts[0]["probe_sha256"],
                "source_hash_entries_each_receipt":[len(d["source_sha256"]) for d in receipts],
                "binary_bytes_provided":False,"replication_command_prefix_matches_original":True,
                "correctness_reused_not_added":True},
      "unique_correctness_family":receipts[0]["totals"],
      "screen":{"abs_conditions":12,"abs_pass_both_sockets":sum(r["both_sockets_pass"] for r in abs_rows),
                "all_abs_wall_pairs_above_one":all(r["wall"]["min_ratio"] > 1 for r in recomputed if r["variant"] == "experimental_input_abs"),
                "abs_wall_pair_count":96,"conditions":combined},
      "recomputed_rows":recomputed,"chronological_retained_arms":samples,
      "separate_block_control_diagnostics":controls,
      "old_bias62_backend_comparison":{"control_decode_steps_per_s":old_rates[0],"candidate_decode_steps_per_s":old_rates[1],
                                       "gain_percent":old_gain,"all_output_token_hashes_equal":old_hash_equal,
                                       "scope":"Earlier actual-backend model ABBA, not a prediction for abs or the new standalone fixture."},
      "older_paired_row_experiments":prior_pair,
      "training_projection_shape_counts":[{"type":t,"K":k,"M":m,"tensors":v} for (t,k,m),v in sorted(composition.items())],
      "memory_geometry":{"actual_projection_payload_bytes_each_expert":2304*5120//256*144,
                         "six_distinct_experts_payload_bytes":6*2304*5120//256*144,
                         "down_q8_k_bytes_six_inputs":6*9*292,"down_abs_bytes_six_inputs":6*2304,
                         "down_combined_bytes_six_inputs":6*(9*292+2304),
                         "scope":"Allocated data totals only. Live L1 demand depends on worker/chunk order and reuse, not the total alone."},
      "amdahl_sensitivity":{"formula":"S_total = 1 / (1 - p + p / r + h); p is affected baseline critical-path fraction, r local speedup, h added overhead / baseline total time.",
                            "not_measured_model_fractions":True,
                            "why_fraction_is_not_identified":"Training counts have no time fields; saved earlier CPU-graph aggregates do not isolate final IQ2_S/IQ3_S compact dots; perf-control files record capture controls, and call counters do not time calls. Whole-model ABBA records overall predicted time only.",
                            "zero_overhead_scenarios":[{"local_speedup":r,"affected_fraction":p,"total_speedup":1/(1-p+p/r)} for r in (1.03,1.04,1.05,1.06) for p in (.1,.25,.5)],
                            "overhead_budget_formula":"h <= p * (1 - 1/r) - g/(1+g) for target total gain g. At g=0 use a strict inequality for any positive gain. Negative budgets mean the target is impossible even with zero added overhead.",
                            "overhead_budgets":[{"local_speedup":r,"affected_fraction":p,"target_total_gain":g,"max_extra_overhead_fraction_of_baseline":p*(1-1/r)-g/(1+g)} for r in (1.03,1.04,1.05,1.06) for p in (.1,.25,.5) for g in (0,.005,.01)],
                            "affected_fraction_needed_for_1pct_total_gain":[{"local_speedup":r,"p_min_no_overhead":(1-1/1.01)/(1-1/r)} for r in (1.03,1.04,1.05,1.06)]},
      "source_files":[{"path":str(p.relative_to(repo)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(source_files)]
    }
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps({"output":str(args.output),"abs_pass_both":result["screen"]["abs_pass_both_sockets"],
                      "all_abs_wall_pairs_above_one":result["screen"]["all_abs_wall_pairs_above_one"],
                      "old_bias_gain_percent":old_gain,"sources":len(source_files)},ensure_ascii=False))


if __name__ == "__main__":
    main()
