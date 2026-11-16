#!/usr/bin/env python3
"""Read frozen saved results; optionally probe a pinned pure analyzer in temp copies.

No model, GPU, API, controller, service, or hardware command is run.  The only
subprocess is the reviewed analyze_heldout.py on synthetic copies of result JSON.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


ANALYZER_SHA = "950f459457224c07f5b820eb56e28112b2367336919050643ad1e90a9e8aa694"
COMMIT = "7d8b077afcf1258602366245f1567e820255defa"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def rate(cases):
    return 1000 * sum(c["timings"]["predicted_n"] - 1 for c in cases) / sum(c["timings"]["predicted_ms"] for c in cases)


def arm_summary(runs):
    cases = [c for r in runs for c in r["cases"]]
    steps = sum(c["timings"]["predicted_n"] - 1 for c in cases)
    tokens = sum(c["output_token_count"] for c in cases)
    elapsed = sum(c["elapsed_s"] for c in cases)
    prompt_ms = sum(c["timings"]["prompt_ms"] for c in cases)
    return {
        "requests": len(cases), "generated_tokens": tokens, "decode_steps": steps,
        "decode_ms_sum": sum(c["timings"]["predicted_ms"] for c in cases),
        "pooled_decode_steps_per_second": rate(cases),
        "arithmetic_mean_reported_decode_tok_s": statistics.mean(c["timings"]["predicted_per_second"] for c in cases),
        "prompt_ms_sum": prompt_ms, "mean_prompt_ms_proxy": prompt_ms / len(cases),
        "request_elapsed_s_sum": elapsed, "mean_request_elapsed_s": elapsed / len(cases),
        "generated_tokens_per_request_elapsed_second": tokens / elapsed,
    }


def compare_arms(control, candidate, names):
    a, b = arm_summary(control), arm_summary(candidate)
    rows = []
    for index, name in enumerate(names):
        ac = [r["cases"][index] for r in control]
        bc = [r["cases"][index] for r in candidate]
        rows.append({
            "name": name, "control_decode_tok_s": rate(ac), "candidate_decode_tok_s": rate(bc),
            "decode_rate_gain_percent": 100 * (rate(bc) / rate(ac) - 1),
            "output_and_token_hashes_identical": len({(c["output_sha256"], c["token_sha256"]) for c in ac + bc}) == 1,
            "prompt_tokens": ac[0]["prompt_tokens"],
        })
    return {
        "control": a, "candidate": b, "cases": rows,
        "decode_rate_gain_percent": 100 * (b["pooled_decode_steps_per_second"] / a["pooled_decode_steps_per_second"] - 1),
        "decode_time_reduction_percent": 100 * (1 - a["pooled_decode_steps_per_second"] / b["pooled_decode_steps_per_second"]),
        "arithmetic_mean_rate_gain_percent": 100 * (b["arithmetic_mean_reported_decode_tok_s"] / a["arithmetic_mean_reported_decode_tok_s"] - 1),
        "request_wall_token_rate_gain_percent": 100 * (b["generated_tokens_per_request_elapsed_second"] / a["generated_tokens_per_request_elapsed_second"] - 1),
        "mean_request_wall_time_reduction_percent": 100 * (1 - b["mean_request_elapsed_s"] / a["mean_request_elapsed_s"]),
        "mean_prompt_time_reduction_percent": 100 * (1 - b["mean_prompt_ms_proxy"] / a["mean_prompt_ms_proxy"]),
    }


def inventory_input_usage(root, read, fresh_sha, older_sha):
    rows = []
    for path in sorted((root / "heldout").glob("*/RESULT.json")):
        value = read(path.relative_to(root))
        source_set = "fresh" if value["input_sha256"] == fresh_sha else "earlier_heldout" if value["input_sha256"] == older_sha else "other"
        rows.append({"source": str(path.relative_to(root)), "label": value["label"],
                     "input_set": source_set, "input_sha256": value["input_sha256"],
                     "started_unix": value.get("started_unix"),
                     "timestamp_in_directory_name": path.parent.name.split("-", 2)[:2],
                     "cases": len(value["cases"]), "complete": value["complete"], "error": value["error"]})
    require(len(rows) == 40 and all(row["input_set"] != "other" for row in rows), "Saved heldout inventory changed")
    fresh_rows = [r for r in rows if r["input_set"] == "fresh"]
    require([r["label"] for r in sorted(fresh_rows, key=lambda r: r["started_unix"])[:4]] ==
            ["final-control-r1", "final-candidate-r1", "final-candidate-r2", "final-control-r2"], "Earlier saved fresh usage exists")
    return {
        "saved_heldout_runs": len(rows), "earlier_set_runs": sum(r["input_set"] == "earlier_heldout" for r in rows),
        "fresh_set_runs": len(fresh_rows), "earliest_four_saved_fresh_runs_are_final_ABBA": True,
        "subsequent_fresh_run_labels": [r["label"] for r in sorted(fresh_rows, key=lambda r: r["started_unix"])[4:]],
        "also_reused_for_fixedprefix_quality": ["hot-quality-control", "hot-quality14"],
        "interpretation": "Final ABBA is the first saved use of fresh inputs. Later poll/hybrid/hot and fixed-prefix evaluations reuse them, so they are no longer a new blind holdout for subsequent choices. This inventory cannot prove the absence of unrecorded prior use or generalization error.",
        "rows": rows,
    }


def gate_probe(root, final_runs, outdir):
    source = root / "analyze_heldout.py"
    require(digest(source) == ANALYZER_SHA, "Unreviewed analyzer; refusing execution")
    results = {}
    for mode in ("unchanged_control", "same_case_missing_in_every_run", "one_arm_case_missing_control", "one_run_incomplete_control"):
        with tempfile.TemporaryDirectory(prefix="saved-analyzer-gate-") as temporary:
            folder = Path(temporary)
            shutil.copy2(source, folder / source.name)
            for manifest in ("FRESH_INPUTS.json", "FRESH_PROMPTS.json", "FINAL_SELECTION.json"):
                shutil.copy2(root / manifest, folder / manifest)
            for index, original in enumerate(final_runs):
                value = copy.deepcopy(original)
                if mode == "same_case_missing_in_every_run":
                    value["cases"] = value["cases"][:-1]
                elif mode == "one_arm_case_missing_control" and index == 0:
                    value["cases"] = value["cases"][:-1]
                elif mode == "one_run_incomplete_control" and index == 0:
                    value["complete"] = False
                path = folder / "heldout" / ("synthetic-" + value["label"]) / "RESULT.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(value))
            command = [sys.executable, str(folder / source.name),
                       "--control", "final-control-r1", "final-control-r2",
                       "--candidate", "final-candidate-r1", "final-candidate-r2",
                       "--out", "ANALYSIS.json", "--scope", "Synthetic copied receipts; not a new inference result", "--final"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=10)
            data = json.loads((folder / "ANALYSIS.json").read_text()) if (folder / "ANALYSIS.json").exists() else None
            results[mode] = {
                "exit_code": process.returncode,
                "analyzer_case_names": [x["name"] for x in data["rows"]] if data else None,
                "comparison_accepted": data["comparison_accepted"] if data else None,
                "requests_per_arm": data["aggregate"]["control"]["case_count"] if data else None,
                "stderr": process.stderr,
            }
    require(results["unchanged_control"]["exit_code"] == 0, "Unchanged positive control failed")
    require(results["same_case_missing_in_every_run"]["comparison_accepted"] is True, "Expected missing-case path changed")
    require(len(results["same_case_missing_in_every_run"]["analyzer_case_names"]) == 7, "Synthetic drop was not applied")
    require(results["one_arm_case_missing_control"]["exit_code"] != 0, "One-arm mismatch should be rejected")
    require(results["one_run_incomplete_control"]["exit_code"] != 0, "Incomplete run should be rejected")
    return {
        "scope": "Pinned analyzer on temporary copied receipts only; original saved runs remain complete and unchanged.",
        "expected_fresh_manifest_retained": True,
        "analyzer_sha256": ANALYZER_SHA,
        "observed_issue": "Equal case lists are checked, but the lists are not matched to the expected frozen input manifest.",
        "cases": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", type=Path, required=True, help="2026-11-16-current directory")
    ap.add_argument("--output", type=Path, required=True, help="new output JSON file")
    ap.add_argument("--skip-gate-probe", action="store_true")
    args = ap.parse_args()
    require(not args.output.exists(), "Output already exists")
    root = args.handoff.resolve() / "snapshot/llm-lab/exploration-20261112"
    sources = {}

    def read(relative):
        path = root / relative
        sources[str(relative)] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        return json.loads(path.read_text())

    def labeled(label, category="heldout"):
        matches = list((root / category).glob("*-" + label + "/RESULT.json"))
        require(len(matches) == 1, "Ambiguous or absent run " + label)
        return read(matches[0].relative_to(root))

    inputs = read("FRESH_INPUTS.json")
    prompts = read("FRESH_PROMPTS.json")
    names = [x["name"] for x in inputs["cases"]]
    input_sha = digest(root / "FRESH_INPUTS.json")
    require(names == [x["name"] for x in prompts["cases"]] and len(names) == len(set(names)) == 8, "Fresh case inventory mismatch")
    require(inputs["source_sha256"] == digest(root / "FRESH_PROMPTS.json"), "Frozen prompt source mismatch")
    final_labels = ["final-control-r1", "final-candidate-r1", "final-candidate-r2", "final-control-r2"]
    final_runs = [labeled(label) for label in final_labels]
    selection = read("FINAL_SELECTION.json")
    expected_configs = [selection["baseline_config_sha256"], selection["candidate_config_sha256"],
                        selection["candidate_config_sha256"], selection["baseline_config_sha256"]]
    checked_runs = []
    for run, expected_config in zip(final_runs, expected_configs):
        require(run["complete"] is True and run["error"] is None, "Incomplete final run")
        require([c["name"] for c in run["cases"]] == names, "Final case inventory differs from frozen inputs")
        require(run["input_sha256"] == input_sha and run["config_sha256"] == expected_config, "Final run source/config mismatch")
        for case, frozen in zip(run["cases"], inputs["cases"]):
            timing = case["timings"]
            require(case["prompt_tokens"] == timing["prompt_n"] == len(frozen["tokens"]) and timing["cache_n"] == 0, "Prompt shape/cache mismatch")
            require(case["output_token_count"] == timing["predicted_n"] == 128 and case["stop_type"] == "limit", "Incomplete generated token count")
            require(all(math.isfinite(timing[key]) and timing[key] > 0 for key in ("predicted_ms", "prompt_ms", "predicted_per_second")), "Invalid timing")
            require(math.isclose(timing["predicted_per_second"], 127000 / timing["predicted_ms"], rel_tol=1e-12), "Server rate formula mismatch")
        checked_runs.append({"label": run["label"], "start": run["started_unix"], "finish": run["finished_unix"],
                             "config_sha256": run["config_sha256"], "pooled_decode_tok_s": rate(run["cases"])})
    require(all(a["finished_unix"] < b["started_unix"] for a, b in zip(final_runs, final_runs[1:])), "ABBA runs overlap or wrong order")
    final = compare_arms([final_runs[0], final_runs[3]], [final_runs[1], final_runs[2]], names)
    final["run_order"] = checked_runs
    final["validation"] = {"matches_exact_frozen_case_inventory": True, "all_32_case_receipts_complete": True,
                           "input_manifest_sha_matches": True, "config_hashes_match_final_selection": True,
                           "ordered_nonoverlapping_ABBA": True,
                           "all_output_hashes_identical": all(x["output_and_token_hashes_identical"] for x in final["cases"]),
                           "original_128_token_sequences_or_text_in_receipts": False}
    saved_final = read("FINAL_ABBA.json")
    require(math.isclose(final["decode_rate_gain_percent"], saved_final["aggregate_gain_percent"], abs_tol=1e-10), "Final aggregate differs")
    require(digest(root / "FINAL_ABBA.json") == digest(root / "checkpoint-1600/FINAL_ABBA.json"), "Checkpoint differs")
    final["saved_summary_exact_aggregate_match"] = True
    final["formulas"] = {
        "decode": "1000 * sum(predicted_n - 1) / sum(predicted_ms); 16 * 127 steps per arm",
        "request_wall_rate": "sum(output_token_count) / sum(elapsed_s); 16 * 128 generated tokens per arm",
        "request_wall_time": "elapsed_s is monotonic wall time around a native nonstream /completion HTTP call; excludes warmup and switching; does not measure Open WebUI rendering or per-token delivery",
        "ttft": "ttft_s = server prompt_ms/1000; this is a prompt-processing proxy, not observed streaming TTFT",
    }

    hot_a = [labeled("hot-disabled-r1"), labeled("hot-disabled-r2")]
    hot_b = [labeled("hot14-r1")]
    for run in hot_a + hot_b:
        require(run["complete"] is True and run["error"] is None and run["input_sha256"] == input_sha, "Hot pilot incomplete or input mismatch")
        require([c["name"] for c in run["cases"]] == names, "Hot case inventory mismatch")
        require(all(c["output_token_count"] == c["timings"]["predicted_n"] == 128 for c in run["cases"]), "Hot output length differs")
    hot = compare_arms(hot_a, hot_b, names)
    hot["scope"] = "Two controls before one candidate; outputs differ in seven of eight cases; not balanced ABBA and not the same autoregressive work."
    hot["run_order"] = [{"label": r["label"], "started_unix": r["started_unix"]} for r in hot_a + hot_b]
    saved_hot = read("hot-ffn/MODEL_PILOT_RESULTS.json")
    require(math.isclose(hot["decode_rate_gain_percent"], saved_hot["free_generation_gain_percent"], abs_tol=1e-10), "Hot aggregate differs")
    prefill_a = read("runs/20261112-175308-hot-final-control-prefill-count-plain-c4096-n16/RESULT.json")
    prefill_b = read("runs/20261112-180852-hot14-prefill-count-plain-c4096-n16/RESULT.json")
    require(prefill_a["error"] is None and prefill_b["error"] is None, "Prefill receipt failed")
    pa, pb = (d["final"]["timings"]["prompt_ms"] for d in (prefill_a, prefill_b))
    hot["prefill4096"] = {"control_ms": pa, "candidate_ms": pb, "time_increase_percent": 100 * (pb / pa - 1), "runs_per_arm": 1}
    hot["all_miss"] = compare_arms(hot_a, [labeled("hot-miss-r1"), labeled("hot-miss-r2")], names)
    dispatch = saved_hot["dispatch_layers"]
    require({r["layer"] for r in dispatch} == set(range(40)), "Missing dispatch layer")
    total = hits = selected_total = selected_hits = 0
    histogram = [0] * 7
    for row in dispatch:
        require(sum(row["H_histogram"]) == row["calls"], "Dispatch counts do not sum")
        h = sum(k * count for k, count in enumerate(row["H_histogram"]))
        total += row["calls"] * 6; hits += h
        histogram = [a + b for a, b in zip(histogram, row["H_histogram"])]
        if row["layer"] < 37:
            selected_total += row["calls"] * 6; selected_hits += h
    hot["routing"] = {"total_routes": total, "hot_routes": hits, "route_hit_fraction_all40": hits / total,
                      "route_hit_fraction_layers0_36": selected_hits / selected_total,
                      "layer_call_H_histogram": histogram,
                      "scope": "Whole pilot mixes free generation, fixed-prefix scoring, long/short counts and image; not per-case decode-only hit rate."}
    require(hits == saved_hot["actual_hot_routes"] and total == saved_hot["actual_total_routes"], "Hot route total differs")

    qa = labeled("hot-quality-control", "hot-ffn/quality")
    qb = labeled("hot-quality14", "hot-ffn/quality")
    sequences = read("hot-ffn/QUALITY_SEQUENCES.json")
    require(qa["complete"] is True and qb["complete"] is True and qa["error"] is None and qb["error"] is None, "Quality run incomplete")
    require(qa["sequence_sha256"] == qb["sequence_sha256"] == digest(root / "hot-ffn/QUALITY_SEQUENCES.json"), "Sequence hash mismatch")
    require(qa["input_sha256"] == qb["input_sha256"] == input_sha, "Quality input mismatch")
    require([c["name"] for c in qa["cases"]] == [c["name"] for c in qb["cases"]] == names, "Quality case inventory mismatch")
    qrows = []
    for ca, cb, seq, inp in zip(qa["cases"], qb["cases"], sequences["cases"], inputs["cases"]):
        require(seq["prompt_tokens"] == inp["tokens"] and len(seq["continuation_tokens"]) == 64, "Quality teacher sequence mismatch")
        require(len(ca["points"]) == len(cb["points"]) == 32, "Quality expected point inventory mismatch")
        for j, (x, y) in enumerate(zip(ca["points"], cb["points"])):
            known = seq["continuation_tokens"][j + 1]
            require(x["j"] == y["j"] == j and x["known_next_id"] == y["known_next_id"] == known, "Quality point/teacher mismatch")
            require(x["timings"]["prompt_n"] == y["timings"]["prompt_n"] == 1 and
                    x["timings"]["cache_n"] == y["timings"]["cache_n"] == len(inp["tokens"]) + j, "Quality cached token count mismatch")
            row = {"name": ca["name"], "j": j, "teacher_token_index": j + 1,
                   "greedy_match": x["generated_id"] == y["generated_id"],
                   "control_teacher_match": x["generated_id"] == known,
                   "candidate_teacher_match": y["generated_id"] == known,
                   "control_prompt_ms": x["timings"]["prompt_ms"], "candidate_prompt_ms": y["timings"]["prompt_ms"],
                   "both_probability_records": x["probability"] is not None and y["probability"] is not None}
            if row["both_probability_records"]:
                da = {t["id"]: t["logprob"] for t in x["probability"]["top_logprobs"]}
                db = {t["id"]: t["logprob"] for t in y["probability"]["top_logprobs"]}
                require(len(da) == len(db) == 100 and all(math.isfinite(v) and v <= 1e-5 for v in list(da.values()) + list(db.values())), "Invalid top100 record")
                aa, bb = {k: math.exp(v) for k, v in da.items()}, {k: math.exp(v) for k, v in db.items()}
                common = aa.keys() & bb.keys()
                row.update({"common_top_ids": len(common), "common_L1": sum(abs(aa[k] - bb[k]) for k in common),
                            "max_common_probability_delta": max(abs(aa[k] - bb[k]) for k in common),
                            "control_top100_mass": sum(aa.values()), "candidate_top100_mass": sum(bb.values()),
                            "teacher_logprob_delta": db[known] - da[known] if known in da and known in db else None})
            qrows.append(row)
    paired = [r for r in qrows if r["both_probability_records"]]
    deltas = [r["teacher_logprob_delta"] for r in paired if r["teacher_logprob_delta"] is not None]
    quality = {
        "frozen_teacher_tokens_per_case": 64, "scored_points_per_case": 32,
        "scored_teacher_token_indices_zero_based": [1, 32], "points": len(qrows),
        "greedy_agreement": sum(r["greedy_match"] for r in qrows),
        "control_self_replay": sum(r["control_teacher_match"] for r in qrows),
        "probability_pairs": len(paired), "teacher_logprob_pairs": len(deltas),
        "mean_teacher_logprob_delta": statistics.mean(deltas),
        "max_common_probability_delta": max(r["max_common_probability_delta"] for r in paired),
        "mean_common_L1": statistics.mean(r["common_L1"] for r in paired),
        "median_common_L1": statistics.median(r["common_L1"] for r in paired),
        "common_probability_delta_over_1_percentage_point": sum(r["max_common_probability_delta"] > .01 for r in paired),
        "common_probability_delta_over_10_percentage_points": sum(r["max_common_probability_delta"] > .1 for r in paired),
        "mean_control_T1_ms": statistics.mean(r["control_prompt_ms"] for r in qrows),
        "mean_candidate_T1_ms": statistics.mean(r["candidate_prompt_ms"] for r in qrows),
        "T1_time_reduction_percent": 100 * (1 - sum(r["candidate_prompt_ms"] for r in qrows) / sum(r["control_prompt_ms"] for r in qrows)),
        "minimum_top100_probability_mass": min(min(r["control_top100_mass"], r["candidate_top100_mass"]) for r in paired),
        "max_top100_mass_above_one_rounding": max(max(r["control_top100_mass"], r["candidate_top100_mass"]) - 1 for r in paired),
        "disagreements": [{k: r[k] for k in ("name", "j", "teacher_token_index")} for r in qrows if not r["greedy_match"]],
        "missing_probability_records": [{k: r[k] for k in ("name", "j")} for r in qrows if not r["both_probability_records"]],
        "scope": "Same token prefixes do not freeze hidden activations, KV values or routing. Partial top100 distribution comparison and baseline self-replay, not general quality/PPL/KL/TVD.",
        "rows": qrows,
    }
    saved_q = read("hot-ffn/QUALITY_COMPARISON.json")
    for ours, saved in [("mean_teacher_logprob_delta", "mean_teacher_logprob_delta"), ("mean_common_L1", "mean_common_probability_L1"),
                        ("max_common_probability_delta", "max_common_probability_delta"), ("T1_time_reduction_percent", "T1_time_reduction_percent")]:
        require(math.isclose(quality[ours], saved_q[saved], rel_tol=1e-12, abs_tol=1e-12), "Quality recomputation differs: " + ours)
    quality["saved_summary_aggregates_match"] = True

    saved_numeric = read("cpu-cuda-arithmetic-audit-20261112/EXISTING_NUMERIC_EVIDENCE.json")
    numeric = []
    for row in saved_numeric["rows"]:
        relative = "execution/" + str(Path(row["source"]).parent.name) + "/stdout.log"
        raw = read(relative)
        require(digest(root / relative) == row["source_sha256"] and raw["pass"], "Arithmetic raw log/hash mismatch")
        gpu = next(x for x in raw["measurements"] if x["hot"] == 6)
        cpu_nmse = raw["CPU_error_vs_FP64_dequant_reference"][3]["nmse"]
        gpu_nmse = gpu["mixed_error_vs_FP64_reference"]["nmse"]
        require(cpu_nmse == row["cpu_down_error_vs_reference"]["nmse"] and gpu_nmse == row["gpu_down_error_vs_reference"]["nmse"], "Arithmetic summary differs")
        numeric.append({"layer": raw["layer"], "seed": raw["seed"], "cpu_nmse_reference": cpu_nmse, "gpu_nmse_reference": gpu_nmse,
                        "gpu_nmse_div_cpu_nmse": gpu_nmse / cpu_nmse, "gpu_vs_cpu_nmse": gpu["mixed_error_vs_CPU"]["nmse"],
                        "serial_concurrent_byte_identical": gpu["serial_concurrent_byte_identical"],
                        "scatter_byte_identical": gpu["GPU_hot_slots_byte_identical"]})
    backend = read("hot-ffn/BACKEND_VALIDATION.json")
    backend_summary = [{k: t[k] for k in ("layer", "cases", "worst_nmse_vs_CPU")} for t in backend["tests"]]

    training = read("training/20261112-075449/MANIFEST.json")
    heldout = read("HELDOUT_INPUTS.json")
    training_text = {p["synthetic_prompt"] for p in training["prompts"]}
    heldout_text = {p["synthetic_prompt"] for p in heldout["cases"]}
    fresh_text = {p["text"] for p in prompts["cases"]}
    census = read("training/20261112-075449/TRAINING_COUNTS.json")
    records = {r["name"]: r for r in census["records"]}
    frequency = read("plans/FREQUENCY_MANIFEST.json")
    prior = read("plans/VALUE_PRIOR_MANIFEST.json")
    capacity = read("hot-ffn/CAPACITY.json")
    plan_path = root / "plans/frequency19200.plan"
    sources["plans/frequency19200.plan"] = {"sha256": digest(plan_path), "size_bytes": plan_path.stat().st_size}
    require(frequency["training_sha256"] == digest(root / "training/20261112-075449/TRAINING_COUNTS.json"), "Frequency census hash differs")
    require(frequency["plan_sha256"] == digest(plan_path), "Frequency plan hash differs")
    chosen = []; denominator = 0
    for line in plan_path.read_text().splitlines()[1:]:
        fields = line.split(); name = fields[0]; ids = list(map(int, fields[6:]))
        require(len(ids) == int(fields[5]) == len(set(ids)), "Malformed frequency selection")
        record = records[name]; denominator += sum(record["counts"])
        chosen.extend((name, index, record["counts"][index]) for index in ids)
    selected_hits = sum(row[2] for row in chosen)
    require(len(chosen) == frequency["selected_projections"] == 19200, "Frequency slot count mismatch")
    require(math.isclose(selected_hits / denominator, frequency["in_sample_hit_rate"], abs_tol=1e-12), "Frequency hit fraction differs")
    design = {
        "training_prompts": len(training_text), "selection_heldout_prompts": len(heldout_text), "final_fresh_prompts": len(fresh_text),
        "exact_text_overlap_training_heldout": len(training_text & heldout_text),
        "exact_text_overlap_training_fresh": len(training_text & fresh_text),
        "exact_text_overlap_heldout_fresh": len(heldout_text & fresh_text),
        "training_names": [x["name"] for x in training["prompts"]],
        "frequency_plan_selected_projection_slots": len(chosen),
        "frequency_plan_training_projection_hits": selected_hits,
        "frequency_plan_training_projection_assignments": denominator,
        "frequency_plan_in_sample_hit_fraction": selected_hits / denominator,
        "frequency_payload_bytes": frequency["payload_bytes"],
        "value_prior_objective": prior["score"],
        "value_prior_report": read("HELDOUT_VALUE_PRIOR_R1.json")["aggregate_decode_gain_percent"],
        "hot14_capacity": next(row for row in capacity["budgets"] if row["budget_GiB"] == 14),
        "limitations": [
            "Different exact prompts do not imply a representative independent workload distribution; several themes overlap (code/SQL/Japanese library fiction).",
            "Eight purpose-written fresh prompts, each 128 tokens, are a local validation set, not long-context, long-conversation, concurrency or general quality coverage.",
            "CPU plan ranks projection assignment counts; GPU plan ranks training count/native bytes. Neither objective directly minimizes critical-path layer latency.",
            "The value/cost-weighted CPU plan was already tested once; its small difference is not independent proof of superiority.",
            "No per-request routing sequence is stored in aggregate training census or whole-pilot H histogram. Joint expert co-occurrence and per-case routes cannot be reconstructed.",
        ],
    }
    design["saved_input_usage"] = inventory_input_usage(root, read, input_sha, digest(root / "HELDOUT_INPUTS.json"))

    for name in ("compare_final_api.py", "fresh_probe.py", "analyze_heldout.py", "hot_quality.py", "analyze_hot_quality.py", "prepare_hot_plans.py"):
        path = root / name; sources[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    result = {
        "reviewed_commit": COMMIT,
        "scope": "Independent reaggregation of published saved measurements; no new hardware performance or inference execution.",
        "final": final, "hot14": hot, "quality": quality,
        "arithmetic_fixture_reaggregation": numeric,
        "separate_backend_validation_summary": backend_summary,
        "experimental_design": design,
        "analyzer_gate_probe": None if args.skip_gate_probe else gate_probe(root, final_runs, args.output.parent),
        "source_files": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2); output.write("\n")
    print(json.dumps({"output": str(args.output), "final_decode_gain_percent": final["decode_rate_gain_percent"],
                      "final_request_wall_token_rate_gain_percent": final["request_wall_token_rate_gain_percent"],
                      "hot14_decode_gain_percent": hot["decode_rate_gain_percent"],
                      "quality_greedy_matches": quality["greedy_agreement"], "quality_points": quality["points"],
                      "source_files": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
