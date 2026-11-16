#!/usr/bin/env python3
"""Validate and analyze the *proposed* mixed CPU operator protocol, schema 1.

This is an offline JSONL/binary-output analyzer, not a benchmark launcher.
It does not execute source, a model, an ELF, or an API. No observations are
discarded. Four overlay processes remain four process replicates; no CI,
bootstrap, p-value, or automatic performance-adoption decision is produced.

Usage: python3 analyze_mixed_run.py --run-root RUN --output NEW_REPORT.json
If --output is omitted, RUN/MIXED_ANALYSIS.json is created. Existing reports
are never overwritten. Invalid/partial evidence yields a failed JSON report
and exit code 1, without deleting or modifying input evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

COMMIT = "31b1292b2767618a28ef6e3f1c9c4450b1fc21be"
SCHEDULE = ("F", "E", "E", "F", "E", "F", "F", "E")
PAIRS = ((0, 1), (3, 2), (5, 4), (6, 7))  # (frozen index, overlay index)
CASES = ("C0", "C6", "Cplan")
CPU_SHA = {
    "F": "33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3",
    "E": "120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3",
}
INPUT_SHA = "9dc5705e04389d3d08d894ddf25ce675fa2ff3bed73eccc4ba9a156c8028ec93"
PLAN_SHA = "cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6"
# Pinned data independently prepared from the existing fixed fixture and final
# frequency19200 plan. These constants match generated_case/mixed_case.json.
ROUTES = (79,176,232,279,273,173,256,363,47,140,154,4,299,223,276,235,133,196,
          245,112,321,101,244,183,300,295,224,288,206,287,226,33,98,120,283,124,
          12,338,14,191,270,81,310,61,107,89,352,27)
CPLAN_IDS = (4,12,14,33,61,81,89,98,101,107,112,120,124,133,140,154,173,176,
             183,191,196,223,232,245,256,270,273,276,279,283,287,288,299,300,
             310,321,338,352,363)
EXPERT_BYTES = 6635520
OUTPUT_VALUES_PER_GROUP = 13824
OUTPUT_BYTES_PER_CASE = 8 * OUTPUT_VALUES_PER_GROUP * 4


class EvidenceError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise EvidenceError(message)


def integer(value):
    return type(value) is int


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text, where):
    def bad_constant(value):
        raise EvidenceError(where + ": nonfinite JSON constant " + value)
    try:
        result = json.loads(text, object_pairs_hook=unique_object, parse_constant=bad_constant)
    except (ValueError, RecursionError) as error:
        raise EvidenceError(where + ": invalid JSON: " + str(error)) from error
    def finite(value):
        if isinstance(value, float):
            check(math.isfinite(value), where + ": nonfinite numeric value")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    finite(result)
    check(isinstance(result, dict), where + ": object required")
    return result


def sample_description(values):
    check(bool(values) and all(math.isfinite(x) for x in values), "Empty or invalid statistics")
    return {"n": len(values), "median": statistics.median(values),
            "arithmetic_mean": statistics.mean(values), "min": min(values), "max": max(values)}


def ratio_description(values):
    result = sample_description(values)
    result.update(values=values, count_above_one=sum(x > 1 for x in values))
    return result


def selected_for(case):
    return [] if case == "C0" else list(ROUTES) if case == "C6" else list(CPLAN_IDS)


def masks_for(selected):
    present = set(selected)
    return [sum(1 << slot for slot, expert in enumerate(ROUTES[6 * group:6 * group + 6])
                if expert in present) for group in range(8)]


def case_order_for(process_index):
    rotation = (process_index // 2) % 3
    return list(CASES[rotation:] + CASES[:rotation])


def logical_order(block):
    reverse = (block % 8 + block // 8) % 2
    return ("S", "O", "O", "S") if reverse else ("O", "S", "S", "O")


def protocol():
    return {"schema": 1, "commit": COMMIT, "schedule": list(SCHEDULE), "pairs_frozen_overlay": [list(p) for p in PAIRS],
            "cases": list(CASES), "tensor": "blk.0.ffn_up_exps.weight", "blocks": 16, "arms": 4,
            "graphs_per_chunk": 32, "routes": list(ROUTES), "inputs_sha256": INPUT_SHA,
            "plan_sha256": PLAN_SHA, "Cplan_selected_ids": list(CPLAN_IDS),
            "Cplan_masks": masks_for(CPLAN_IDS), "Cplan_cached_of_six": [x.bit_count() for x in masks_for(CPLAN_IDS)]}


def load_process(run_root, index):
    folder = (run_root / f"process-{index}").resolve()
    check(folder.is_relative_to(run_root) and folder.is_dir(), f"Missing/unsafe process directory {index}")
    status_path = folder / "STATUS.json"
    status = parse_json(status_path.read_text(encoding="utf-8"), str(status_path))
    label = SCHEDULE[index]
    check(integer(status.get("schema")) and status["schema"] == 1, f"process {index}: STATUS schema")
    check(integer(status.get("exit_code")) and status["exit_code"] == 0 and "error" in status and status["error"] is None
          and status.get("passed") is True and status.get("state") == "completed",
          f"process {index}: unsuccessful/partial STATUS")
    check(status.get("process_index") == index and integer(status.get("process_index")) and status.get("library_label") == label,
          f"process {index}: STATUS identity")
    start = status.get("started_monotonic_ns"); end = status.get("ended_monotonic_ns")
    check(integer(start) and integer(end) and 0 <= start < end, f"process {index}: invalid STATUS clock")
    identity = status.get("identity")
    check(isinstance(identity, dict), f"process {index}: missing identity")
    for field in ("cpu_library_sha256", "base_library_sha256", "harness_sha256", "inputs_sha256", "plan_sha256",
                  "case_spec_sha256", "case_header_sha256", "launcher_sha256"):
        check(valid_sha(identity.get(field)), f"process {index}: invalid {field}")
    for field in ("cpu_library_path", "base_library_path"):
        check(isinstance(identity.get(field), str) and identity[field], f"process {index}: invalid {field}")
    shard = identity.get("shard_stat")
    check(isinstance(shard, dict) and isinstance(shard.get("path"), str) and shard["path"]
          and all(integer(shard.get(k)) for k in ("device", "inode", "bytes", "mtime_ns"))
          and shard["bytes"] > 0, f"process {index}: invalid shard_stat")
    extra_dirs = identity.get("extra_library_dirs")
    check(isinstance(extra_dirs, list) and all(isinstance(p, str) and p for p in extra_dirs),
          f"process {index}: invalid extra_library_dirs")
    check(identity["cpu_library_sha256"] == CPU_SHA[label], f"process {index}: unexpected CPU library hash")
    check(identity["inputs_sha256"] == INPUT_SHA and identity["plan_sha256"] == PLAN_SHA,
          f"process {index}: input/plan differs from pinned fixture")
    files = status.get("files")
    check(isinstance(files, dict), f"process {index}: missing file receipt")
    verified = {}; payloads = {}
    for name in ("raw.jsonl", "expected-C0.bin", "expected-C6.bin", "expected-Cplan.bin"):
        info = files.get(name)
        check(isinstance(info, dict) and integer(info.get("bytes")) and info["bytes"] >= 0 and valid_sha(info.get("sha256")),
              f"process {index}: invalid file receipt {name}")
        path = (folder / name).resolve()
        check(path.is_relative_to(folder), f"process {index}: file path escape {name}")
        check(path.stat().st_size <= (8 * 1024 * 1024 if name == "raw.jsonl" else OUTPUT_BYTES_PER_CASE),
              f"process {index}: unexpected file size {name}")
        data = path.read_bytes()
        check(len(data) == info["bytes"] and digest(data) == info["sha256"], f"process {index}: file hash/length mismatch {name}")
        if name != "raw.jsonl":
            check(len(data) == OUTPUT_BYTES_PER_CASE, f"process {index}: output shape/byte length {name}")
            check(all((word & 0x7f800000) != 0x7f800000 for (word,) in struct.iter_unpack("<I", data)),
                  f"process {index}: nonfinite expected output {name}")
        verified[name] = {"bytes": len(data), "sha256": digest(data)}; payloads[name] = data
    raw_lines = [line for line in payloads["raw.jsonl"].decode("utf-8").splitlines() if line.strip()]
    records = [parse_json(line, f"process {index} raw line {i + 1}") for i, line in enumerate(raw_lines)]
    completion = next((r for r in reversed(records) if r.get("kind") == "mixed_complete"), None)
    check(completion is not None and completion.get("passed") is True,
          f"process {index}: missing/failed mixed_complete; evidence retained")
    check(len(records) == 197, f"process {index}: expected 197 JSONL records, got {len(records)}")
    meta = records[0]
    check(meta.get("kind") == "mixed_meta" and integer(meta.get("schema")) and meta["schema"] == 1, f"process {index}: meta schema")
    for field, wanted in (("process_index", index), ("threads", 12), ("blocks", 16), ("repeats", 32)):
        check(integer(meta.get(field)) and meta[field] == wanted, f"process {index}: meta {field}")
    check(meta.get("library_label") == label and meta.get("commit") == COMMIT
          and meta.get("tensor") == "blk.0.ffn_up_exps.weight", f"process {index}: meta identity")
    check(meta.get("routes") == list(ROUTES) and all(integer(x) for x in meta["routes"]), f"process {index}: routes differ from pinned fixture")
    check(meta.get("case_order") == case_order_for(index), f"process {index}: case order rotation")
    check(meta.get("inputs_sha256") == INPUT_SHA and meta.get("plan_sha256") == PLAN_SHA and meta.get("ggml_numa") is False,
          f"process {index}: meta hashes/NUMA")
    check(meta.get("clock") == "CLOCK_MONOTONIC" and meta.get("hashes_verified_by_harness") is False,
          f"process {index}: clock/hash-verification contract")
    for provider in ("cpu_provider", "base_provider", "cache_provider"):
        check(isinstance(meta.get(provider), str) and meta[provider], f"process {index}: missing {provider}")
    check(meta["cache_provider"] == meta["cpu_provider"], f"process {index}: cache provider differs from CPU provider")
    check(meta.get("mode_provider") == (meta["cpu_provider"] if label == "E" else ""), f"process {index}: mode provider identity")
    check(meta["cpu_provider"] == identity["cpu_library_path"] and meta["base_provider"] == identity["base_library_path"],
          f"process {index}: provider path differs from launcher identity")
    cases = {}; cursor = 1; previous_end = start
    for case in case_order_for(index):
        header = records[cursor]; cursor += 1
        check(header.get("kind") == "mixed_case" and header.get("case") == case, f"process {index}: case header/order")
        selected = header.get("selected_ids")
        check(isinstance(selected, list) and all(integer(x) for x in selected)
              and len(selected) == len(set(selected)) and set(selected) == set(selected_for(case)),
              f"process {index} {case}: selected IDs differ from pinned membership")
        masks = masks_for(selected)
        check(header.get("masks") == masks and all(integer(x) for x in header["masks"]), f"process {index} {case}: masks")
        check(integer(header.get("cache_bytes")) and header["cache_bytes"] == len(selected) * EXPERT_BYTES,
              f"process {index} {case}: cache bytes")
        pins = header.get("pins")
        check(isinstance(pins, list) and all(integer(x) for x in pins) and sorted(pins) == list(range(12)), f"process {index} {case}: pins")
        check(header.get("expected_file") == f"expected-{case}.bin", f"process {index} {case}: expected filename")
        for field, wanted in (("expected_groups", 8), ("expected_values_per_group", OUTPUT_VALUES_PER_GROUP),
                              ("warmup_arm_calls_per_group", 4), ("bit_mismatches", 0)):
            check(integer(header.get(field)) and header[field] == wanted, f"process {index} {case}: {field}")
        check(header.get("dispatch_checked") is (label == "E"), f"process {index} {case}: dispatch check missing/unexpected")
        samples = []
        for block in range(16):
            group = block % 8; order = logical_order(block)
            for arm in range(4):
                chunk = records[cursor]; cursor += 1
                context = f"process {index} {case} block {block} arm {arm}"
                check(chunk.get("kind") == "mixed_chunk" and chunk.get("case") == case, context + ": chunk type/case")
                for field, wanted in (("block", block), ("route_group", group), ("arm", arm), ("repeats", 32),
                                      ("cached_of_six", masks[group].bit_count()), ("hit_mask", masks[group]), ("bit_mismatches", 0)):
                    check(integer(chunk.get(field)) and chunk[field] == wanted, context + ": " + field)
                check(chunk.get("ids") == list(ROUTES[6 * group:6 * group + 6]) and all(integer(x) for x in chunk["ids"]), context + ": route IDs")
                check(chunk.get("logical_mode") == order[arm], context + ": logical mode/order")
                wanted_mode = 1 if label == "E" and order[arm] == "S" else 0
                check(integer(chunk.get("actual_mode")) and chunk["actual_mode"] == wanted_mode, context + ": actual mode")
                begin = chunk.get("begin_ns"); stop = chunk.get("end_ns"); elapsed = chunk.get("elapsed_ns")
                check(all(integer(v) for v in (begin, stop, elapsed)) and start <= previous_end <= begin < stop <= end,
                      context + ": invalid/overlapping/out-of-process clocks")
                check(elapsed == stop - begin and elapsed > 0, context + ": elapsed mismatch")
                mean_ms = chunk.get("ms_per_graph")
                check(type(mean_ms) in (int, float) and math.isfinite(mean_ms) and mean_ms > 0
                      and math.isclose(mean_ms, elapsed / (32 * 1e6), rel_tol=1e-12, abs_tol=1e-12), context + ": ms/graph mismatch")
                previous_end = stop; samples.append(chunk)
        cases[case] = {"header": header, "samples": samples}
    check(cursor == len(records) - 1 and records[cursor] == completion, f"process {index}: trailing/duplicate completion")
    for field, wanted in (("chunks", 192), ("timed_graph_calls", 6144), ("outputs_bytes", 3 * OUTPUT_BYTES_PER_CASE)):
        check(integer(completion.get(field)) and completion[field] == wanted, f"process {index}: completion {field}")
    check(completion.get("expected_output_scope") == "mode0 once per case/group; final output of each timed chunk compared in memory",
          f"process {index}: output scope missing or changed")
    return {"index": index, "label": label, "status": status, "meta": meta, "cases": cases,
            "verified_files": verified, "payloads": payloads}


def block_sums(process, case, logical=None):
    return [sum(s["elapsed_ns"] for s in process["cases"][case]["samples"] if s["block"] == block
                and (logical is None or s["logical_mode"] == logical)) for block in range(16)]


def within_process(process, case):
    o = block_sums(process, case, "O"); s = block_sums(process, case, "S")
    ratios = [a / b for a, b in zip(o, s)]
    group_rows = []
    for group in range(8):
        blocks = (group, group + 8)
        group_rows.append({"route_group": group, "cached_of_six": process["cases"][case]["header"]["masks"][group].bit_count(),
                           "paired_ratios": [ratios[b] for b in blocks],
                           "ratio_of_total_times": sum(o[b] for b in blocks) / sum(s[b] for b in blocks),
                           "timed_graph_calls_each_logical_mode": 128})
    total_ratio = sum(o) / sum(s)
    return {"process_index": process["index"], "library_label": process["label"], "case": case,
            "scope": "Within-process actual off/shared" if process["label"] == "E" else "Placebo logical O/S: all actual modes are 0",
            "block_paired_ratios": ratio_description(ratios), "ratio_of_total_times": total_ratio,
            "total_time_change_S_vs_O_percent": 100 * (1 / total_ratio - 1),
            "logical_O_total_ns": sum(o), "logical_S_total_ns": sum(s), "timed_graph_calls_each_logical_mode": 1024,
            "logical_O_mean_ms_per_graph": sum(o) / (1024 * 1e6), "logical_S_mean_ms_per_graph": sum(s) / (1024 * 1e6),
            "logical_O_chunk_ms": sample_description([x["elapsed_ns"] / (32 * 1e6) for x in process["cases"][case]["samples"] if x["logical_mode"] == "O"]),
            "logical_S_chunk_ms": sample_description([x["elapsed_ns"] / (32 * 1e6) for x in process["cases"][case]["samples"] if x["logical_mode"] == "S"]),
            "groups": group_rows}


def analyze_run(run_root):
    root = Path(run_root).resolve()
    check(root.is_dir(), "Run root is missing")
    process_dirs = {p.name for p in root.iterdir() if p.name.startswith("process-")}
    check(process_dirs == {f"process-{i}" for i in range(8)}, "Expected exactly process-0 through process-7")
    processes = [load_process(root, i) for i in range(8)]
    for i in range(1, 8):
        check(processes[i - 1]["status"]["ended_monotonic_ns"] <= processes[i]["status"]["started_monotonic_ns"],
              "Outer processes overlap or are out of scheduled order")
    for field in ("base_library_sha256", "harness_sha256", "inputs_sha256", "plan_sha256",
                  "case_spec_sha256", "case_header_sha256", "launcher_sha256"):
        check(len({p["status"]["identity"][field] for p in processes}) == 1, "Common identity differs: " + field)
    for field in ("shard_stat", "extra_library_dirs"):
        values = {json.dumps(p["status"]["identity"][field], sort_keys=True, separators=(",", ":"), allow_nan=False) for p in processes}
        check(len(values) == 1, "Common identity differs: " + field)
    check(len({p["meta"]["base_provider"] for p in processes}) == 1, "Base provider path differs")
    for label in ("F", "E"):
        check(len({p["meta"]["cpu_provider"] for p in processes if p["label"] == label}) == 1, "CPU provider path changes within library label")
    output_checks = {}
    for case in CASES:
        first = processes[0]["payloads"][f"expected-{case}.bin"]
        check(all(p["payloads"][f"expected-{case}.bin"] == first for p in processes), "Saved expected bytes differ across processes: " + case)
        check(all(p["cases"][case]["header"]["selected_ids"] == processes[0]["cases"][case]["header"]["selected_ids"] for p in processes),
              "Cache packing order differs across processes: " + case)
        output_checks[case] = {"bytes_each_process": len(first), "sha256": digest(first), "all_eight_processes_byte_equal": True}
    primary = [within_process(p, case) for p in processes if p["label"] == "E" for case in CASES]
    placebo = [within_process(p, case) for p in processes if p["label"] == "F" for case in CASES]
    cross = []
    for pair_index, (fi, ei) in enumerate(PAIRS):
        frozen = processes[fi]; overlay = processes[ei]
        for case in CASES:
            f = block_sums(frozen, case)
            for logical in ("O", "S"):
                e = block_sums(overlay, case, logical)
                # F has FOUR arms and E logical mode has TWO, each 32 graphs.
                # Every comparison therefore uses F/2, never raw F/E sums.
                ratios = [a / (2 * b) for a, b in zip(f, e)]
                ratio_total = sum(f) / (2 * sum(e))
                cross.append({"pair_index": pair_index, "frozen_process_index": fi, "overlay_process_index": ei,
                              "case": case, "overlay_logical_mode": logical, "block_normalized_ratios": ratio_description(ratios),
                              "ratio_of_normalized_total_times": ratio_total,
                              "time_change_E_vs_F_percent": 100 * (1 / ratio_total - 1),
                              "frozen_raw_total_ns": sum(f), "overlay_logical_total_ns": sum(e),
                              "frozen_raw_graph_calls": 2048, "overlay_logical_graph_calls": 1024,
                              "normalization": "F raw total / 2 versus E logical total; equal 1024 graph calls",
                              "frozen_mean_ms_per_graph": sum(f) / (2048 * 1e6), "overlay_mean_ms_per_graph": sum(e) / (1024 * 1e6),
                              "groups": [{"route_group": g, "cached_of_six": frozen["cases"][case]["header"]["masks"][g].bit_count(),
                                          "ratio_of_normalized_total_times": (f[g] + f[g + 8]) / (2 * (e[g] + e[g + 8]))} for g in range(8)]})
    summaries = {"overlay_within_process": [], "frozen_placebo": [], "cross_library": []}
    for case in CASES:
        for name, rows in (("overlay_within_process", primary), ("frozen_placebo", placebo)):
            selected = [r for r in rows if r["case"] == case]
            check(len(selected) == 4, "Expected four process-level statistics")
            summaries[name].append({"case": case, "process_indices": [r["process_index"] for r in selected],
                                    "process_paired_medians": [r["block_paired_ratios"]["median"] for r in selected],
                                    "pooled_block_paired_median": statistics.median(v for r in selected for v in r["block_paired_ratios"]["values"]),
                                    "pooled_block_ratio_count": 64,
                                    "process_total_time_ratios": [r["ratio_of_total_times"] for r in selected],
                                    "median_of_four_process_total_time_ratios": statistics.median(r["ratio_of_total_times"] for r in selected),
                                    "pooled_equal_work_total_ratio": sum(r["logical_O_total_ns"] for r in selected) / sum(r["logical_S_total_ns"] for r in selected),
                                    "process_replicate_count": 4, "scope": "Descriptive aggregation; no inference from 64 block ratios as independent processes."})
        for logical in ("O", "S"):
            selected = [r for r in cross if r["case"] == case and r["overlay_logical_mode"] == logical]
            summaries["cross_library"].append({"case": case, "overlay_logical_mode": logical, "process_pairs": 4,
                                                "pair_paired_medians": [r["block_normalized_ratios"]["median"] for r in selected],
                                                "pooled_block_paired_median": statistics.median(v for r in selected for v in r["block_normalized_ratios"]["values"]),
                                                "pooled_block_ratio_count": 64,
                                                "pair_total_time_ratios": [r["ratio_of_normalized_total_times"] for r in selected],
                                                "pooled_normalized_total_ratio": sum(r["frozen_raw_total_ns"] for r in selected) / (2 * sum(r["overlay_logical_total_ns"] for r in selected))})
    synthetic_flags = [p["status"].get("synthetic", False) is True or p["meta"].get("synthetic", False) is True for p in processes]
    return {"analysis_schema": 1, "passed": True, "analysis_complete": True,
            "performance_adoption_decision": "not made", "input_scope": "SYNTHETIC TEST DATA; NOT PERFORMANCE EVIDENCE" if any(synthetic_flags) else "User-supplied protocol records; not independently remeasured",
            "run_root": str(root), "protocol": protocol(),
            "checks": {"processes": 8, "overlay_processes": 4, "frozen_processes": 4, "cache_cases": 3,
                       "chunks": 1536, "timed_graph_calls": 49152, "required_files_verified": 32,
                       "all_outputs_equal_within_each_case": True, "all_observations_used_no_exclusions": True,
                       "input_raw_files_unmodified": True,
                       "no_confidence_intervals_or_bootstrap": True},
            "expected_output_checks": output_checks,
            "process_receipts": [{"process_index": p["index"], "library_label": p["label"], "identity": p["status"]["identity"],
                                  "started_monotonic_ns": p["status"]["started_monotonic_ns"], "ended_monotonic_ns": p["status"]["ended_monotonic_ns"],
                                  "providers": {k: p["meta"][k] for k in ("cpu_provider", "base_provider", "cache_provider", "mode_provider")},
                                  "verified_files": p["verified_files"]} for p in processes],
            "primary_overlay_within_process": primary, "frozen_logical_placebo": placebo,
            "cross_library_pairs": cross, "descriptive_summaries": summaries,
            "limitations": ["Four overlay processes and four frozen processes, not 64 independent block/request replicates.",
                            "No outlier removal or automatic performance gate. Median, mean-of-ratios and total-time ratios are distinct.",
                            "F has twice the timed graph count of one E logical mode; cross-library statistics normalize this explicitly.",
                            "Cplan is the fixed fixture's plan intersection, not production hit rate or routed activation distribution.",
                            "Saved bytes are mode0 expected outputs; final timed-chunk output equality is reported by the producer.",
                            "Recorded library hashes and provider paths are validated for consistency, not verified by loading a library.",
                            "Synthetic fixtures, if marked, are not performance measurements."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_root / "MIXED_ANALYSIS.json"
    try:
        result = analyze_run(args.run_root)
    except Exception as error:
        result = {"analysis_schema": 1, "passed": False, "analysis_complete": False,
                  "performance_adoption_decision": "not made", "run_root": str(args.run_root.resolve()),
                  "failure": {"type": type(error).__name__, "message": str(error)},
                  "evidence_retained": True, "scope": "Offline evidence validation failed; no input files deleted or modified."}
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    except Exception as error:
        print(json.dumps({"passed": False, "analysis_complete": result.get("analysis_complete", False),
                          "failure": result.get("failure"), "report_write_error": str(error),
                          "evidence_retained": True}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"passed": result["passed"], "output": str(output),
                      "performance_adoption_decision": "not made", "failure": result.get("failure")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
