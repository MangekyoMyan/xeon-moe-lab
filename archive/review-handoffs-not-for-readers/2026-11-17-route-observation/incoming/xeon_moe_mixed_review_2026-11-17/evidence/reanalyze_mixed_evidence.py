#!/usr/bin/env python3
"""Independent, read-only reaggregation of the ff61680 mixed-operator export.

Only standard-library data parsing is used. No archived Python module is imported,
and no model, library, executable, API, build, or performance benchmark is run.
All 1,536 supplied chunks are retained. Restored expected bytes and new reports
are written only outside the repository; existing outputs are never overwritten.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import struct
import sys

COMMIT = "ff6168094858141aeeebf45abc5c309882fe5216"
BASE_COMMIT = "31b1292b2767618a28ef6e3f1c9c4450b1fc21be"
HANDOFF = "archive/review-handoffs-not-for-readers/2026-11-17-mixed-operator"
INCOMING = "incoming/xeon_moe_abs_next_2026-11-16"
LABELS = "FEEFEFFE"
PAIRS = ((0, 1), (3, 2), (5, 4), (6, 7))
CASES = ("C0", "C6", "Cplan")
CPU_HASHES = {
    "F": "33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3",
    "E": "120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3",
}
BASE_HASH = "2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558"
INPUT_HASH = "9dc5705e04389d3d08d894ddf25ce675fa2ff3bed73eccc4ba9a156c8028ec93"
PLAN_HASH = "cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6"
EXPECTED_HASH = "6d3ffb5c2345af061c67ac3d8bdd4b23043b76308164136648f22f41603bf110"
EXPECTED_BYTES = 8 * 13824 * 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def load_json(path):
    require(path.stat().st_size <= 16 * 1024**2, "oversize JSON: " + str(path))
    return parse_json(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha(data)}


def safe_path(root, name):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in name,
            "unsafe relative path: " + name)
    target = root.joinpath(*rel.parts)
    require(target.resolve().is_relative_to(root.resolve()), "path escapes root")
    return target


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def valid_sha(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def near(a, b):
    return math.isclose(a, b, rel_tol=2e-13, abs_tol=2e-13)


def verify_manifest(root):
    manifest = load_json(root / "FILES_SHA256.json")
    require(isinstance(manifest, dict), "manifest object required")
    for name, record in manifest.items():
        path = safe_path(root, name)
        require(path.is_file() and not path.is_symlink(), "missing or symlink: " + name)
        require(file_record(path) == record, "manifest mismatch: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(manifest) | {"FILES_SHA256.json"}, "manifest file-set mismatch")
    return {"entries": len(manifest), "all_bytes_and_hashes_match": True,
            "file_set_exact": True, "manifest": file_record(root / "FILES_SHA256.json"),
            "files": manifest}


def same_return(incoming, prior):
    a = {p.relative_to(incoming).as_posix() for p in incoming.rglob("*") if p.is_file()}
    b = {p.relative_to(prior).as_posix() for p in prior.rglob("*") if p.is_file()}
    require(a == b, "incoming/prior return file sets differ")
    for name in sorted(a):
        require(safe_path(incoming, name).read_bytes() == safe_path(prior, name).read_bytes(),
                "incoming differs from previous local return: " + name)
    return {"files": len(a), "all_exact_bytes_match": True,
            "prior_return": str(prior), "filenames": sorted(a)}


def restore_expected(root, out):
    index = load_json(root / "correctness/INDEX.json")
    require(len(index) == 1 and EXPECTED_HASH in index, "unexpected expected-array inventory")
    targets = {}
    blob_records = []
    for digest, entry in index.items():
        require(valid_sha(digest) and entry["bytes"] == EXPECTED_BYTES, "invalid expected-array declaration")
        compressed = safe_path(root, entry["gzip"])
        with gzip.open(compressed, "rb") as stream:
            data = stream.read(EXPECTED_BYTES + 1)
            require(len(data) == EXPECTED_BYTES and stream.read(1) == b"", "gzip expansion exceeds bound")
        require(sha(data) == digest, "expanded expected SHA mismatch")
        values = [x[0] for x in struct.iter_unpack("<f", data)]
        require(len(values) == 110592 and all(math.isfinite(x) for x in values), "invalid expected floats")
        for name in entry["targets"]:
            require(name not in targets and re.fullmatch(r"mixed-run/process-[0-7]/expected-(C0|C6|Cplan)\.bin", name),
                    "duplicate/invalid restoration target")
            destination = safe_path(out / "restored", name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
            targets[name] = {"bytes": len(data), "sha256": digest}
        blob_records.append({"compressed_file": entry["gzip"], **file_record(compressed),
                             "expanded_bytes": len(data), "expanded_sha256": digest,
                             "finite_float_count": len(values), "min": min(values), "max": max(values),
                             "target_count": len(entry["targets"])})
    required = {f"mixed-run/process-{i}/expected-{case}.bin" for i in range(8) for case in CASES}
    require(set(targets) == required, "expected output targets do not cover all 24")
    return {"targets": targets, "blobs": blob_records,
            "all_24_files_exactly_equal_including_across_cache_cases": True,
            "scope": "Saved mode0 expected arrays only, not independent saved candidate outputs."}


def stats(values):
    require(bool(values), "empty statistics")
    return {"count": len(values), "median": statistics.median(values),
            "arithmetic_mean": statistics.fmean(values), "minimum": min(values),
            "maximum": max(values), "above_one": sum(x > 1 for x in values), "values": values}


def delta_percent(ratio):
    return 100.0 * (1.0 / ratio - 1.0)


def per_case(rows, index, case):
    blocks = []
    origin = rows[0]["begin_ns"]
    for block in range(16):
        arms = [r for r in rows if r["block"] == block]
        o = sum(r["elapsed_ns"] for r in arms if r["logical_mode"] == "O")
        s = sum(r["elapsed_ns"] for r in arms if r["logical_mode"] == "S")
        blocks.append({"block": block, "route_group": block % 8,
                       "cached_of_six": arms[0]["cached_of_six"],
                       "order": "".join(r["logical_mode"] for r in arms),
                       "begin_ns": arms[0]["begin_ns"], "end_ns": arms[-1]["end_ns"],
                       "begin_relative_to_case_s": (arms[0]["begin_ns"] - origin) / 1e9,
                       "O_ns": o, "S_ns": s, "ratio_O_over_S": o / s,
                       "all_arm_mean_ms_per_graph": (o + s) / (128 * 1e6),
                       "all_arms": arms})
    o = sum(b["O_ns"] for b in blocks)
    s = sum(b["S_ns"] for b in blocks)
    return {"process_index": index, "library_label": LABELS[index], "case": case,
            "O_ns": o, "S_ns": s, "total_ns": o + s, "graphs_per_logical_mode": 1024,
            "ratio_of_sums": o / s, "candidate_time_change_percent": delta_percent(o / s),
            "O_mean_ms_per_graph": o / (1024 * 1e6), "S_mean_ms_per_graph": s / (1024 * 1e6),
            "block_ratios": stats([b["ratio_O_over_S"] for b in blocks]),
            "begin_ns": rows[0]["begin_ns"], "end_ns": rows[-1]["end_ns"],
            "span_ns": rows[-1]["end_ns"] - rows[0]["begin_ns"],
            "non_timed_gap_ns": rows[-1]["end_ns"] - rows[0]["begin_ns"] - o - s,
            "blocks": blocks}


def load_process(root, index, case_spec, spec, build, restored):
    folder = root / f"mixed-run/process-{index}"
    status = load_json(folder / "STATUS.json")
    label = LABELS[index]
    require(status["schema"] == 1 and status["process_index"] == index and status["library_label"] == label,
            "STATUS identity mismatch")
    require(status["passed"] is True and status["state"] == "completed" and status["exit_code"] == 0 and
            "error" in status and status["error"] is None, "unsuccessful process")
    ident = status["identity"]
    expected_identity = {
        "cpu_library_sha256": CPU_HASHES[label], "base_library_sha256": BASE_HASH,
        "harness_sha256": build["harness_sha256"], "inputs_sha256": INPUT_HASH, "plan_sha256": PLAN_HASH,
        "case_spec_sha256": file_record(root / INCOMING / "next_experiment/case/mixed_case.json")["sha256"],
        "case_header_sha256": build["case_header_sha256"],
        "launcher_sha256": file_record(root / INCOMING / "next_experiment/run_one_mixed.py")["sha256"],
    }
    for key, value in expected_identity.items():
        require(valid_sha(ident[key]) and ident[key] == value, "STATUS hash mismatch: " + key)
    for name in ("raw.jsonl", "maps.txt", "stdout.log", "stderr.log"):
        recorded = status["files"][name] if name == "raw.jsonl" else status["auxiliary_files"][name]
        require(recorded == file_record(folder / name), "STATUS file hash mismatch: " + name)
    require((folder / "stderr.log").read_bytes() == b"", "stderr is nonempty")
    require(parse_json((folder / "stdout.log").read_text())["passed"] is True, "stdout success missing")
    for case in CASES:
        name = f"expected-{case}.bin"
        require(status["files"][name] == restored[f"mixed-run/process-{index}/{name}"], "expected STATUS mismatch")
    start, end = status["started_monotonic_ns"], status["ended_monotonic_ns"]
    require(integer(start, 1) and integer(end, 1) and end > start, "invalid STATUS clock interval")
    rows = []
    for number, line in enumerate((folder / "raw.jsonl").read_text().splitlines(), 1):
        row = parse_json(line)
        require(isinstance(row, dict), "raw row must be object")
        row["source_line"] = number
        rows.append(row)
    require(len(rows) == 197, "raw record count mismatch")
    meta, complete = rows[0], rows[-1]
    require(meta["kind"] == "mixed_meta" and complete["kind"] == "mixed_complete", "raw endpoints wrong")
    expected_meta = {
        "schema": 1, "process_index": index, "library_label": label, "commit": BASE_COMMIT,
        "tensor": "blk.0.ffn_up_exps.weight", "threads": 12, "blocks": 16, "repeats": 32,
        "routes": case_spec["routes"], "case_order": spec["schedule"][index]["case_order"],
        "inputs_sha256": INPUT_HASH, "plan_sha256": PLAN_HASH,
        "cpu_provider": ident["cpu_library_path"], "cache_provider": ident["cpu_library_path"],
        "base_provider": ident["base_library_path"],
        "mode_provider": ident["cpu_library_path"] if label == "E" else "",
        "shard": ident["shard_stat"]["path"], "clock": "CLOCK_MONOTONIC",
    }
    for key, value in expected_meta.items():
        require(meta[key] == value, "raw metadata mismatch: " + key)
    require(meta["ggml_numa"] is False and meta["hashes_verified_by_harness"] is False, "raw scope mismatch")
    expected_complete = {"passed": True, "chunks": 192, "timed_graph_calls": 6144,
                         "outputs_bytes": 3 * EXPECTED_BYTES,
                         "expected_output_scope": "mode0 once per case/group; final output of each timed chunk compared in memory"}
    require(all(complete[k] == v for k, v in expected_complete.items()) and complete["passed"] is True and
            complete["error"] is None, "incomplete or failed raw")
    offset = 1
    previous_end = start
    case_summaries = {}
    for case in meta["case_order"]:
        header = rows[offset]
        require(header["kind"] == "mixed_case" and header["case"] == case, "case ordering mismatch")
        selected = [] if case == "C0" else case_spec["routes"] if case == "C6" else case_spec["cache_ids"]
        masks = [sum(1 << j for j in range(6) if case_spec["routes"][6*g+j] in selected) for g in range(8)]
        require(header["selected_ids"] == selected and header["masks"] == masks and
                header["cache_bytes"] == len(selected) * 6635520, "actual cache membership mismatch")
        require(sorted(header["pins"]) == list(range(12)) and all(type(x) is int for x in header["pins"]), "pin sample mismatch")
        require(header["dispatch_checked"] is (label == "E") and header["bit_mismatches"] == 0 and
                header["expected_file"] == f"expected-{case}.bin" and header["expected_groups"] == 8 and
                header["expected_values_per_group"] == 13824 and header["warmup_arm_calls_per_group"] == 4,
                "case checks or expected scope mismatch")
        chunks = rows[offset + 1:offset + 65]
        for k, chunk in enumerate(chunks):
            block, arm = divmod(k, 4)
            group = block % 8
            reverse = (group + block // 8) % 2
            logical = "S" if ((arm in (1, 2)) != bool(reverse)) else "O"
            expected = {"kind": "mixed_chunk", "case": case, "block": block, "route_group": group,
                        "arm": arm, "logical_mode": logical, "actual_mode": int(label == "E" and logical == "S"),
                        "cached_of_six": masks[group].bit_count(), "hit_mask": masks[group],
                        "ids": case_spec["routes"][group*6:group*6+6], "repeats": 32, "bit_mismatches": 0}
            require(all(chunk[key] == value for key, value in expected.items()), "chunk protocol differs")
            for key in ("block", "route_group", "arm", "actual_mode", "cached_of_six", "hit_mask", "repeats", "bit_mismatches"):
                require(integer(chunk[key]), "noninteger chunk field: " + key)
            begin, finish, elapsed = chunk["begin_ns"], chunk["end_ns"], chunk["elapsed_ns"]
            require(all(integer(v, 1) for v in (begin, finish, elapsed)) and finish - begin == elapsed,
                    "chunk duration mismatch")
            require(previous_end <= begin < finish <= end, "chunk overlap or outside STATUS")
            require(near(chunk["ms_per_graph"], elapsed / 32e6), "derived ms/graph mismatch")
            previous_end = finish
        summary = per_case(chunks, index, case)
        summary["header"] = header
        case_summaries[case] = summary
        offset += 65
    require(offset == 196, "record traversal did not consume all cases")
    maps = (folder / "maps.txt").read_text().splitlines()
    map_paths = sorted({line.split(None, 5)[5] for line in maps if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/")})
    for marker, expected in (("libggml-cpu.so", ident["cpu_library_path"]), ("libggml-base.so", ident["base_library_path"])):
        require([p for p in map_paths if marker in p] == [expected], "mapped provider differs: " + marker)
    require(ident["harness_path"] in map_paths, "harness not found in maps")
    require(not any("cuda" in p.lower() for p in map_paths), "unexpected CUDA mapping")
    host = status["host_preflight"]
    require(host["pool_cpus"] == list(range(12)) and host["physical_cores"] == 12 and host["packages"] == [0, 1], "preflight topology mismatch")
    require(len({tuple(host["target_topology"][str(i)]) for i in range(12)}) == 12, "pool aliases physical cores")
    require(host["reserve_bytes"] == 64 * 1024**3 and host["mem_available_bytes"] >= host["reserve_bytes"] + host["cache_budget_bytes"], "preflight reserve mismatch")
    argv = status["argv"]
    require(argv[1:4] == ["--physcpubind=0-23", "--interleave=0,1", ident["harness_path"]] and argv[-3:] == [str(index), ident["cpu_library_path"], ident["base_library_path"]], "launch arguments differ")
    return {"process_index": index, "library_label": label, "identity": ident,
            "started_monotonic_ns": start, "ended_monotonic_ns": end,
            "case_order": meta["case_order"], "metadata": meta, "mapped_files": map_paths,
            "cases": case_summaries, "status": status}


def aggregate_within(processes, label, case):
    selected = [p["cases"][case] for p in processes if p["library_label"] == label]
    ratios = [v for p in selected for v in p["block_ratios"]["values"]]
    o = sum(p["O_ns"] for p in selected)
    s = sum(p["S_ns"] for p in selected)
    return {"case": case, "label": label, "process_indices": [p["process_index"] for p in selected],
            "independent_process_units": 4, "descriptive_block_ratio_count": 64,
            "pooled_block_ratios": stats(ratios), "pooled_ratio_of_sums": o / s,
            "pooled_candidate_time_change_percent": delta_percent(o / s), "O_total_ns": o, "S_total_ns": s,
            "graphs_per_logical_mode": 4096, "O_mean_ms_per_graph": o / 4096e6,
            "S_mean_ms_per_graph": s / 4096e6,
            "process_ratio_of_sums": [p["ratio_of_sums"] for p in selected],
            "arithmetic_mean_of_process_ratios": statistics.fmean(p["ratio_of_sums"] for p in selected),
            "median_of_process_ratios": statistics.median(p["ratio_of_sums"] for p in selected),
            "process_total_improvements": sum(p["ratio_of_sums"] > 1 for p in selected)}


def cross_pair(processes, pair_number, f, e, case, logical):
    frozen = processes[f]["cases"][case]
    overlay = processes[e]["cases"][case]
    ratios = []
    deltas_twice = []
    for fb, eb in zip(frozen["blocks"], overlay["blocks"]):
        frozen_raw = fb["O_ns"] + fb["S_ns"]
        candidate = eb[logical + "_ns"]
        ratios.append(frozen_raw / (2 * candidate))
        deltas_twice.append(2 * candidate - frozen_raw)
    fr = frozen["total_ns"]
    et = overlay[logical + "_ns"]
    ratio = fr / (2 * et)
    return {"pair_index": pair_number, "frozen_process_index": f, "overlay_process_index": e,
            "case": case, "overlay_logical_mode": logical, "frozen_raw_total_ns": fr,
            "frozen_normalized_total_ns": fr / 2, "overlay_total_ns": et,
            "frozen_graph_calls": 2048, "overlay_logical_graph_calls": 1024,
            "ratio_of_normalized_sums": ratio, "candidate_time_change_percent": delta_percent(ratio),
            "delta_E_minus_normalized_F_ns": (2 * et - fr) / 2,
            "delta_E_minus_normalized_F_twice_ns": 2 * et - fr,
            "block_ratios": stats(ratios), "block_delta_twice_ns": deltas_twice}


def aggregate_cross(rows, case, logical):
    selected = [r for r in rows if r["case"] == case and r["overlay_logical_mode"] == logical]
    fr = sum(r["frozen_raw_total_ns"] for r in selected)
    et = sum(r["overlay_total_ns"] for r in selected)
    ratios = [x for r in selected for x in r["block_ratios"]["values"]]
    return {"case": case, "overlay_logical_mode": logical, "independent_process_pairs": 4,
            "pooled_block_ratios": stats(ratios), "pooled_ratio_of_normalized_sums": fr / (2 * et),
            "pooled_candidate_time_change_percent": delta_percent(fr / (2 * et)),
            "frozen_raw_total_ns": fr, "frozen_normalized_total_ns": fr / 2, "overlay_total_ns": et,
            "frozen_graph_calls": 8192, "overlay_logical_graph_calls": 4096,
            "frozen_mean_ms_per_graph": fr / 8192e6, "overlay_mean_ms_per_graph": et / 4096e6,
            "pair_ratio_of_normalized_sums": [r["ratio_of_normalized_sums"] for r in selected],
            "arithmetic_mean_of_pair_ratios": statistics.fmean(r["ratio_of_normalized_sums"] for r in selected),
            "pair_total_improvements": sum(r["ratio_of_normalized_sums"] > 1 for r in selected),
            "delta_E_minus_normalized_F_ns": (2 * et - fr) / 2}


def inspect_guard(root, name):
    g = load_json(root / "outer-guard" / name / "STATUS.json")
    require(g["exit_code"] == 0 and g["error"] is None, "outer guard failure")
    require(g["api_pid"] == 11151 and g["finished_unix"] >= g["started_unix"], "guard identity/clock mismatch")
    snapshots = [g["initial"]] + g["samples"]
    keys = set(snapshots[0]["host_counters"])
    require(all(set(x["host_counters"]) == keys for x in snapshots), "host counter inventory changed")
    changed = {key: [x["host_counters"][key] for x in snapshots] for key in sorted(keys)
               if len({x["host_counters"][key] for x in snapshots}) > 1}
    sensors = sorted({x["unix"] for x in snapshots})
    activity = [x["api_cores"] for x in g["samples"] if "api_cores" in x]
    return {"name": name, "status": "success", "started_unix": g["started_unix"],
            "finished_unix": g["finished_unix"], "elapsed_s": g["elapsed_s"],
            "start_utc": datetime.datetime.fromtimestamp(g["started_unix"], datetime.timezone.utc).isoformat(),
            "sample_count": len(g["samples"]), "distinct_sensor_timestamps": sensors,
            "sensor_timestamp_count": len(sensors), "reported_cpu_max_C": max(x["cpu_max_C"] for x in snapshots),
            "counter_fields": len(keys), "changed_counter_fields": changed,
            "all_reported_counters_zero": all(v == 0 for x in snapshots for v in x["host_counters"].values()),
            "minimum_available_bytes": min(x["available_kib"] for x in snapshots) * 1024,
            "API_cores_min": min(activity) if activity else None, "API_cores_max": max(activity) if activity else None,
            "sensor_latest_minus_guard_start_s": sensors[-1] - g["started_unix"],
            "precision_limit": "Sensor timestamps repeat; no per-chunk process CPU time or synchronized wall/monotonic clock pair."}


def verify_saved_analysis(saved, processes, within, placebo, cross_rows, cross_summaries):
    count = 0
    require(saved["passed"] is True and saved["analysis_complete"] is True, "saved analysis incomplete")
    require(saved["checks"]["chunks"] == 1536 and saved["checks"]["timed_graph_calls"] == 49152,
            "saved analysis count mismatch")
    for key in ("primary_overlay_within_process", "frozen_logical_placebo"):
        require(len(saved[key]) == 12, "saved primary case inventory wrong")
        for row in saved[key]:
            ours = processes[row["process_index"]]["cases"][row["case"]]
            require(row["logical_O_total_ns"] == ours["O_ns"] and row["logical_S_total_ns"] == ours["S_ns"], "saved raw totals differ")
            require(row["block_paired_ratios"]["values"] == ours["block_ratios"]["values"] and
                    near(row["ratio_of_total_times"], ours["ratio_of_sums"]), "saved block or sum ratios differ")
            count += 19
    for row, ours in zip(saved["cross_library_pairs"], cross_rows):
        require((row["pair_index"], row["case"], row["overlay_logical_mode"]) ==
                (ours["pair_index"], ours["case"], ours["overlay_logical_mode"]), "saved cross row order mismatch")
        require(row["frozen_raw_total_ns"] == ours["frozen_raw_total_ns"] and
                row["overlay_logical_total_ns"] == ours["overlay_total_ns"], "saved cross raw totals differ")
        require(row["block_normalized_ratios"]["values"] == ours["block_ratios"]["values"] and
                near(row["ratio_of_normalized_total_times"], ours["ratio_of_normalized_sums"]), "saved normalized ratio differs")
        count += 19
    require(len(saved["cross_library_pairs"]) == 24, "saved cross inventory wrong")
    for key, ours in (("overlay_within_process", within), ("frozen_placebo", placebo), ("cross_library", cross_summaries)):
        rows = saved["descriptive_summaries"][key]
        require(len(rows) == len(ours), "saved aggregate inventory wrong")
        for row, independently in zip(rows, ours):
            require(row["case"] == independently["case"], "saved aggregate case ordering mismatch")
            if key == "cross_library":
                require(row["overlay_logical_mode"] == independently["overlay_logical_mode"], "saved cross mode mismatch")
                a, b = row["pooled_normalized_total_ratio"], independently["pooled_ratio_of_normalized_sums"]
            else:
                a, b = row["pooled_equal_work_total_ratio"], independently["pooled_ratio_of_sums"]
            require(near(a, b) and row["pooled_block_paired_median"] == independently["pooled_block_ratios"]["median"], "saved aggregate statistic differs")
            count += 2
    return {"compared_numeric_values": count, "all_compared_values_match": True,
            "scope": "Every process/block ratio, integer total, cross pair ratio, pooled median and pooled ratio of sums."}


def screen(spec, within, cross, saved):
    criteria = spec["proposed_forward_screen"]
    rows = []
    for comparison, data in (("overlay_within_process", within), ("cross_library", [x for x in cross if x["overlay_logical_mode"] == "S"])):
        for x in data:
            case = x["case"]
            if comparison == "overlay_within_process":
                minimum = criteria["E_C6_min_ratio_both_pooled_block_median_and_equal_work_total"] if case == "C6" else 1 / (1 + criteria["E_C0_Cplan_max_candidate_time_increase_both_metrics"])
                total = x["pooled_ratio_of_sums"]
            else:
                minimum = criteria["F_vs_E_shared_C6_min_ratio_both_metrics"] if case == "C6" else 1 / (1 + criteria["F_vs_E_shared_C0_Cplan_max_candidate_time_increase_both_metrics"])
                total = x["pooled_ratio_of_normalized_sums"]
            median = x["pooled_block_ratios"]["median"]
            rows.append({"comparison": comparison, "case": case, "minimum_ratio": minimum,
                         "median_ratio": median, "total_time_ratio": total,
                         "median_time_change_percent": delta_percent(median),
                         "total_time_change_percent": delta_percent(total),
                         "median_pass": median >= minimum, "total_pass": total >= minimum})
    c6 = next(x for x in within if x["case"] == "C6")
    result = {"rows": rows, "overlay_C6_process_total_ratios": c6["process_ratio_of_sums"],
              "overlay_C6_improving_processes": c6["process_total_improvements"],
              "numeric_screen_passed": all(x["median_pass"] and x["total_pass"] for x in rows) and c6["process_total_improvements"] >= 3,
              "production_adoption": False}
    for key in result:
        require(result[key] == saved[key], "screen result differs: " + key)
    result["comparison_to_saved_SCREEN_DECISION"] = "exact equality of all reproduced rows and decisions"
    result["criterion_status"] = "Unchanged prospective criterion; no new threshold or exclusions."
    return result


def design_diagnostics(processes, cross_rows):
    contributions = []
    for pair_index, (f, e) in enumerate(PAIRS):
        o = next(x for x in cross_rows if x["pair_index"] == pair_index and x["case"] == "C0" and x["overlay_logical_mode"] == "O")
        s = next(x for x in cross_rows if x["pair_index"] == pair_index and x["case"] == "C0" and x["overlay_logical_mode"] == "S")
        contributions.append({"pair": [f, e], "delta_E_O_minus_F_ns": o["delta_E_minus_normalized_F_ns"],
                              "delta_E_S_minus_F_ns": s["delta_E_minus_normalized_F_ns"],
                              "S_time_change_percent": s["candidate_time_change_percent"]})
    total = sum(x["delta_E_S_minus_F_ns"] for x in contributions)
    later = sum(x["delta_E_S_minus_F_ns"] for x in contributions[2:])
    periods = []
    for p in processes:
        for position, case in enumerate(p["case_order"]):
            c = p["cases"][case]
            periods.append({"process_index": p["process_index"], "library_label": p["library_label"],
                            "case": case, "case_position": position,
                            "begin_ns": c["begin_ns"], "end_ns": c["end_ns"],
                            "timed_ns": c["total_ns"], "all_arm_mean_ms_per_graph": c["total_ns"] / 2048e6})
    moments = []
    for exponent in range(4):
        moments.append({"power": exponent,
                        "F_sum_process_index_power": sum(i**exponent for i in range(8) if LABELS[i] == "F"),
                        "E_sum_process_index_power": sum(i**exponent for i in range(8) if LABELS[i] == "E")})
    return {"C0_pair_contributions": contributions, "C0_total_E_S_minus_F_ns": total,
            "C0_last_two_pairs_signed_delta_ns": later,
            "C0_last_two_pairs_share_of_signed_total_percent": 100 * later / total,
            "case_periods": periods, "outer_index_moments": moments,
            "outer_pair_orders": ["FE" if f < e else "EF" for f, e in PAIRS],
            "identifiability": [
                "E O/S alternates within one process and cache case; this is the narrowest direct shared-mode comparison.",
                "F O/S is an identical-actual-mode placebo; its measured imbalance is not a correction to subtract from E.",
                "Each process loads exactly one CPU library. A library indicator is a linear combination of process indicators, so arbitrary process effects cannot be separated from a library effect.",
                "All eight processes are sequential in one fixed suite. Balanced FE/EF and index moments reduce specified simple drift confounding, not arbitrary periods, allocator state, clock or scheduler differences.",
                "Index powers 0,1,2 balance only by design index; actual elapsed time and cache-case timing are not uniformly spaced.",
                "The 95.4 percent concentration is an exact signed-difference allocation, not the fraction causally explained by external interference.",
                "No 32 individual graph latencies, process CPU time, scheduler counters, page faults or per-chunk frequency trace are available."]}


def run(args):
    repository = args.repository.resolve()
    root = repository / HANDOFF
    out = args.output_dir.resolve()
    require(not out.is_relative_to(repository), "audit output must be outside repository")
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "MIXED_INDEPENDENT_REANALYSIS.json").exists() and not (out / "restored").exists(), "audit outputs already exist")
    source_identity = {"script": str(Path(__file__).resolve()), **file_record(Path(__file__))}
    manifest = verify_manifest(root)
    incoming_manifest = verify_manifest(root / INCOMING)
    prior_check = same_return(root / INCOMING, args.prior_return.resolve())
    restored = restore_expected(root, out)
    spec = load_json(root / INCOMING / "NEXT_RUN_SPEC.json")
    case_spec = load_json(root / INCOMING / "next_experiment/case/mixed_case.json")
    require(spec["commit"] == case_spec["commit"] == BASE_COMMIT, "fixture base commit differs")
    require(spec["schedule"] == [{"index": i, "library": LABELS[i], "case_order": list(CASES[(i//2)%3:] + CASES[:(i//2)%3])} for i in range(8)], "schedule differs")
    require(spec["cross_pairs_frozen_overlay"] == [list(x) for x in PAIRS], "pairing differs")
    case_dir = root / INCOMING / "next_experiment/case"
    require(file_record(case_dir / "inputs.bin") == {"bytes": 163840, "sha256": INPUT_HASH}, "input fixture differs")
    require(file_record(case_dir / "source-plan.txt")["sha256"] == PLAN_HASH, "plan differs")
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", (case_dir / "inputs.bin").read_bytes())), "nonfinite input")
    target = [line.split() for line in (case_dir / "source-plan.txt").read_text().splitlines() if line.split()[:1] == ["blk.0.ffn_up_exps.weight"]]
    require(len(target) == 1 and target[0][1:6] == ["iq2_s", "5120", "2304", "384", "313"], "plan row differs")
    require(sorted(set(map(int, target[0][6:])) & set(case_spec["routes"])) == case_spec["cache_ids"], "plan intersection differs")
    for source in case_spec["sources"]:
        require(file_record(safe_path(repository, source["repo_relative_path"])) == {k: source[k] for k in ("bytes", "sha256")}, "pinned prior fixture source differs")
    build = load_json(root / "build/BUILD.json")
    require(build["cpu_library_rebuilt"] is False and valid_sha(build["harness_sha256"]), "build scope differs")
    require(build["source_sha256"] == file_record(root / INCOMING / "next_experiment/mixed_probe.cpp")["sha256"], "build source hash differs")
    require(build["case_header_sha256"] == file_record(case_dir / "mixed_case_data.h")["sha256"], "build header hash differs")
    previous_build = repository / "archive/review-handoffs-not-for-readers/2026-11-16-abs-operator/FIXED_BUILD.json"
    require(build["previous_build_receipt_sha256"] == file_record(previous_build)["sha256"], "previous build receipt differs")
    processes = [load_process(root, i, case_spec, spec, build, restored["targets"]) for i in range(8)]
    invariants = ("base_library_sha256", "harness_sha256", "inputs_sha256", "plan_sha256", "case_spec_sha256",
                  "case_header_sha256", "launcher_sha256", "shard_stat", "extra_library_dirs")
    for i, p in enumerate(processes):
        require(all(canonical(p["identity"][key]) == canonical(processes[0]["identity"][key]) for key in invariants), "process invariant mismatch")
        if i:
            require(processes[i-1]["ended_monotonic_ns"] <= p["started_monotonic_ns"], "processes overlap or reordered")
    within = [aggregate_within(processes, "E", case) for case in CASES]
    placebo = [aggregate_within(processes, "F", case) for case in CASES]
    cross_rows = [cross_pair(processes, j, f, e, case, logical) for j, (f, e) in enumerate(PAIRS) for case in CASES for logical in ("O", "S")]
    cross_summaries = [aggregate_cross(cross_rows, case, logical) for case in CASES for logical in ("O", "S")]
    saved_analysis = verify_saved_analysis(load_json(root / "MIXED_ANALYSIS.json"), processes, within, placebo, cross_rows, cross_summaries)
    gates = screen(spec, within, cross_summaries, load_json(root / "SCREEN_DECISION.json"))
    guard = [inspect_guard(root, "mixed-build-20261117")] + [inspect_guard(root, f"iq2-mixed-20261117-p{i}") for i in range(8)]
    initial = load_json(root / "INITIAL_RUNTIME.json")
    final = load_json(root / "FINAL_RUNTIME.json")
    require(initial["files"] == final["unchanged_hashes"] and initial["health"]["pid"] == final["api_pid"] == 11151 and final["same_pid_as_initial"] is True,
            "initial/final runtime receipts differ")
    diagnostic = design_diagnostics(processes, cross_rows)
    result = {
        "audit_schema": 1, "audit_passed": True, "source_commit": COMMIT,
        "scope": "Independent offline evidence audit; no new performance or model execution. Numeric screen remains failed.",
        "repository": str(repository), "source_identity": source_identity,
        "manifest": manifest, "incoming_manifest": incoming_manifest, "prior_return_byte_comparison": prior_check,
        "correctness_restoration": restored, "build_receipt": build,
        "runtime_receipts": {"unchanged_recorded_hashes": initial["files"], "same_API_pid": 11151, "final_health": final["API_health"],
                             "limit": "Receipts and maps were audited; live binaries/API/model were neither available nor executed here."},
        "checks": {"processes": 8, "chunks": 1536, "timed_graph_calls": 49152, "cases": 24,
                   "expected_files": 24, "exact_float_values_per_expected_file": 110592,
                   "raw_records": 1576, "all_chunks_used": True, "excluded_observations": 0,
                   "timed_elapsed_ns_all_chunks": sum(c["total_ns"] for p in processes for c in p["cases"].values()),
                   "independent_processes_per_library": 4, "new_benchmarks": 0,
                   "all_processes_nonoverlapping": True, "common_identity_fields": list(invariants)},
        "processes": processes, "within_overlay": within, "frozen_placebo": placebo,
        "cross_library_pairs": cross_rows, "cross_library_summaries": cross_summaries,
        "saved_analysis_comparison": saved_analysis, "prospective_screen": gates,
        "guard_summaries": guard, "design_and_C0_diagnostics": diagnostic,
        "statistical_limits": [
            "A ratio of sums equals a candidate-duration-weighted mean of paired ratios, not their arithmetic mean or median.",
            "Because graph counts are equal after F/2 normalization, total-time ratios equal ratios of arithmetic mean time per graph.",
            "All 32 calls inside a chunk are averaged; individual graph latencies and tails cannot be recovered.",
            "No confidence interval, pseudo-independent bootstrap, p-value, causal attribution, outlier deletion or retrospective gate change is performed.",
            "4 process units per library, one suite, one tensor, artificial inputs and fixed routes; no whole-model speed, production hit rate or answer-quality conclusion."]}
    with (out / "MIXED_INDEPENDENT_REANALYSIS.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    sources = {str(Path(HANDOFF) / name): record for name, record in manifest["files"].items()}
    sources[str(Path(HANDOFF) / "FILES_SHA256.json")] = manifest["manifest"]
    for source in case_spec["sources"]:
        sources[source["repo_relative_path"]] = {k: source[k] for k in ("bytes", "sha256")}
    sources[previous_build.relative_to(repository).as_posix()] = file_record(previous_build)
    with (out / "MIXED_AUDIT_SOURCE_SHA256.json").open("x", encoding="utf-8") as stream:
        json.dump({"source_commit": COMMIT, "audit_script": source_identity, "files": sources}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"audit_passed": True, "new_performance_runs": 0, "raw_chunks": 1536,
                      "gate_passed": gates["numeric_screen_passed"],
                      "C0_late_pair_signed_delta_share_percent": diagnostic["C0_last_two_pairs_share_of_signed_total_percent"],
                      "output": str(out / "MIXED_INDEPENDENT_REANALYSIS.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--prior-return", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print("AUDIT FAILED; original evidence retained: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
