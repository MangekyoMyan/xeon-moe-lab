#!/usr/bin/env python3
"""Independent read-only audit of the saved 2026-11-16 abs-operator handoff.

Only Python standard-library parsing/hash/statistics are used. No imported
repository code, model, executable, API, benchmark, or build is executed.
Gzip restoration is bounded by declared lengths and confined outside the repo.
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics as stats
import struct

COMMIT = "31b1292b2767618a28ef6e3f1c9c4450b1fc21be"
REL = Path("archive/review-handoffs-not-for-readers")
TENSORS = {"iq2": ("iq2_s", "blk.0.ffn_up_exps.weight", 5120, 2304, 1),
           "iq3s": ("iq3_s", "blk.13.ffn_down_exps.weight", 2304, 5120, 6)}
ORDER = ("frozen", "off", "shared", "shared", "off", "frozen")
FROZEN_SHA = "33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3"
OVERLAY_SHA = "120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "Duplicate JSON key: " + key)
        out[key] = value
    return out


def decode_json(text):
    def invalid(x):
        raise ValueError("Nonfinite JSON literal: " + x)
    return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=invalid)


def safe_relative(value):
    require(isinstance(value, str) and "\\" not in value, "Invalid relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and value and all(x not in (".", "..") for x in path.parts), "Unsafe relative path")
    return Path(*path.parts)


def description(values):
    require(values and all(math.isfinite(x) for x in values), "Invalid statistical input")
    return {"n": len(values), "median": stats.median(values), "arithmetic_mean": stats.mean(values),
            "min": min(values), "max": max(values), "all": values}


def expected_golden_headers(final):
    configs = []
    for t, k, inputs, strided, fp32 in itertools.product((22, 21), (256, 2304, 5120), (1, 6), (0, 1), (0, 1)):
        configs.append((t, k, inputs, 1, strided, fp32, 0))
    for t in (22, 21):
        for inputs, fp32 in itertools.product((64, 65), (0, 1)):
            configs.append((t, 256, inputs, 1, 1, fp32, 1))
        configs.extend(((t, 256, 6, 3, 1, 0, 1), (t, 2304, 6, 1, 1, 0, 1)))
    configs.append((18, 256, 6, 1, 1, 0, 1))
    if final:
        for k, inputs, tokens in itertools.product((256, 2304), (1, 6), (1, 3)):
            configs.append((39, k, inputs, tokens, 1, 1, 1))
    for config in configs:
        t, k, inputs, tokens, strided, fp32, duplicate = config
        for selection, threads, phase, reference in itertools.product(range(1 if t == 39 else 3), (12, 1, 24), range(5 if final else 3), (0, 1)):
            yield (*config, selection, threads, phase, reference, 19 * max(6, inputs) * tokens)


def parse_golden(data, final):
    offset = 0; count = 0; values = 0
    floats = collections.Counter(); by_type = collections.Counter(); by_phase = collections.Counter()
    header_digest = hashlib.sha256()
    for expected in expected_golden_headers(final):
        require(offset + 96 <= len(data), "Truncated golden header")
        raw_header = data[offset:offset + 96]; header = struct.unpack("<12Q", raw_header); offset += 96
        require(header == expected, "Golden header/order differs from explicit source-derived inventory")
        header_digest.update(raw_header)
        size = 4 * header[-1]
        require(offset + size <= len(data), "Truncated golden payload")
        payload = data[offset:offset + size]; offset += size
        for (word,) in struct.iter_unpack("<I", payload):
            category = "finite" if (word & 0x7f800000) != 0x7f800000 else "nan" if word & 0x007fffff else "inf"
            floats[category] += 1
        count += 1; values += header[-1]; by_type[str(header[0])] += 1; by_phase[str(header[9])] += 1
    require(offset == len(data), "Extra golden bytes")
    return {"records": count, "output_values_per_mode": values, "bytes": len(data),
            "float_categories": dict(floats), "records_by_weight_type": dict(by_type),
            "records_by_phase": dict(by_phase), "ordered_header_sha256": header_digest.hexdigest(),
            "scope": "Frozen expected outputs with exact source-derived header inventory; candidate outputs are checked by recorded executable receipts, not independently dumped here."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repository.resolve(); out = args.output_dir.resolve()
    require(not out.is_relative_to(repo), "Audit output must be outside repository")
    out.mkdir(parents=True, exist_ok=True)
    output = out / "ABS_OPERATOR_REANALYSIS.json"
    require(not output.exists(), "Use a new output directory or remove only your previous audit output deliberately")
    n = repo / REL / "2026-11-16-abs-operator"
    h = repo / REL / "2026-11-16-current"
    tracked = set()

    def bytes_(path):
        tracked.add(path); return path.read_bytes()

    def read(path):
        return decode_json(bytes_(path).decode("utf-8"))

    manifest = read(n / "FILES_SHA256.json")
    manifest_files = []
    for rel, entry in manifest.items():
        path = (n / safe_relative(rel)).resolve()
        require(path.is_relative_to(n), "Manifest path escape")
        data = bytes_(path)
        require(len(data) == entry["bytes"] and sha(data) == entry["sha256"], "Export manifest mismatch: " + rel)
        manifest_files.append(rel)
    actual = {str(p.relative_to(n)) for p in n.rglob("*") if p.is_file()}
    require(actual == set(manifest) | {"FILES_SHA256.json"}, "Export contains undeclared files or is missing files")

    # Restore only authenticated data, with a hard bound on decompressed bytes.
    index = read(n / "correctness/INDEX.json")
    require(len(index) == 6, "Expected six content-addressed blobs")
    restored = {}; blob_report = []
    for digest, entry in index.items():
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid digest")
        expected_size = entry["bytes"]
        require(isinstance(expected_size, int) and 0 < expected_size <= 16 * 1024 * 1024, "Unexpected blob size")
        source = (n / safe_relative(entry["gzip"])).resolve()
        require(source.is_relative_to(n), "Gzip path escape")
        tracked.add(source)
        with gzip.open(source, "rb") as stream:
            data = stream.read(expected_size + 1)
        require(len(data) == expected_size and sha(data) == digest, "Decompressed hash/size mismatch")
        require(entry["targets"], "Blob without target")
        for rel in entry["targets"]:
            require(rel not in restored, "Duplicate restoration target")
            target = (out / "restored" / safe_relative(rel)).resolve()
            require(target.is_relative_to(out / "restored"), "Restoration path escape")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.read_bytes() == data, "Existing restored data differs")
            else:
                with target.open("xb") as stream:
                    stream.write(data)
            restored[rel] = {"path": str(target), "bytes": expected_size, "sha256": digest}
        blob_report.append({"sha256": digest, "bytes": expected_size, "target_count": len(entry["targets"]), "targets": entry["targets"]})

    golden = {}
    for rel, final in (("golden.bin", False), ("golden-r2.bin", True)):
        golden[rel] = parse_golden(Path(restored[rel]["path"]).read_bytes(), final)

    # Identity is recorded source/command/path provenance, not a rehash of absent ELF bytes.
    source = read(n / "SOURCE.json"); build = read(n / "BUILD.json")
    base = h / "snapshot/llm-lab/exploration-20261112/mx-prepack/release-candidate-a"
    require(sha(bytes_(base / "sources/ggml-cpu.c")) == source["baseline_sha256"], "Baseline source hash")
    require(sha(bytes_(base / "BUILD.json")) == build["baseline_build_sha256"], "Baseline build hash")
    require(build["sha256"] == OVERLAY_SHA, "Overlay identity changed")
    for name, digest in source["files"].items():
        require(sha(bytes_(n / name)) == digest == build["source_hashes"][name], "Overlay source hash")
        require(bytes_(n / name) == bytes_(n / "checkpoint-screen-r1r2" / name), "Kernel changed since screens")
    checkpoint = read(n / "checkpoint-screen-r1r2/SHA256.json")
    missing_binaries = []
    for rel, digest in checkpoint.items():
        path = n / "checkpoint-screen-r1r2" / safe_relative(rel)
        if path.exists():
            require(sha(bytes_(path)) == digest, "Checkpoint hash: " + rel)
        elif rel in restored:
            require(restored[rel]["sha256"] == digest, "Checkpoint restored hash")
        else:
            require(rel in ("probe", "runtime/libggml-cpu.so.0.28.0"), "Unexpected missing checkpoint artifact")
            missing_binaries.append({"name": rel, "recorded_sha256": digest})
    probe_builds = {}
    for label, meta, code in (("screen", "checkpoint-screen-r1r2/PROBE_BUILD.json", "checkpoint-screen-r1r2/probe.cpp"),
                              ("final_validation", "PROBE_BUILD.json", "probe.cpp"),
                              ("fixed", "FIXED_BUILD.json", "fixed_probe.cpp")):
        d = read(n / meta)
        require(sha(bytes_(n / code)) == d["source_sha256"], "Probe source hash")
        probe_builds[label] = {"source_sha256": d["source_sha256"], "recorded_elf_sha256": d["probe_sha256"]}
    old = bytes_(n / "checkpoint-screen-r1r2/probe.cpp").decode()
    new = bytes_(n / "probe.cpp").decode()
    def bench_body(text):
        return text.split("static void benchmark(", 1)[1].split("int main(", 1)[0]
    require(bench_body(old) == bench_body(new), "Benchmark body changed between checkpoint and final source")

    statuses = {}; stdout = {}; build_history = []
    for folder in sorted((n / "execution").iterdir()):
        require(folder.is_dir(), "Unexpected execution entry")
        st = read(folder / "STATUS.json"); content = bytes_(folder / "stdout.log").decode()
        err = bytes_(folder / "stderr.log")
        require(st["exit_code"] == 0 and st["error"] is None, "Failed recorded execution: " + folder.name)
        require(st["finished_unix"] >= st["started_unix"] and st["elapsed_s"] > 0, "Invalid execution time")
        statuses[folder.name] = st; stdout[folder.name] = content
        build_history.append({"run": folder.name, "start_unix": st["started_unix"], "finish_unix": st["finished_unix"],
                              "exit_code": st["exit_code"], "stderr_bytes": len(err), "stdout_sha256": sha(content.encode())})
        if "build" not in folder.name:
            require(not err, "Unexpected stderr: " + folder.name)
    for suffix, name in (("r1", "golden.bin"), ("r2", "golden-r2.bin")):
        for role in ("frozen", "abs"):
            d = decode_json(stdout[f"validate-{role}-{suffix}"])
            mult = 1 if role == "frozen" else 3
            require(d["passed"] and d["graph_executions"] == golden[name]["records"] * mult
                    and d["checked_outputs"] == golden[name]["output_values_per_mode"] * mult
                    and d["experimental"] == (role == "abs") and d["work_data_mod64"] == 16, "Golden validation count")
    require(decode_json(stdout["validate-sanitized-r2"]) == decode_json(stdout["validate-abs-r2"]), "Sanitized validation receipt differs")

    route_baseline = None; parsed_runs = {}; env = []
    def timing_run(name, tag, fixed=False, label=None):
        nonlocal route_baseline
        rows = [decode_json(line) for line in stdout[name].splitlines() if line.strip()]
        require(len(rows) == 3 and rows[-1] == {"passed": True}, "Timing stdout inventory")
        rows = rows[:-1]
        require([r["cached_of_six"] for r in rows] == [6, 0], "Cache order/inventory")
        typ, tensor, k, m, inputs = TENSORS[tag]
        st = statuses[name]; argv = st["argv"]
        require("--interleave=0,1" in argv and "--physcpubind=0-23" in argv and tensor in argv, "Timing launch identity")
        expected_dir = "/exploration-20261112/mx-prepack/release-candidate-a/" if label == "frozen" else "/optimization-20261116/abs-operator/runtime/"
        require(expected_dir in argv[-1] and argv[-1].endswith("libggml-cpu.so.0.28.0"), "Requested CPU provider path")
        if fixed:
            require(argv[-2] == ("1" if label == "shared" else "0"), "Fixed CLI mode")
        for row in rows:
            require(row["kind"] == ("fixed" if fixed else "operator") and row["type"] == typ and row["tensor"] == tensor, "Tensor/type identity")
            require(row["threads"] == 12 and sorted(row["pins"]) == list(range(12)) and row["ggml_numa"] is False, "Worker/NUMA identity")
            require(row["bit_mismatches"] == 0 and row["cache_bytes"] == (318504960 if row["cached_of_six"] else 0), "Correctness/cache receipt")
            route = row["routes"]
            require(len(route) == len(set(route)) == 48 and all(isinstance(x, int) and 0 <= x < 384 for x in route), "Route inventory")
            if route_baseline is None:
                route_baseline = route
            require(route == route_baseline, "Routes differ across runs")
            if fixed:
                require(row["mode"] == (1 if label == "shared" else 0) and row["experimental"] == (label != "frozen"), "Fixed mode/export identity")
            else:
                require((row["K"], row["M"], row["inputs"]) == (k, m, inputs), "Actual shape")
        samples = st["samples"]; times = sorted(set(x["unix"] for x in samples))
        env.append({"run": name, "guard_elapsed_s": st["elapsed_s"], "guard_samples": len(samples),
                    "unique_sensor_timestamps": len(times), "sensor_timestamps_inside_run": [t for t in times if st["started_unix"] <= t <= st["finished_unix"]],
                    "first_sensor_age_at_start_s": st["started_unix"] - st["initial"]["unix"],
                    "api_cores_max": max(x["api_cores"] for x in samples),
                    "available_kib_min": min(x["available_kib"] for x in samples),
                    "scope": "Saved daemon snapshots; not continuous fresh sensors or per-timed-chunk timestamps."})
        parsed_runs[name] = rows
        return rows

    same = []; chronological = []; timing_count = 0
    published = read(n / "ANALYSIS.json")
    for tag, repeat in itertools.product(TENSORS, (1, 2)):
        name = f"real-{tag}-r{repeat}"
        for row in timing_run(name, tag):
            samples = row["pairs"]; timing_count += len(samples)
            expected_keys = {(b, c, o) for b in range(16) for c in (1, 2) for o in range(4)}
            require(len(samples) == 128 and {(s["block"], s["candidate"], s["order"]) for s in samples} == expected_keys, "Same-library timing inventory")
            for i, sample in enumerate(samples):
                b, c, o = sample["block"], sample["candidate"], sample["order"]
                rev = (b % 8 + b // 8) % 2
                expected_order = [c, 0, 0, c] if rev else [0, c, c, 0]
                require(sample["mode"] == expected_order[o] and sample["route_group"] == b % 8 and math.isfinite(sample["ms"]) and sample["ms"] > 0, "Timing order/data invalid")
                expected_c = (2 - ((i % 8) // 4)) if rev else 1 + ((i % 8) // 4)
                require(b == i // 8 and c == expected_c and o == i % 4, "Raw execution order differs from source")
                chronological.append({"run": name, "hits": row["cached_of_six"], "index_in_condition": i, **sample})
            for c in (1, 2):
                blocks = []; av = []; bv = []
                for b in range(16):
                    group = [s for s in samples if s["candidate"] == c and s["block"] == b]
                    a = [s["ms"] for s in group if s["mode"] == 0]; candidate = [s["ms"] for s in group if s["mode"] == c]
                    require(len(a) == len(candidate) == 2, "Block arm inventory")
                    av.extend(a); bv.extend(candidate)
                    blocks.append({"block": b, "route_group": b % 8, "reverse": (b % 8 + b // 8) % 2,
                                   "current_mean_ms": stats.mean(a), "candidate_mean_ms": stats.mean(candidate), "ratio": sum(a) / sum(candidate)})
                ratios = [x["ratio"] for x in blocks]; median = stats.median(ratios)
                pub = [p for p in published["same_library"] if (p["run"], p["hits"], p["candidate"]) == (name, row["cached_of_six"], c)]
                require(len(pub) == 1 and all(math.isclose(x, y, rel_tol=1e-13) for x, y in zip(ratios, pub[0]["all"])), "Published same-library ratios mismatch")
                gate = median >= 1.02 and sum(r > 1 for r in ratios) >= 12 if row["cached_of_six"] else 1 / median <= 1.01
                require(gate == pub[0]["screen_pass"], "Published same-library gate mismatch")
                same.append({"run": name, "tag": tag, "repeat": repeat, "hits": row["cached_of_six"], "candidate": c,
                             "paired_ratios": description(ratios), "wins": sum(r > 1 for r in ratios),
                             "median_time_change_percent": 100 * (1 / median - 1), "prespecified_median_gate_pass": gate,
                             "total_time_ratio": sum(av) / sum(bv), "total_time_change_percent": 100 * (sum(bv) / sum(av) - 1),
                             "current_sample_ms": description(av), "candidate_sample_ms": description(bv),
                             "current_block_mean_median_ms": stats.median(x["current_mean_ms"] for x in blocks),
                             "candidate_block_mean_median_ms": stats.median(x["candidate_mean_ms"] for x in blocks),
                             "timed_graphs_per_arm": len(av) * 8,
                             "order_diagnostics": [{"reverse": rev, "total_time_ratio": sum(x["current_mean_ms"] for x in blocks if x["reverse"] == rev) / sum(x["candidate_mean_ms"] for x in blocks if x["reverse"] == rev)} for rev in (0, 1)],
                             "blocks": blocks})
    require(timing_count == 1024, "Total same-library timing count")

    receipt = read(n / "FIXED_RECEIPT.json")
    require(receipt["complete"] is True and len(receipt["runs"]) == 12, "Fixed receipt inventory")
    cross = []; fixed_processes = []; fixed_count = 0; file_checks = 0
    for tag in TENSORS:
        runs = [r for r in receipt["runs"] if r["tag"] == tag]
        require([r["label"] for r in runs] == list(ORDER) and [r["position"] for r in runs] == list(range(6)), "Fixed process order")
        for i, r in enumerate(runs):
            require(r["run"] == f"fixed-{tag}-{i}-{ORDER[i]}", "Fixed run ID")
            require(set(r["files"]) == {"inputs.bin", "outputs-hit0.bin", "outputs-hit6.bin"}, "Fixed file inventory")
            for fname, info in r["files"].items():
                rel = f"fixed-outputs/{r['run']}/{fname}"
                require(info["sha256"] == restored[rel]["sha256"] and info["bytes"] == restored[rel]["bytes"], "Fixed file receipt hash/size")
                data = Path(restored[rel]["path"]).read_bytes()
                require(sha(data) == info["sha256"], "Restored fixed hash")
                typ, tensor, k, m, inputs = TENSORS[tag]
                expected_bytes = 8 * k * inputs * 4 if fname == "inputs.bin" else 32 * 6 * m * 4
                require(len(data) == expected_bytes, "Fixed shape/byte count")
                require(all((word & 0x7f800000) != 0x7f800000 for (word,) in struct.iter_unpack("<I", data)), "Nonfinite fixed input/output")
                if fname != "inputs.bin":
                    stride = 6 * m * 4
                    require(all(data[b * stride:(b + 1) * stride] == data[(b % 8) * stride:(b % 8 + 1) * stride] for b in range(32)), "Fixed repeated group outputs differ")
                file_checks += 1
            timing_run(r["run"], tag, True, r["label"])
            if i:
                require(statuses[runs[i - 1]["run"]]["finished_unix"] <= statuses[r["run"]]["started_unix"], "Fixed processes overlap/out of order")
        for fname in ("inputs.bin", "outputs-hit0.bin", "outputs-hit6.bin"):
            require(len({r["files"][fname]["sha256"] for r in runs}) == 1, "Fixed inputs/outputs differ across processes")
        for hits in (6, 0):
            groups = {label: [] for label in ("frozen", "off", "shared")}
            for r in runs:
                row = next(x for x in parsed_runs[r["run"]] if x["cached_of_six"] == hits)
                samples = row["samples"]; fixed_count += len(samples)
                require(len(samples) == 32 and [s["block"] for s in samples] == list(range(32)), "Fixed block inventory")
                require(all(s["route_group"] == s["block"] % 8 and math.isfinite(s["ms"]) and s["ms"] > 0 for s in samples), "Fixed timing invalid")
                times = [s["ms"] for s in samples]; groups[r["label"]].append(times)
                fixed_processes.append({"run": r["run"], "tag": tag, "label": r["label"], "position": r["position"], "hits": hits,
                                        "sample_ms": description(times), "timed_graphs": 32 * 32, "summed_timed_graph_ms": 32 * sum(times)})
            means = {label: [stats.mean(v[b] for v in sets) for b in range(32)] for label, sets in groups.items()}
            for label in ("off", "shared"):
                ratios = [a / b for a, b in zip(means["frozen"], means[label])]
                pub = [p for p in published["cross_library"] if (p["type"], p["hits"], p["candidate"]) == (tag, hits, label)]
                require(len(pub) == 1 and all(math.isclose(x, y, rel_tol=1e-13) for x, y in zip(ratios, pub[0]["all"])), "Published cross-library ratios mismatch")
                total_ratio = sum(map(sum, groups["frozen"])) / sum(map(sum, groups[label]))
                median = stats.median(ratios)
                gate = median >= 1.02 if hits else 1 / median <= 1.01
                require(gate == pub[0]["screen_pass"], "Cross-library published gate")
                cross.append({"tag": tag, "hits": hits, "candidate": label, "paired_process_mean_block_ratios": description(ratios),
                              "median_time_change_percent": 100 * (1 / median - 1), "prespecified_median_gate_pass": gate,
                              "total_time_ratio": total_ratio, "total_time_change_percent": 100 * (1 / total_ratio - 1),
                              "process_medians_ms": {lab: [stats.median(v) for v in vv] for lab, vv in groups.items()},
                              "process_means_ms": {lab: [stats.mean(v) for v in vv] for lab, vv in groups.items()},
                              "first_half_process_median_ratio": stats.median(groups["frozen"][0]) / stats.median(groups[label][0]),
                              "second_half_process_median_ratio": stats.median(groups["frozen"][1]) / stats.median(groups[label][1]),
                              "scope": "Two processes per label; 32 block ratios are not 32 independent process comparisons."})
    require(fixed_count == 768 and file_checks == 36, "Fixed total inventory")
    all_missed_stats = []
    for name, rows in parsed_runs.items():
        graphs = 32 if name.startswith("fixed-") else 8
        total_ms = sum(s["ms"] for row in rows for s in row["samples" if name.startswith("fixed-") else "pairs"]) * graphs
        require(total_ms < statuses[name]["elapsed_s"] * 1000, "Recorded timed work exceeds process wall time")
        all_missed_stats.append({"run": name, "summed_timed_graph_ms": total_ms, "guard_elapsed_ms": statuses[name]["elapsed_s"] * 1000})
    result = {"source_commit": COMMIT,
              "scope": "Offline parsing, bounded gzip restoration outside repository, independent reaggregation; no model/library/benchmark/API execution.",
              "manifest": {"files_verified": len(manifest), "exact_export_inventory": True},
              "compressed_evidence": {"blobs": blob_report, "restored_targets": len(restored), "fixed_files_checked": file_checks,
                                      "golden_structure": golden},
              "identity": {"recorded_overlay_library_sha256": OVERLAY_SHA, "recorded_frozen_library_sha256": FROZEN_SHA,
                           "elf_bytes_supplied": False, "kernel_sources_unchanged_since_screen": True,
                           "same_library_benchmark_body_unchanged_in_final_probe": True, "probe_builds": probe_builds,
                           "missing_checkpoint_executables": missing_binaries,
                           "limits": "Provider path verified by source dladdr check and successful receipts; build hashes recorded but absent remote ELFs cannot be independently rehashed. Same-library inputs are seeded in source but their original byte files are not exported."},
              "inventory": {"same_library_timed_chunks": timing_count, "same_library_graph_calls_in_timing": timing_count * 8,
                            "fixed_timed_chunks": fixed_count, "fixed_graph_calls_in_timing": fixed_count * 32,
                            "same_library_processes": 4, "fixed_processes": 12, "fixed_processes_per_label_per_type": 2,
                            "all_timing_ms_positive_finite": True, "all_published_ratios_and_gates_reproduced": True,
                            "same_routes_all_timing_records": route_baseline},
              "same_library": same, "same_library_chronological_samples": chronological,
              "cross_library": cross, "fixed_processes": fixed_processes,
              "guard_scope": env, "timed_work_vs_process_wall": all_missed_stats,
              "execution_history": sorted(build_history, key=lambda x: x["start_unix"]),
              "limitations": ["All timing outliers retained. Median-ratio gate is not silently replaced by a total-time gate.",
                              "Mean of paired ratios, median paired ratio, and ratio of total measured time are distinct estimands.",
                              "Within-process repeats and eight fixed input/route groups do not create independent process replicates.",
                              "Per-chunk wall timestamps and CPU clocks are not saved; exact association with guard events is unavailable.",
                              "Fixed output files save mode0 expected arrays; candidate equality is asserted after timing, not independently dumped.",
                              "Stored bytes do not include model weights or candidate validation output dumps; this is not an independent arithmetic re-execution."],
              "source_files": [{"path": str(p.relative_to(repo)), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in sorted(tracked)]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"output": str(output), "manifest_files": len(manifest), "restored_targets": len(restored),
                      "same_library_chunks": timing_count, "fixed_chunks": fixed_count,
                      "golden_r2_records": golden["golden-r2.bin"]["records"]}))


if __name__ == "__main__":
    main()
