#!/usr/bin/env python3
"""Read-only audit of saved route37 evidence. Never imports/runs supplied code.

Writes bounded golden restoration and independent JSON only below --output.
No subprocess, network, model, CUDA, shared-library loading, or timing benchmark.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import struct


COMMIT = "688683c71803ed94ac7a8e41e8e916d4e0174e6d"
N_REL = "archive/review-handoffs-not-for-readers/2026-11-17-route-observation"
H_REL = "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab"
MIB = 1024**2
GIB = 1024**3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jread(path: Path):
    return json.loads(path.read_text())


def need(condition, description: str):
    if not condition:
        raise AssertionError(description)


def manifest_audit(root: Path, wrapped: bool = False):
    manifest = jread(root / "FILES_SHA256.json")
    if wrapped:
        need(manifest["schema"] == 1, "incoming manifest schema")
        manifest = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    need(actual == set(manifest) | {"FILES_SHA256.json"}, f"manifest complete set {root}")
    for name, expected in manifest.items():
        p = root / name
        need(not p.is_symlink(), f"manifest symlink not accepted: {name}")
        need(p.resolve().is_relative_to(root.resolve()), f"manifest escapes root: {name}")
        data = p.read_bytes()
        need(len(data) == expected["bytes"], f"size {name}")
        need(sha(data) == expected["sha256"], f"digest {name}")
    return {"manifest_entries": len(manifest), "actual_files": len(actual),
            "manifest_sha256": sha((root / "FILES_SHA256.json").read_bytes()),
            "all_bytes_and_hashes_match": True, "no_missing_or_unlisted_files": True}


def apply_unified(base: str, patch: str) -> str:
    """Small independent exact-context unified-diff applicator; no shell patch."""
    old = base.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    need(lines[0].startswith("--- ") and lines[1].startswith("+++ "), "patch header")
    result = []
    cursor = 0
    i = 2
    while i < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        need(match is not None, "patch hunk header")
        start = int(match.group(1)) - 1
        old_count = int(match.group(2) or "1")
        new_count = int(match.group(4) or "1")
        need(start >= cursor, "patch hunk ordering")
        result.extend(old[cursor:start])
        cursor = start
        seen_old = seen_new = 0
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            line = lines[i]
            need(line[:1] in (" ", "+", "-"), "supported patch content")
            if line[0] in (" ", "-"):
                need(cursor < len(old) and old[cursor] == line[1:], "exact patch old context")
                cursor += 1
                seen_old += 1
            if line[0] in (" ", "+"):
                result.append(line[1:])
                seen_new += 1
            i += 1
        need((seen_old, seen_new) == (old_count, new_count), "patch hunk sizes")
    result.extend(old[cursor:])
    return "".join(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--previous", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    n = repo / N_REL
    h = repo / H_REL
    out = args.output.resolve()
    need(not out.is_relative_to(repo), "output must be outside read-only repo")
    out.mkdir(parents=True, exist_ok=True)
    prior = args.previous.resolve()
    incoming = n / "incoming" / prior.name
    report = {"schema": 1, "scope": "Independent saved-evidence consistency audit; not execution reproduction",
              "pinned_commit": COMMIT, "repo": str(repo), "handoff": N_REL,
              "supplied_code_executed": False, "shared_library_loaded": False,
              "model_API_or_GPU_used": False, "new_performance_measurement": False}

    report["package"] = manifest_audit(n)
    report["incoming"] = manifest_audit(incoming, wrapped=True)
    report["previous"] = manifest_audit(prior, wrapped=True)
    prior_names = {x.relative_to(prior).as_posix() for x in prior.rglob("*") if x.is_file()}
    incoming_names = {x.relative_to(incoming).as_posix() for x in incoming.rglob("*") if x.is_file()}
    need(prior_names == incoming_names, "prior/incoming file sets")
    for name in prior_names:
        need((prior / name).read_bytes() == (incoming / name).read_bytes(), "prior byte identity " + name)
    zip_path = prior.with_suffix(".zip")
    zip_data = zip_path.read_bytes()
    intake = jread(n / "INTAKE.json")
    need(sha(zip_data) == intake["zip_sha256"], "INTAKE previous ZIP hash")
    need(intake["verified_entries"] == report["incoming"]["manifest_entries"], "INTAKE manifest entry count")
    report["incoming"].update(prior_all_files_byte_identical=True,
                              prior_files_compared=len(prior_names), previous_zip_bytes=len(zip_data),
                              previous_zip_sha256=sha(zip_data), intake_zip_matches=True,
                              count_explanation="30 manifest-covered members plus manifest itself = 31 files")

    source = jread(n / "route37/SOURCE.json")
    base_cpu = h / "exploration-20261112/mx-prepack/release-candidate-a/sources/ggml-cpu.c"
    base_context = h / "deepseek41-bringup-20261101/source/src/llama-context.cpp"
    for name, f in (("base_cpu", base_cpu), ("base_context", base_context)):
        need(sha(f.read_bytes()) == source[name]["sha256"], "base source " + name)
    for name, digest in source["generated"].items():
        need(sha((n / "route37" / name).read_bytes()) == digest, "generated source " + name)
    for before, after, patch in ((base_cpu, "ggml-cpu.c", "cpu.patch"),
                                  (base_context, "llama-context.cpp", "caller.patch")):
        need(apply_unified(before.read_text(), (n / "route37" / patch).read_text()) ==
             (n / "route37" / after).read_text(), "full patch reconstruction " + after)
    cpu_diff = list(difflib.unified_diff(base_cpu.read_text().splitlines(),
                    (n / "route37/ggml-cpu.c").read_text().splitlines(), n=0))
    additions = [v[1:] for v in cpu_diff if v.startswith("+") and not v.startswith("+++")]
    removals = [v[1:] for v in cpu_diff if v.startswith("-") and not v.startswith("---")]
    need(not removals, "CPU prototype must only add hook/include")
    need(additions == ['#include "route37.h"', '',
                       '    // Diagnostic only: registered gates, one worker, before the existing barrier.',
                       '    if (ith == 0) xeon_route37_record_v1(src0, ids);'], "CPU exact additive diff")

    build = jread(n / "route37/BUILD.json")
    probe_build = jread(n / "route37/PROBE_BUILD.json")
    san = jread(n / "route37/sanitized/BUILD.json")
    for evidence in (build, san):
        for name, digest in evidence["source_sha256"].items():
            need(sha((n / "route37" / name).read_bytes()) == digest, "build source digest " + name)
    need(sha((n / "route37/probe.cpp").read_bytes()) == probe_build["source_sha256"], "probe digest")
    commands = build["commands"]
    need(len(commands) == 5, "five ordinary build commands")
    for command in commands[:2]:
        need("-c" in command and "-fno-fast-math" in command and "-ffp-contract=off" in command and
             "-mno-fma" in command, "changed CPU TU exact arithmetic flags")
    need("-shared" in commands[2] and "-Wl,--no-undefined" in commands[2], "diagnostic CPU shared link")
    need(sum(v.endswith("/route37.o") for v in commands[2]) == 1, "one route37 state in CPU link")
    need(all("-fsyntax-only" in x for x in commands[3:]), "caller syntax only")
    need(build["libllama_linked"] is False and build["full_model_executed"] is False, "build scope")
    need(len(san["commands"]) == 4, "sanitizer commands")
    need(all("-fsanitize=address,undefined" in cmd for cmd in san["commands"]), "ASAN/UBSAN flags")
    need(not any("thread" in flag for cmd in san["commands"] for flag in cmd if flag.startswith("-fsanitize=")), "not TSan")
    need(not any("-fsyntax-only" in cmd for cmd in san["commands"]), "sanitizer does not cover caller")
    # Independently reconstruct the recorded command deltas from archived baselines.
    baseline_build_path = h / "exploration-20261112/mx-prepack/release-candidate-a/BUILD.json"
    compile_table_path = h / "deepseek41-bringup-20261101/build/compile_commands.json"
    old_probe_build_path = repo / "archive/review-handoffs-not-for-readers/2026-11-16-abs-operator/PROBE_BUILD.json"
    baseline_build = jread(baseline_build_path)
    recorded_root = "/home/codex/projects/llm-lab/observation-20261117/route37"
    reconstructed = []
    for original, source_name, object_name in ((baseline_build["cpp"], "route37.cpp", "route37.o"),
            (baseline_build["variants"]["candidate"]["compile"], "ggml-cpu.c", "ggml-cpu.c.o")):
        c = list(original)
        c.insert(1, "-I" + recorded_root)
        c[c.index("-c") + 1] = recorded_root + "/" + source_name
        c[c.index("-o") + 1] = recorded_root + "/" + object_name
        reconstructed.append(c)
    old_obj = baseline_build["variants"]["candidate"]["compile"]
    old_obj = old_obj[old_obj.index("-o") + 1]
    c = [recorded_root + "/ggml-cpu.c.o" if arg == old_obj else
         "-Wl,--dependency-file=" + recorded_root + "/link.d" if arg.startswith("-Wl,--dependency-file=") else arg
         for arg in baseline_build["variants"]["candidate"]["link"]["argv"]]
    c[c.index("-o") + 1] = recorded_root + "/runtime/libggml-cpu.so.0.28.0"
    c.extend([recorded_root + "/route37.o", "-Wl,--no-undefined"])
    reconstructed.append(c)
    entries = [x for x in jread(compile_table_path) if x["file"].endswith("/llama-context.cpp")]
    need(len(entries) == 1, "unique original caller compile command")
    c = shlex.split(entries[0]["command"])
    for flag in ("-o", "-c"):
        i = c.index(flag)
        del c[i:i + 2]
    c.extend(["-I" + recorded_root, "-fsyntax-only"])
    reconstructed.extend([c + [recorded_root + "/" + name] for name in ("llama-context.cpp", "route37-bridge.cpp")])
    need(reconstructed == commands, "all five ordinary command deltas exactly match archived baselines")
    need(build["CPU_link_cwd"] == baseline_build["variants"]["candidate"]["link"]["cwd"], "CPU link cwd identity")
    old_probe_root = "/home/codex/projects/llm-lab/optimization-20261116/abs-operator"
    c = [arg.replace(old_probe_root, recorded_root) for arg in jread(old_probe_build_path)["argv"]]
    c.insert(1, "-I" + recorded_root)
    c.extend(["-pthread"])
    need(c == probe_build["argv"], "probe command delta matches archived predecessor")
    san_flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    rewritten = []
    mapping = {}
    for old in commands[:2]:
        c = list(old)
        target = c[c.index("-o") + 1]
        dest = recorded_root + "/sanitized/" + Path(target).name
        mapping[target] = dest
        c[c.index("-o") + 1] = dest
        rewritten.append(c + san_flags)
    c = ["-Wl,--dependency-file=" + recorded_root + "/sanitized/link.d" if arg.startswith("-Wl,--dependency-file=") else mapping.get(arg, arg) for arg in commands[2]]
    c[c.index("-o") + 1] = recorded_root + "/sanitized/libggml-cpu.so.0.28.0"
    rewritten.append(c + san_flags)
    c = list(probe_build["argv"])
    c[c.index("-o") + 1] = recorded_root + "/sanitized/probe"
    rewritten.append(c + san_flags)
    need(rewritten == san["commands"], "four sanitizer command deltas exactly match")
    report["build_and_sources"] = {
        "base_source_hashes_match_export": True, "generated_hashes_match": True,
        "full_cpu_and_context_reconstructed_from_patch": True,
        "CPU_delta": additions, "abs_candidate_added": False,
        "normal_build_commands": len(commands), "sanitizer_build_commands": len(san["commands"]),
        "all_normal_probe_sanitizer_command_deltas_match_prior_receipts": True,
        "normal_compiled_CPU_TUs": ["route37.cpp", "ggml-cpu.c"],
        "caller_syntax_only": ["llama-context.cpp", "route37-bridge.cpp"],
        "full_libllama_linked": False, "single_route37_object_in_recorded_CPU_link": True,
        "sanitized_translation_units": ["route37.cpp", "ggml-cpu.c", "probe.cpp"],
        "not_sanitized": "Reused old objects, libggml-base, caller integration; no TSan",
        "diagnostic_CPU_library_recorded_sha256": build["CPU_library_sha256"],
        "ELF_hashes_independently_recomputed": False,
        "ELF_limit": "No ELF/object files are in the handoff; source hashes and recorded commands are independently checked, binary digests remain saved receipts."}

    statuses = []
    for file in sorted((n / "execution").glob("*/STATUS.json")):
        status = jread(file)
        need(status["exit_code"] == 0 and status["error"] is None, "execution successful receipt " + file.parent.name)
        need(status["api_pid"] == 11151, "execution API PID")
        need((file.parent / "stderr.log").read_bytes() == b"", "empty stderr " + file.parent.name)
        need(status["samples"], "execution has saved samples")
        initial = status["initial"]
        samples = status["samples"]
        need(all(s["host_counters"] == initial["host_counters"] for s in samples), "saved counter continuity")
        need(all(s["available_kib"] >= 64 * 1024**2 for s in samples), "saved host reserve")
        need(all(s["cpu_max_C"] < 57 and s["api_cores"] <= 0.5 for s in samples), "saved CPU/API limits")
        need(abs(status["finished_unix"] - status["started_unix"] - status["elapsed_s"]) < 0.01, "receipt wall elapsed consistency")
        stdout = (file.parent / "stdout.log").read_text().strip()
        log_json = json.loads(stdout) if stdout.startswith("{") else None
        unique = sorted({s["unix"] for s in samples})
        statuses.append({
            "name": file.parent.name, "recorded_argv": status["argv"], "exit_code": status["exit_code"],
            "error": status["error"], "stderr_bytes": 0, "stdout": log_json or stdout,
            "started_unix": status["started_unix"], "finished_unix": status["finished_unix"],
            "guard_wall_elapsed_s_not_inference_latency": status["elapsed_s"],
            "sample_count": len(samples), "distinct_daemon_sensor_timestamps": len(unique),
            "sensor_timestamps_in_test_window": sum(status["started_unix"] <= t <= status["finished_unix"] for t in unique),
            "initial_sensor_age_s_at_start": status["started_unix"] - initial["unix"],
            "last_saved_sensor_age_s_at_finish": status["finished_unix"] - unique[-1],
            "saved_cpu_max_C": max(s["cpu_max_C"] for s in samples),
            "saved_MemAvailable_min_bytes": min(s["available_kib"] for s in samples) * 1024,
            "api_cores_max": max(s["api_cores"] for s in samples),
            "host_counter_fields": len(initial["host_counters"]), "saved_host_counters_unchanged": True,
        })
    need(len(statuses) == 6, "six guarded receipts")
    ordered = sorted(statuses, key=lambda x: x["started_unix"])
    need(all(a["finished_unix"] <= b["started_unix"] for a, b in zip(ordered, ordered[1:])), "six receipts non-overlapping")
    report["execution_receipts"] = statuses
    report["execution_limits"] = [
        "All six success receipts, empty stderr and corresponding command/source build entries agree; this audit did not reproduce their execution.",
        "Guard wall elapsed values include controller sampling/waiting; they are not kernel, token, route-observer, or end-to-end model latencies.",
        "Daemon sensor samples are repeated/stale within short tests. Several tests have no fresh in-run sensor timestamp; snapshots do not establish continuous peaks or continuous idleness.",
        "New handoff does not record the invoked guard script path/hash or its complete launch argv. Its schema agrees with the archived abs-operator guard; exact guard provenance is not independently established."]

    correctness = jread(n / "CORRECTNESS.json")
    expected_bytes = correctness["uncompressed_bytes"]
    need(expected_bytes == 17760 and expected_bytes <= 1024 * 1024, "bounded artificial golden size")
    with gzip.open(n / correctness["path"], "rb") as f:
        golden = f.read(expected_bytes + 1)
    need(len(golden) == expected_bytes, "bounded golden exact decompressed length")
    need(sha(golden) == correctness["uncompressed_sha256"], "golden uncompressed SHA")
    golden_path = out / "RESTORED_ARTIFICIAL_GOLDEN.bin"
    golden_path.write_bytes(golden)
    values = list(struct.unpack("<" + "f" * (len(golden) // 4), golden))
    need(all(math.isfinite(x) for x in values), "golden finite floats")
    probe = (n / "route37/probe.cpp").read_text()
    structural_markers = [
        "std::vector<float>v(37*24)", "for(int i=0;i<5;++i)", "for(int observe:{0,1})",
        "std::memcmp(v.data(),out.data(),v.size()*4)", "std::memcmp(out.data(),expected[i].data(),out.size()*4)",
        "ggml_new_tensor_3d(ctx,types[i%4],256,4,384)", "ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,1)",
        "const int phases[]={XR_WARMUP,XR_PREFILL,XR_UNKNOWN,XR_DECODE,XR_DECODE}",
        "for(int which=0;which<7;++which)", "std::thread worker", "ggml_threadpool_params_default(12)"]
    need(all(x in probe for x in structural_markers), "source-derived probe count/shape structure")
    golden_log = next(s["stdout"] for s in statuses if s["name"] == "route37-golden-20261117")
    test_logs = [s["stdout"] for s in statuses if s["name"] in ("route37-check-20261117", "route37-san-check-20261117")]
    per_graph = 37 * 6 * 4
    baseline_count = 5 * per_graph
    compare_count = 3 * baseline_count
    need(baseline_count == len(values) == golden_log["values"], "golden scalar count")
    need(all(log["golden_comparisons"] == compare_count == 13320 and log["graph_executions"] == 16 for log in test_logs), "test scalar comparisons and graph count")
    need(test_logs[0] == test_logs[1], "normal/sanitized test result scope agreement")
    report["correctness"] = {
        "golden_bytes": len(golden), "golden_sha256": sha(golden), "decompression_bounded": True,
        "restored_path": str(golden_path), "float32_count": len(values),
        "all_finite": True, "nonzero": sum(v != 0 for v in values), "minimum": min(values), "maximum": max(values),
        "fixture": {"nodes": 37, "K": 256, "M": 4, "experts": 384, "top_k": 6, "threads": 12,
                    "quantization_node_counts": {"IQ2_S": 10, "IQ3_S": 9, "IQ3_XXS": 9, "MXFP4": 9},
                    "inputs": 5, "weights": "artificial byte fixture, not exported model weights",
                    "phases": ["warmup", "prefill T1", "unknown T1", "decode", "decode"]},
        "source_derived_scalar_comparisons": compare_count,
        "formula": "5 inputs * 37 gates * 6 selected experts * 4 output values * 3 observation states = 13320",
        "states": ["diagnostic default-off", "session observed=0", "session observed=1"],
        "baseline_scalars": baseline_count, "actual_graph_runs_each_test": 5 + 2 * 5 + 1,
        "observed_decode_attempts_in_main_five_graph_sequence": 2,
        "negative_contract_categories": ["missing", "duplicate", "alias mapping", "record overflow", "invalid IDs",
                                         "graph failed", "unsynchronized completion", "attempt overflow", "failed attempt followed by retry"],
        "negative_categories_count": 9, "negative_categories_graph_executed": False,
        "strided_ID_test": "direct hook only", "multitoken_prefill_test": "metadata nt=2, actual artificial graph remains T1",
        "sanitizer_repeat_is_new_unique_fixture_set": False,
        "candidate_output_bytes_exported": False,
        "limits": ["memcmp is in saved probe source and success is in logs; candidate output buffers are not exported for an independent re-comparison.",
                   "The 13320 count is scalar output comparisons, not 13320 independent model examples or routes.",
                   "The exported golden is a frozen CPU artificial fixture reference, not FP64 mathematical ground truth or original-model quality.",
                   "ASAN/UBSAN success is limited to changed TUs/probe; no TSan, full graph/caller link, actual model coverage, latency or quality conclusion."]}

    capacity = jread(n / "CAPACITY_SNAPSHOT.json")
    previous_snapshot = incoming / "tools/snapshot_capacity.py"
    need(sha(previous_snapshot.read_bytes()) == capacity["script_sha256"], "capacity script source digest")
    memory = capacity["gpu"]["memory"]
    raw_memory = capacity["gpu"]["memory_raw"]
    for name in ("total", "used", "free", "reserved"):
        match = re.fullmatch(r"(\d+) MiB", raw_memory[name])
        need(match is not None and int(match.group(1)) * MIB == memory[name + "_bytes"], "NVML parsed field " + name)
    old_capacity_path = h / "exploration-20261112/hot-ffn/CAPACITY.json"
    old_capacity = jread(old_capacity_path)
    payload = min(x["expert_bytes"] for x in old_capacity["layers"] if x["layer"] <= 36)
    need(payload == 12072960 == capacity["capacity_screen"]["candidate_payload_bytes"], "minimum native triplet")
    free = memory["free_bytes"]
    pre = payload + 3072 * MIB
    need(pre == capacity["capacity_screen"]["candidate_preallocation_threshold_bytes"], "preallocation threshold")
    need((free > pre) is capacity["capacity_screen"]["reported_free_passes_candidate_memory_inequality"], "screen recomputed")
    hot_path = h / "exploration-20261112/hot-ffn/hot-backend.cpp"
    hot = hot_path.read_text()
    need('require(free>payload+size_t(3072)*1024*1024,' in hot and 'require(free>size_t(2560)*1024*1024,' in hot, "exact strict existing guards")
    need('if(payload){' in hot and 'size_t free,total;cuda_check(cudaMemGetInfo(&free,&total));\n        ready=true;' in hot, "zero payload still queries CUDA")
    need(capacity["capacity_screen"]["postallocation_guard_evaluated"] is False and
         capacity["capacity_screen"]["runtime_admission_or_launch_authorized"] is False, "no actual CUDA admission")
    host = capacity["host_memory"]["values"]["MemAvailable_bytes"]
    report["capacity"] = {
        "NVML_bytes": memory, "reported_MiB": {k: v // MIB for k, v in memory.items()},
        "total_minus_used_MiB": (memory["total_bytes"] - memory["used_bytes"]) / MIB,
        "used_plus_free_plus_reserved_minus_total_MiB": (sum(memory[k] for k in ("used_bytes", "free_bytes", "reserved_bytes")) - memory["total_bytes"]) / MIB,
        "minimum_eligible_native_payload_bytes": payload,
        "minimum_payload_layers": [x["layer"] for x in old_capacity["layers"] if x["layer"] <= 36 and x["expert_bytes"] == payload],
        "strict_pre_guard_threshold_bytes": pre, "strict_pre_guard_threshold_MiB": pre / MIB,
        "NVML_screen_shortfall_to_threshold_bytes": pre - free,
        "NVML_screen_shortfall_to_threshold_MiB": (pre - free) / MIB,
        "NVML_screen_positive_payload_passes": free > pre,
        "strict_pre_guard_pass_needs_more_than_shortfall_bytes": pre - free,
        "strict_post_guard_bytes": 2560 * MIB,
        "reported_current_free_minus_postguard_MiB_not_postallocation_success": (free - 2560 * MIB) / MIB,
        "CUDA_guard_actually_evaluated": False,
        "source_guard_operators": "free > payload + 3072 MiB; after warmup/synchronize free > 2560 MiB",
        "zero_payload_is_neutral_route_only": False,
        "host_MemAvailable_bytes": host, "host_MemAvailable_GiB": host / GIB,
        "host_available_above_64GiB_bytes": host - 64 * GIB,
        "host_available_above_64GiB_GiB": (host - 64 * GIB) / GIB,
        "raw_XML_exported": False, "raw_XML_hash_not_independently_recomputable": capacity["query"]["stdout_sha256"],
        "query_interval_ms": (capacity["query"]["end_monotonic_ns"] - capacity["query"]["begin_monotonic_ns"]) / 1e6,
        "conclusion": "Reject positive-hot planning at this coarse snapshot; preserve 3072/2560 MiB guards. This is not an actual CUDA allocation/admission verdict.",
        "limits": ["Use explicit free; total-used and per-process used do not replace it.",
                   "MiB fields are rounded/query interval values. Their one-MiB nonclosure is not evidence of exact accounting failure.",
                   "Current free minus 2560 is before allocations, not a postallocation reserve check.",
                   "Host MemAvailable is not per-process/cgroup allocation headroom."]}

    identity = jread(n / "IDENTITY_SNAPSHOT.json")
    final = jread(n / "FINAL_RUNTIME.json")
    need(identity["pid"] == final["API_pid"] == 11151 and identity["API_health"] == final["API_health"] == "ok", "PID/health receipts")
    need(identity["configuration"]["gpu_resident_expert_layers"] == [37, 38, 39], "config GPU placement")
    need(identity["environment"]["CUDA_VISIBLE_DEVICES"] == capacity["gpu"]["uuid"], "capacity/process CUDA UUID identity")
    process = identity["GPU_compute_process_query"]["processes"]
    need(len(process) == 1 and process[0]["pid"] == 11151 and process[0]["gpu_uuid"] == capacity["gpu"]["uuid"], "compute process receipt identity")
    plan = h / "exploration-20261112/plans/frequency19200.plan"
    need(sha(plan.read_bytes()) == identity["plan_sha256"], "exported current plan SHA")
    files = identity["mapped_file_identity"]
    selected = {}
    for path, entry in files.items():
        for lib in ("libggml-cpu", "libggml-cuda", "libggml-base", "libllama.so"):
            if Path(path).name.startswith(lib):
                selected[lib] = {"path": path, **entry}
    need(selected["libggml-cpu"]["sha256"] == "33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3", "frozen CPU hash receipt")
    need(selected["libggml-base"]["sha256"] == "2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558", "actual base hash receipt")
    capture = (n / "capture_identity.py").read_text()
    need("st.st_ino==int(fields[4])" in capture and "os.major(st.st_dev),os.minor(st.st_dev)" in capture, "capture performs maps/stat inode/device assertions")
    for key in ("same_PID", "runtime_file_hashes_unchanged", "original_sources_unchanged", "config_and_plan_hashes_unchanged"):
        need(final[key] is True, "final unchanged receipt " + key)
    for key in ("API_restarted", "model_loaded_for_this_study", "CUDA_allocated_for_this_study"):
        need(final[key] is False, "final no action receipt " + key)
    report["runtime_identity"] = {
        "PID": identity["pid"], "boot_id": identity["boot_id"], "start_ticks": identity["start_ticks_since_boot"],
        "config_sha256": identity["config_sha256"], "plan_sha256_matches_exported_plan": True,
        "configured_GPU_layers": identity["configuration"]["gpu_resident_expert_layers"],
        "selected_mapped_backing_files": selected,
        "all_mapped_file_records": len(files), "maps_stat_device_and_inode_assertions_in_source": True,
        "saved_raw_maps_or_device_pairs": False,
        "hash_scope": identity["mapped_file_hash_scope"],
        "device_used_MiB": memory["used_bytes"] / MIB, "process_used_MiB": int(process[0]["used_MiB_raw"]),
        "device_used_minus_process_used_MiB": memory["used_bytes"] / MIB - int(process[0]["used_MiB_raw"]),
        "capacity_to_identity_receipt_seconds": identity["unix"] - capacity["clock_after"]["unix_ns"] / 1e9,
        "identity_to_final_seconds": final["unix"] - identity["unix"],
        "services_active_in_both_receipts": identity["services"] == final["services"] and all(v == "active" for v in final["services"].values()),
        "final_monitor_age_s": final["unix"] - final["monitor_unix"],
        "final_runtime_checks_are_boolean_receipts": True,
        "not_independently_verified": ["live remote process memory or file inode/device at audit time", "original library binary hashes (ELFs not exported)",
                                      "final full mapping/start-tick/hash list (only saved booleans)", "continuous absence of API requests or process changes"],
        "fan_scope": "70% is preserved policy; no fan write; monitor supplies RPM/duty fields, no fresh independent duty query.",
        "next_build_provenance_need": "Future full libllama/driver receipt should record build/link commands and loaded CPU/provider identity, ensuring a single route37 state; current syntax-only receipt cannot establish this."}

    report["material_gaps_for_next_decision"] = [
        "No complete diagnostic libllama/driver link or runtime caller/provider identity proof. Must establish before model route capture.",
        "No actual-model K5120/M2304 coverage or real scheduler/graph-reuse route capture. Artificial K256/M4 byte equivalence does not prove 37-layer capture.",
        "No real route distribution, observer overhead, CPU/GPU overlap, first/streamed-token intervals or quality evidence. Do not promote these probe counts into an optimization gain.",
        "Current coarse free memory cannot support positive hot under unchanged guards. Ready/event experiments with positive payload require a separately reviewed memory plan; no zero-payload substitute."]
    report["receipt_improvements_not_reasons_to_repeat_existing_suite"] = [
        "Future guard receipt: guard source hash and full launcher argv; distinguish sample collection time from daemon sensor time.",
        "Future identity receipt: save filtered maps device/inode + opened-file stat identity and final full start/boot/hash receipt; current data only partially allows independent backing-file recheck.",
        "Future capacity query: retain bounded selected raw XML/subtree or machine-readable raw query result, because the current full-XML hash alone is not independently recomputable."]
    report["audit_script_sha256"] = sha(Path(__file__).read_bytes())
    report["all_independent_checks_passed"] = True
    (out / "ROUTE_EVIDENCE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    source_ledger = {}
    for file in sorted(n.rglob("*")):
        if file.is_file():
            source_ledger[file.relative_to(repo).as_posix()] = {"bytes": file.stat().st_size, "sha256": sha(file.read_bytes())}
    for file in (base_cpu, base_context, baseline_build_path, compile_table_path, old_probe_build_path, old_capacity_path, hot_path, plan,
                 repo / "archive/review-handoffs-not-for-readers/2026-11-16-abs-operator/guard.py"):
        source_ledger[file.relative_to(repo).as_posix()] = {"bytes": file.stat().st_size, "sha256": sha(file.read_bytes())}
    (out / "ROUTE_EVIDENCE_SOURCE_SHA256.json").write_text(json.dumps({"schema": 1, "commit": COMMIT, "files": source_ledger}, indent=2) + "\n")
    print(json.dumps({"checks_passed": True, "package_files": report["package"]["actual_files"],
                      "previous_files_identical": len(prior_names), "execution_receipts": len(statuses),
                      "scalar_comparisons": compare_count, "restored_golden_sha256": sha(golden),
                      "NVML_screen_shortfall_MiB": (pre-free)/MIB, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
