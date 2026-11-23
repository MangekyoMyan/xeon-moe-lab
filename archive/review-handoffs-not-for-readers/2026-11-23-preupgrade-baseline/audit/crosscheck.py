"""Compare the independent replay to the study's summaries and bounded quality telemetry."""

import bisect
import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def confirm_summaries(root, replay):
    trial = root / "trial"
    state = load(trial / "STATE.json")
    timing = load(root / "TIMINGS.json")
    fields = ["N", "mean_ms", "median_ms", "p95_ms_nearest_rank", "max_ms", "T1_per_second"]
    for name, result in timing["cases"].items():
        independent = replay["timing_combined_by_case"][name]
        require(all(result[key] == independent[key] for key in fields + ["process_mean_range_ms"]), "pooled stats")
    for proc in timing["processes"]:
        run = trial / "runs" / proc["name"]
        report = load(run / "RUN.json")
        for block in report["blocks"]:
            name = block["case_id"]
            independent = replay["timing_per_process"][proc["name"]][name]
            published = proc["cases"][name]
            require(all(published[k] == independent[k] for k in fields), "per-process stats")
            require(published["durations_ns"] == [e - b for b, e in zip(block["start_ns"], block["output_ready_ns"], strict=True)]
                    and published["prefill_s"] == independent["prefill_seconds"], "all raw durations")
            tail = load(run / "TAIL_COMPARISON.json")[block["file"]]
            require(tail["byte_equal"] == replay["timing_tails_vs_OLD_P12_quality"][run.name][name]["byte_equal"],
                    "tail summary")
    quality = load(trial / "QUALITY_COMPARISON.json")["cases"]
    for name, independent in replay["quality_vs_Nov19_normal_12_24"].items():
        result = quality[name + ".f32"]
        pairs = [("byte_equal", "byte_equal"), ("max_abs_error", "max_abs_error"), ("nmse", "NMSE"),
                 ("argmax_changes", "argmax_changes"), ("old_p12_sha256", "observed_sha256"),
                 ("historical_sha256", "reference_file_sha256")]
        require(all(result[key] == independent[other] for key, other in pairs), "quality summary")
    manifests = {}
    for name, stage in state["runs"].items():
        run = trial / "runs" / name
        manifest = run / "RESULTS_SHA256.json"
        entries = load(manifest)
        require(sha(manifest) == stage["manifest_sha256"], "run manifest identity")
        require(set(entries) == {p.name for p in run.iterdir() if p.is_file() and p != manifest}, "manifest coverage")
        require(all(sha(run / key) == value for key, value in entries.items()), "run file digest")
        manifests[name] = {"entries": len(entries), "sha256": sha(manifest)}
    return manifests


def sample_evidence(root):
    run = root / "trial/runs/old-p12-quality-01"
    report = load(run / "RUN.json")
    pid = report["linux_entry"]["pid"]
    cpu_lib = load(run / "LAUNCH.json")["libraries"]["libggml-cpu.so.0.28.0"]["path"]
    block = report["blocks"][0]
    starts, ends = block["start_ns"], block["output_ready_ns"]
    expect = set(report["requested"]["process_cpu_ids"])
    per_call = [set() for _ in starts]
    cores = defaultdict(set)
    count, prefill, prefill_cpu_dso, decode = 0, 0, 0, 0
    with (root / "quality-host/worker-samples.tsv").open() as inp:
        for row in csv.DictReader(inp, delimiter="\t"):
            require(int(row["pid"]) == pid, "PC target")
            count += 1
            ts, cpu, tid = (int(row[k]) for k in ("monotonic_ns", "cpu", "tid"))
            require(cpu in expect, "PC sampled outside launch CPU allowance")
            if block["prefill_begin_ns"] <= ts < block["prefill_ready_ns"]:
                prefill += 1
                prefill_cpu_dso += row["dso"] == cpu_lib
            index = bisect.bisect_right(starts, ts) - 1
            if index < 0 or ts >= ends[index]:
                continue
            decode += 1
            if row["dso"] == cpu_lib and ("compact_dot" in row["symbol"] or "ggml_vec_dot" in row["symbol"]):
                per_call[index].add(tid)
                cores[tid].add(cpu)
    pc_receipt = load(root / "quality-host/PC_DECODE.json")
    require(count == pc_receipt["kept"] == pc_receipt["input_sample_lines"] and pc_receipt["unparsed_rows"] == 0,
            "PC TSV source coverage")
    main = load(root / "QUALITY_HOST_EVIDENCE.json")["phases"]
    require(prefill == main["prefill"]["sampled_pid_pc_count"] and decode == main["decode"]["sampled_pid_pc_count"],
            "PC timestamps in model intervals")
    return {"PC_rows_in_export": count, "PC_rows_inside_JA_prefill": prefill, "prefill_PC_rows_in_pinned_CPU_DSO": prefill_cpu_dso,
            "PC_rows_inside_64_JA_decode_calls": decode, "decode_dot_TIDs_and_observed_CPUs": {k: sorted(v) for k, v in cores.items()},
            "calls_with_12_distinct_TIDs_in_compact_dot_or_ggml_vec_dot": sum(len(tids) == 12 for tids in per_call),
            "minimum_distinct_dot_TIDs_in_a_decode_call": min(map(len, per_call)), "exact_per_graph_team_observed": None}


def clock_evidence(root):
    groups = defaultdict(dict)
    with (root / "quality-host/msr.csv").open() as inp:
        for row in csv.reader(inp, delimiter=";"):
            if len(row) < 7:
                continue
            group = groups[(Decimal(row[0]), int(row[1].removeprefix("CPU")))]
            key = row[4].split("/")[1]
            require(key not in group and row[2].isdigit() and Decimal(row[6]) == 100, "MSR row validity")
            group[key] = (int(row[2]), int(row[5]))
    active_ns, aperf = Decimal(0), Decimal(0)
    for group in groups.values():
        require(set(group) == {"aperf", "mperf", "tsc"} and all(v[0] > 0 and v[1] > 0 for v in group.values()),
                "complete nonzero per-CPU MSR interval")
        active_ns += Decimal(group["mperf"][0]) * group["tsc"][1] / group["tsc"][0]
        aperf += group["aperf"][0]
    rate = float(1000 * aperf / active_ns)
    main = load(root / "QUALITY_HOST_EVIDENCE.json")["effective_clock"]
    require(len(groups) == main["usable_cpu_intervals"] and math.isclose(rate, main["busy_MHz_estimated"], rel_tol=1e-13),
            "active clock aggregation")
    return {"interval_timestamps": len({t for t, c in groups}), "selected_CPUs": len({c for t, c in groups}),
            "complete_per_CPU_intervals": len(groups), "active_time_weighted_busy_MHz": rate}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    replay = load(root / "audit/P12_REPLAY.json")
    print(json.dumps({"matching_run_manifests": confirm_summaries(root, replay),
                      "timing_and_quality_summaries_agree": True, "quality_PC": sample_evidence(root),
                      "quality_cpu_wide_clock": clock_evidence(root)}, indent=2))
