#!/usr/bin/env python3
"""Offline fixed-token comparisons. Reads saved files only; never loads a model or calls an API."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

VOCAB = 129280
QUALITY_ROWS = 65
DRIVER_SHA256 = "b32c3777afd478e0a726f898795c3bcbe14005c69fd4f116083c87520f646994"
CASES_SHA256 = "0458b7a358d0eb3a0c289eb1a1aa9903ee0bb972158acc64abbf0952f472d746"
GOLD_SHA256 = {
    "text_cache_budget_ja.f32": "35d028fed9fa08bbe8f304c598ced362ff466aff176ecf407133c0a1870bbd64",
    "code_stable_grouping_python.f32": "077eefd33928bfbec340162c5b5e54c2bbad9e4ac3ae72bc23f38a7a33d69d01",
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floats(raw):
    result = array.array("f")
    need(result.itemsize == 4, "This interpreter does not use 4-byte array float")
    result.frombytes(raw)
    if sys.byteorder != "little":
        result.byteswap()
    need(all(math.isfinite(x) for x in result), "Non-finite saved F32 value")
    return result


def compare_f32(reference, candidate, rows, vocab=VOCAB, reference_tail=False):
    """Compare full rows, or a single timing row against the quality file's last row."""
    expected = rows * vocab * 4
    need(candidate.stat().st_size == expected, "Candidate F32 size mismatch: " + str(candidate))
    need(reference.stat().st_size == (QUALITY_ROWS if reference_tail else rows) * vocab * 4,
         "Reference F32 size mismatch: " + str(reference))
    details = []
    with reference.open("rb") as a, candidate.open("rb") as b:
        if reference_tail:
            need(rows == 1, "A timing comparison has one saved row")
            a.seek(-expected, 2)
        for row in range(rows):
            raw_a, raw_b = a.read(vocab * 4), b.read(vocab * 4)
            need(len(raw_a) == len(raw_b) == vocab * 4, "Incomplete F32 row")
            va = floats(raw_a)
            equal = raw_a == raw_b
            vb = va if equal else floats(raw_b)
            energy = math.fsum(x * x for x in va)
            error = 0.0 if equal else math.fsum((x-y) ** 2 for x, y in zip(va, vb))
            maximum = 0.0 if equal else max(abs(x-y) for x, y in zip(va, vb))
            top_a = max(range(vocab), key=va.__getitem__)
            top_b = top_a if equal else max(range(vocab), key=vb.__getitem__)
            details.append({"row": QUALITY_ROWS-1 if reference_tail else row,
                            "byte_equal": equal, "max_abs": maximum,
                            "squared_error": error, "reference_energy": energy,
                            "nmse": error / energy if energy else None,
                            "reference_argmax": top_a, "candidate_argmax": top_b})
    energy = math.fsum(r["reference_energy"] for r in details)
    error = math.fsum(r["squared_error"] for r in details)
    return {"shape": [rows, vocab], "finite": True,
            "byte_equal": all(r["byte_equal"] for r in details),
            "argmax_changes": sum(r["reference_argmax"] != r["candidate_argmax"] for r in details),
            "max_abs": max(r["max_abs"] for r in details),
            "nmse": error / energy if energy else None,
            "reference_file_sha256": sha(reference), "candidate_file_sha256": sha(candidate),
            "reference_is_last_row_only": reference_tail, "per_row": details}


def metrics(durations):
    need(bool(durations) and all(type(x) is int and x > 0 for x in durations), "Invalid durations")
    n = len(durations)
    return {"N": n, "mean_ms": sum(durations) / n / 1e6,
            "median_ms": statistics.median(durations) / 1e6,
            "p95_ms_nearest_rank": sorted(durations)[math.ceil(n * .95)-1] / 1e6,
            "max_ms": max(durations) / 1e6,
            "T1_per_second": n * 1e9 / sum(durations)}


def load_run(directory, stage, cases):
    run, spec, launch = (read(directory / f) for f in ("RUN.json", "SPEC.json", "LAUNCH.json"))
    need(run.get("passed") is True and run["spec"] == spec, "Run failed or SPEC differs")
    need(spec["stage"] == stage and spec["cases"] in ([0, 1], [1, 0]), "Unexpected stage/order")
    rows = QUALITY_ROWS if stage == "quality" else 1
    need(spec["rows"] == rows and len(run["blocks"]) == 2, "Unexpected row/block count")
    condition = spec["condition"]
    need(run["requested"] == condition, "Requested condition mismatch")
    need(run["linux_entry"]["effective_cpu_ids"] == condition["process_cpu_ids"], "Process CPU allowance mismatch")
    need(run["parsed_cpu"] == {
        "decode_count": condition["decode_threads"], "prefill_count": condition["prefill_threads"],
        "decode_ids": condition["decode_cpu_ids"], "prefill_ids": condition["prefill_cpu_ids"],
        "decode_strict": True, "prefill_strict": True}, "Parsed CPU settings mismatch")
    need(run["context"]["threads"] == condition["decode_threads"] and
         run["context"]["threads_batch"] == condition["prefill_threads"] and
         run["projector_init_threads_argument"] == condition["projector_threads"], "Actual thread counts mismatch")
    need(launch["driver_sha256"] == DRIVER_SHA256, "Different benchmark driver")
    need(launch["spec"] == spec and launch["cases_sha256"] == CASES_SHA256, "Launch spec/cases mismatch")
    need(run["compact_bytes"] == 127401984000 and run["compact_bytes_after_unload"] == 0 and
         run["after_unload_KiB"]["VmLck"] == 0, "Compact registration/unload mismatch")
    need([v["routed_host"] for v in run["placement"]] == [True]*37+[False]*3, "Routed placement changed")
    result = {}
    for block, index in zip(run["blocks"], spec["cases"]):
        case = cases[index]
        name = case["case_id"]
        need(block["case_index"] == index and block["case_id"] == name, "Case identity mismatch")
        need(block["prefix_ids"] == case["prefix_ids"] and block["positions_begin"] == case["prefix_count"] and
             block["input_tokens"] == case["used_T1_ids"], "Fixed tokens/positions changed")
        need(block["file"] == name + ".f32" and block["rows"] == rows, "F32 filename/rows mismatch")
        need(block["status"] == [0]*64 and block["traced_calls"] == 0, "Decode status or tracing changed")
        starts, ends = block["start_ns"], block["output_ready_ns"]
        need(len(starts) == len(ends) == 64 and all(type(t) is int for t in starts + ends), "Bad clock arrays")
        need(block["prefill_ready_ns"] > block["prefill_begin_ns"] and starts[0] > block["prefill_ready_ns"], "Bad prefill clock")
        need(all(a < b for a, b in zip(starts, ends)) and
             all(a > b for a, b in zip(starts[1:], ends[:-1])), "Non-monotonic per-call clocks")
        durations = [b-a for a, b in zip(starts, ends)]
        result[name] = {"file": directory / block["file"], "durations_ns": durations,
                        "prefill_s": (block["prefill_ready_ns"]-block["prefill_begin_ns"])/1e9}
    return {"directory": str(directory), "run": run, "spec": spec, "launch": launch, "blocks": result}


def cohort(quality, timings):
    need(len(timings) == 2 and [r["spec"]["cases"] for r in timings] == [[0, 1], [1, 0]],
         "Expected timing processes in JA/Python then Python/JA order")
    need(all(r["spec"]["condition"] == quality["spec"]["condition"] for r in timings),
         "Timing data belongs to a different condition than its quality reference")
    need(len({r["run"]["linux_entry"]["pid"] for r in [quality] + timings}) == 3,
         "Expected three independent model processes")
    per_case = {}
    tails = {}
    for name, gold in quality["blocks"].items():
        groups = [r["blocks"][name]["durations_ns"] for r in timings]
        per_case[name] = {**metrics([t for group in groups for t in group]),
                          "per_process": [metrics(group) for group in groups],
                          "prefill_mean_s": statistics.fmean(r["blocks"][name]["prefill_s"] for r in timings)}
        for r in timings:
            key = r["spec"]["name"] + "/" + name
            tails[key] = compare_f32(gold["file"], r["blocks"][name]["file"], 1, reference_tail=True)
    return {"cases": per_case, "timing_tail_comparisons": tails,
            "all_timing_tails_byte_equal": all(v["byte_equal"] for v in tails.values()),
            "timing_processes": 2, "timed_T1_calls": 256,
            "sampling_note": "128 serial calls per case across two processes; not 128 independent experiments"}


def compare(reference_quality, candidate_quality, timing_runs, cases_path, reference_timing_runs=None):
    need(sha(cases_path) == CASES_SHA256, "Frozen CASES hash mismatch")
    cases = read(cases_path)["cases"]
    need(len(cases) == 2, "Expected two fixed cases")
    old = load_run(reference_quality, "quality", cases)
    new = load_run(candidate_quality, "quality", cases)
    need(old["spec"]["condition"]["name"] == "OLD_P12", "Reference is not OLD_P12 quality")
    for block in old["blocks"].values():
        need(sha(block["file"]) == GOLD_SHA256[block["file"].name], "Pinned OLD_P12 quality changed")
    # This first comparison intentionally fixes the engine. Changed kernels/plans need a separate protocol.
    old_libraries = {k: v["sha256"] for k, v in old["launch"]["libraries"].items()}
    for r in [new] + [load_run(p, "timing", cases) for p in timing_runs]:
        need({k: v["sha256"] for k, v in r["launch"]["libraries"].items()} == old_libraries,
             "Runtime libraries changed; this is not the fixed-engine hardware comparison")
        need(r["launch"]["plan_sha256"] == old["launch"]["plan_sha256"], "Compact plan changed")
    cross = {name: compare_f32(old["blocks"][name]["file"], block["file"], QUALITY_ROWS)
             for name, block in new["blocks"].items()}
    timed = [load_run(p, "timing", cases) for p in timing_runs]
    observed = cohort(new, timed)
    exact = all(x["byte_equal"] for x in cross.values()) and observed["all_timing_tails_byte_equal"]
    output = {"status": "exact_on_saved_outputs" if exact else "needs_numerical_review",
              "scope": "Offline saved-output and fixed-token timing comparison; does not replace hardware guards or provenance review",
              "self_comparison": reference_quality.resolve() == candidate_quality.resolve(),
              "reference_condition": old["spec"]["condition"], "candidate_condition": new["spec"]["condition"],
              "full_quality_comparisons": cross, "candidate_timing": observed,
              "finite_scope": "Quality all 65 rows per case and timing final row only; intermediate timing logits were not saved",
              "acceptance_note": "No arbitrary numerical tolerance is used. Nonidentical finite data is retained for diagnosis, not silently accepted."}
    if reference_timing_runs is not None:
        old_timings = [load_run(p, "timing", cases) for p in reference_timing_runs]
        old_cohort = cohort(old, old_timings)
        need(old_cohort["all_timing_tails_byte_equal"], "OLD_P12 timing tails differ from pinned quality")
        output["reference_timing"] = old_cohort
        output["throughput_ratio_candidate_over_reference"] = {
            name: observed["cases"][name]["T1_per_second"] / old_cohort["cases"][name]["T1_per_second"]
            for name in observed["cases"]}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-quality", type=Path, required=True)
    parser.add_argument("--candidate-quality", type=Path, required=True)
    parser.add_argument("--timing-run", type=Path, action="append", required=True)
    parser.add_argument("--reference-timing-run", type=Path, action="append")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    need(not args.out.exists(), "Output already exists")
    try:
        result = compare(args.reference_quality, args.candidate_quality, args.timing_run,
                         args.cases, args.reference_timing_run)
        exitcode = 0 if result["status"] == "exact_on_saved_outputs" else 2
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        result = {"status": "invalid_or_incomplete_input", "error": str(error)}
        exitcode = 1
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.out), "exitcode": exitcode}))
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
