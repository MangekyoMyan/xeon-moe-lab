"""Summarize CPU split intervals and CUPTI device activity in a saved trace169 capture."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def require(test, message):
    if not test:
        raise ValueError(message)


def union_ns(intervals):
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def summarize(directory):
    files = {}
    counts = {}

    def load(name):
        path = directory / ("trace169-" + name)
        with path.open("rb") as handle:
            files[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        with path.open() as handle:
            if name.endswith(".json"):
                return json.load(handle)
            result = list(csv.DictReader(handle, delimiter="\t"))
        counts[name] = len(result)
        return result

    receipt = load("summary.json")
    for field in ("unknown_kernel_correlation", "foreign_thread_calls", "dropped_inside",
                  "unknown_records", "errors"):
        require(receipt[field] == 0, f"Observer reported {field}")
    require(receipt["schema"] == 1 and receipt["cupti_version"] == 28, "Expected trace169 / CUPTI 28")
    calls = {}
    cases = set()
    intervals = defaultdict(lambda: defaultdict(list))
    counts_device = defaultdict(Counter)
    transfer_bytes = defaultdict(Counter)
    rows = load("calls.tsv")
    for row in rows:
        key = int(row["call"])
        case = (int(row["case_id"]), int(row["call_id"]))
        begin, end = int(row["start_ns"]), int(row["end_ns"])
        require(key not in calls and case not in cases and 0 < begin < end, "Invalid T1 identity or span")
        calls[key] = (case, begin, end, int(row["n_splits"]))
        cases.add(case)
    require(cases == {(c, t) for c in (0, 1) for t in range(1, 5)}, "Expected four T1 per case")
    ordered = sorted((begin, end) for _, begin, end, _ in calls.values())
    require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), "Overlapping T1 windows")

    def add(call, label, row, begin_key, end_key):
        _, begin, end, _ = calls[call]
        a, b = int(row[begin_key]), int(row[end_key])
        require(begin <= a <= b <= end, f"{label} event outside correlated T1")
        intervals[call][label].append((a, b))

    # External IDs in trace169.cpp are either a split, the enclosing T1 or a host copy.
    tag = 1 << 32
    external_calls = {tag + number: number for number in calls}
    split_ordinals = defaultdict(set)
    for row in load("splits.tsv"):
        split, call, ordinal = int(row["split"]), int(row["call"]), int(row["ordinal"])
        require(0 < split < tag and split not in external_calls and ordinal not in split_ordinals[call],
                "Invalid split identity")
        external_calls[split] = call
        split_ordinals[call].add(ordinal)
        require(int(row["status"]) == 0 and row["backend"] in ("CPU", "CUDA0"), "Unexpected backend/status")
        add(call, row["backend"], row, "dispatch_ns", "return_ns")
        add(call, "metadata", row, "metadata_start_ns", "dispatch_ns")
    for number, (_, _, _, n_splits) in calls.items():
        require(split_ordinals[number] == set(range(n_splits)), "Incomplete split ordinals")
    for row in load("copies.tsv"):
        copy, call = int(row["copy"]), int(row["call"])
        ext = 2 * tag + copy
        require(0 < copy < tag and ext not in external_calls, "Invalid host copy identity")
        external_calls[ext] = call
        add(call, "host_copy", row, "start_ns", "return_ns")
    correlations = {}
    for row in load("correlations.tsv"):
        cid, ext = int(row["correlation"]), int(row["external_id"])
        require(int(row["external_kind"]) == 3 and ext in external_calls, "Unrecognized external ID")
        require(correlations.setdefault(cid, ext) == ext, "Conflicting CUPTI correlation")
    device_contexts = set()
    for filename, label in [("kernels.tsv", "kernel"), ("device-copies.tsv", "dma")]:
        for row in load(filename):
            ext = correlations[int(row["correlation"])]
            call = external_calls[ext]
            add(call, label, row, "start_ns", "end_ns")
            device_contexts.add((int(row["device"]), int(row["context"])))
            counts_device[call][label] += 1
            if label == "dma":
                # Keep the raw CUpti_ActivityMemcpyKind ID. 1 HtoD, 2 DtoH, 8 DtoD.
                direction = row["direction"]
                byte_count = int(row["bytes"])
                require(byte_count >= 0, "Invalid device copy size")
                transfer_bytes[call][direction] += byte_count
    require(len(device_contexts) == 1, "Expected one GPU/context")
    for name, field in [("calls", "calls"), ("splits", "splits"), ("copies", "host_copies"),
                        ("kernels", "kernels"), ("device-copies", "device_copies"),
                        ("correlations", "correlations")]:
        require(counts[name + ".tsv"] == receipt[field], f"Count differs from observer receipt: {name}")

    result = []
    for call, ((case, step), start, end, _) in sorted(calls.items()):
        group = intervals[call]
        cpu, kernel, dma = (group[k] for k in ("CPU", "kernel", "dma"))
        require(cpu and kernel, "Missing CPU or CUDA timeline")
        cpu_ns, kernel_ns = union_ns(cpu), union_ns(kernel)
        device_ns = union_ns(kernel + dma)
        covered = union_ns(cpu + kernel + dma)
        result.append({
            "case_id": case, "call_id": step, "span_ns": end - start,
            "cpu_split_span_union_ns": cpu_ns, "device_kernel_union_ns": kernel_ns,
            "device_copy_union_ns": union_ns(dma), "device_kernel_or_copy_union_ns": device_ns,
            "cpu_split_and_kernel_ns": cpu_ns + kernel_ns - union_ns(cpu + kernel),
            "cpu_split_and_device_ns": cpu_ns + device_ns - covered,
            "outside_cpu_split_and_all_device_activity_ns": end - start - covered,
            "cuda_dispatch_host_union_ns": union_ns(group["CUDA0"]),
            "host_copy_call_union_ns": union_ns(group["host_copy"]),
            "observer_split_metadata_union_ns": union_ns(group["metadata"]),
            "device_activity_count": dict(counts_device[call]),
            "device_copy_bytes_by_cupti_direction": dict(sorted(transfer_bytes[call].items())),
        })
    return {"schema": 1, "scope": "CUPTI-instrumented T1, one process/GPU/context, two cases x first four calls",
            "meaning": "CPU is synchronous CPU graph dispatch wall time including barriers. Device is observed kernels/memcpy. Overlap is temporal coincidence. Outside means unclassified; it is not stall or reclaimable time.",
            "host_scopes": "Host CUDA dispatch, host copy, and observer metadata can overlap device activity; do not add them to GPU durations.",
            "input_sha256": dict(sorted(files.items())), "calls": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace_directory), indent=2, allow_nan=False))
