#!/usr/bin/env python3
"""Prepare the pinned IQ2 mixed-cache diagnostic from saved evidence only.

Standard library only. Reads the repository without modifying it. Does not load
model weights, execute a library, call an API, or run inference or a benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import struct
import sys


COMMIT = "31b1292b2767618a28ef6e3f1c9c4450b1fc21be"
N = "archive/review-handoffs-not-for-readers/2026-11-16-abs-operator"
E = ("archive/review-handoffs-not-for-readers/2026-11-16-current/"
     "snapshot/llm-lab/exploration-20261112")
TENSOR = "blk.0.ffn_up_exps.weight"
K, M, EXPERTS, USED, GROUPS, INPUTS = 5120, 2304, 384, 6, 8, 1
INPUT_BYTES = GROUPS * INPUTS * K * 4
EXPERT_BYTES = (K // 256) * M * 144
INPUT_SHA = "9dc5705e04389d3d08d894ddf25ce675fa2ff3bed73eccc4ba9a156c8028ec93"
PLAN_SHA = "cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6"
N_MANIFEST_SHA = "f4c17f54a5b31e486d5762e2fed7fa30252264cfea3f8f3e34c95aa691a4f163"
PLAN_MANIFEST_SHA = "7f3a4b8c54adc6355ba6093eb42b9f6c0898504dc3f47fcdf1154cfab01256f1"
EXPECTED_COUNTS = [5, 5, 5, 5, 3, 5, 6, 5]
EXPECTED_MASKS = [62, 59, 55, 47, 41, 62, 63, 31]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def json_data(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data.decode("utf-8"), parse_constant=invalid)


def read_source(repo, relative, expected_sha, sources, expected_bytes=None,
                max_bytes=2 * 1024 * 1024):
    path = (repo / relative).resolve(strict=True)
    require(inside(path, repo) and path.is_file(), "Source path escapes repository: " + relative)
    size = path.stat().st_size
    require(size <= max_bytes, "Source exceeds bounded input limit: " + relative)
    if expected_bytes is not None:
        require(size == expected_bytes, "Source byte count mismatch: " + relative)
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    require(len(data) == size and len(data) <= max_bytes, "Source changed or exceeded limit: " + relative)
    digest = sha256(data)
    require(digest == expected_sha, "Source SHA256 mismatch: " + relative)
    sources.append({"repo_relative_path": relative, "sha256": digest, "bytes": len(data)})
    return data


def make_header(routes, selected, masks):
    def array(name, values):
        lines = []
        for i in range(0, len(values), 8):
            lines.append("    " + ", ".join(str(x) for x in values[i:i + 8]) + ",")
        return ("inline constexpr int " + name + "[" + str(len(values)) + "] = {\n"
                + "\n".join(lines) + "\n};\n")
    header = ["#pragma once\n",
              "// Saved synthetic inputs / route IDs only; no model weights.\n",
              "// Cplan is a cache-membership diagnostic, not typical API traffic.\n",
              "namespace xeon_mixed_case {\n"]
    for name, value in (("K", K), ("M", M), ("EXPERTS", EXPERTS), ("USED", USED),
                        ("GROUPS", GROUPS), ("INPUTS", INPUTS)):
        header.append("inline constexpr int " + name + " = " + str(value) + ";\n")
    header.extend([array("routes", routes), array("cache_ids", selected), array("masks", masks)])
    for name, value in (("commit", COMMIT), ("tensor_name", TENSOR),
                        ("inputs_sha256", INPUT_SHA), ("plan_sha256", PLAN_SHA)):
        header.append("inline constexpr const char* " + name + " = " + json.dumps(value) + ";\n")
    header.append("} // namespace xeon_mixed_case\n")
    return "".join(header).encode("utf-8")


def prepare(repo, output):
    repo = repo.resolve(strict=True)
    output = output.resolve()
    require(repo.is_dir(), "Repository must be a directory")
    require(not inside(output, repo), "Output must be outside the read-only repository")
    require(not output.exists(), "Output directory must be fresh and nonexistent")
    sources = []
    manifest = json_data(read_source(repo, N + "/FILES_SHA256.json", N_MANIFEST_SHA, sources))

    def from_handoff(relative):
        entry = manifest.get(relative)
        require(isinstance(entry, dict), "Source missing from handoff manifest: " + relative)
        return read_source(repo, N + "/" + relative, entry["sha256"], sources, entry["bytes"])

    plan_manifest = json_data(read_source(repo, E + "/plans/FREQUENCY_MANIFEST.json",
                                          PLAN_MANIFEST_SHA, sources))
    require(plan_manifest["plan_sha256"] == PLAN_SHA, "Plan manifest disagrees with pinned plan")
    plan_data = read_source(repo, E + "/plans/frequency19200.plan", PLAN_SHA, sources)
    lines = plan_data.decode("utf-8").splitlines()
    head = lines[0].split()
    require(len(head) == 4 and head[0] == "XEON_IQ_CACHE_PLAN_V2", "Invalid plan header")
    plans = {}
    for line_number, line in enumerate(lines[1:], 2):
        f = line.split()
        require(len(f) >= 6 and f[0] not in plans, "Malformed or duplicate plan row")
        k, m, experts, count = map(int, f[2:6])
        selected = list(map(int, f[6:]))
        require(k > 0 and k % 256 == 0 and m > 0 and experts > 0, "Invalid plan dimensions")
        require(count == len(selected) and selected == sorted(set(selected))
                and all(0 <= x < experts for x in selected), "Invalid plan selected IDs")
        plans[f[0]] = {"type": f[1], "k": k, "m": m, "experts": experts,
                       "selected": selected, "line": line_number,
                       "expert_bytes": (k // 256) * m * 144}
    require(len(plans) == int(head[1]) == 80, "Plan tensor count mismatch")
    require(sum(len(x["selected"]) for x in plans.values()) == int(head[2]) == 19200,
            "Plan selected total mismatch")
    require(sum(len(x["selected"]) * x["expert_bytes"] for x in plans.values())
            == int(head[3]) == 127401984000, "Plan payload mismatch")
    target = plans[TENSOR]
    require((target["type"], target["k"], target["m"], target["experts"], len(target["selected"]))
            == ("iq2_s", K, M, EXPERTS, 313), "Target tensor differs from pinned geometry")

    log = from_handoff("execution/fixed-iq2-0-frozen/stdout.log")
    records = [json_data(x.encode("utf-8")) for x in log.decode("utf-8").splitlines() if x]
    require(len(records) == 3 and records[-1] == {"passed": True}, "Invalid fixed stdout receipt")
    require({x["cached_of_six"] for x in records[:2]} == {0, 6}, "Missing C0/C6 fixed receipt")
    for row in records[:2]:
        require(row["kind"] == "fixed" and row["tensor"] == TENSOR and row["type"] == "iq2_s"
                and row["threads"] == 12 and row["bit_mismatches"] == 0, "Unexpected fixed probe")
        ids = row["routes"]
        require(len(ids) == len(set(ids)) == USED * GROUPS
                and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < EXPERTS for x in ids),
                "Expected 48 distinct valid ordered route IDs")
    require(records[0]["routes"] == records[1]["routes"], "C0 and C6 routes differ")
    routes = records[0]["routes"]
    selected = sorted(set(routes).intersection(target["selected"]))
    masks = [sum(1 << slot for slot, expert in enumerate(routes[g * USED:(g + 1) * USED])
                 if expert in selected) for g in range(GROUPS)]
    counts = [bin(mask).count("1") for mask in masks]
    require(len(selected) == 39 and masks == EXPECTED_MASKS and counts == EXPECTED_COUNTS,
            "Derived mixed case differs from independently checked masks/counts")

    fixed = json_data(from_handoff("FIXED_RECEIPT.json"))
    matching = [row for row in fixed["runs"] if row["run"] == "fixed-iq2-0-frozen"]
    require(fixed["complete"] is True and len(matching) == 1, "Missing fixed input receipt")
    saved_input = matching[0]["files"]["inputs.bin"]
    require(saved_input == {"bytes": INPUT_BYTES, "sha256": INPUT_SHA}, "Fixed input identity mismatch")
    index = json_data(from_handoff("correctness/INDEX.json"))
    entry = index[INPUT_SHA]
    require(entry["bytes"] == INPUT_BYTES
            and entry["gzip"] == "correctness/" + INPUT_SHA + ".gz"
            and "fixed-outputs/fixed-iq2-0-frozen/inputs.bin" in entry["targets"],
            "Input index does not identify the fixed IQ2 input")
    compressed = from_handoff(entry["gzip"])
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
        inputs = stream.read(INPUT_BYTES + 1)
        require(len(inputs) == INPUT_BYTES, "Input expands beyond or below the exact byte budget")
        require(stream.read(1) == b"", "Unexpected extra gzip payload")
    require(sha256(inputs) == INPUT_SHA, "Uncompressed input SHA256 mismatch")
    require(all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", inputs)),
            "Saved input contains nonfinite float32 values")

    header = make_header(routes, selected, masks)
    metadata = {
        "schema_version": 1, "commit": COMMIT,
        "source_pin_method": "Fixed content hashes and cross-checked saved manifests; no git command or engine execution",
        "tensor_name": TENSOR,
        "shape": {"K": K, "M": M, "experts": EXPERTS, "used": USED, "groups": GROUPS, "inputs": INPUTS},
        "routes": routes, "cache_ids": selected, "masks": masks, "cached_of_six": counts,
        "plan_sha256": PLAN_SHA, "plan_line": target["line"],
        "source_plan_file": "source-plan.txt",
        "inputs_sha256": INPUT_SHA, "inputs_bytes": len(inputs),
        "inputs_encoding": "little-endian float32; eight exact saved synthetic input vectors",
        "expert_compact_bytes": EXPERT_BYTES,
        "cache_cases": {
            "C0": {"selected_count": 0, "cache_payload_bytes": 0, "cached_of_six": [0] * GROUPS},
            "C6": {"selected_count": len(routes), "cache_payload_bytes": len(routes) * EXPERT_BYTES,
                   "cached_of_six": [6] * GROUPS},
            "Cplan": {"selected_count": len(selected), "cache_payload_bytes": len(selected) * EXPERT_BYTES,
                      "cached_of_six": counts}},
        "full_final_cache_payload_bytes": int(head[3]),
        "target_tensor_final_selected_count": len(target["selected"]),
        "target_tensor_final_cache_payload_bytes": len(target["selected"]) * EXPERT_BYTES,
        "output_files": {
            "inputs.bin": {"bytes": len(inputs), "sha256": INPUT_SHA},
            "mixed_case_data.h": {"bytes": len(header), "sha256": sha256(header)},
            "source-plan.txt": {"bytes": len(plan_data), "sha256": PLAN_SHA}},
        "sources": sources,
        "extractor_sha256": sha256(Path(__file__).read_bytes()),
        "limitations": [
            "Cplan matches final-plan cache membership for these fixed probe experts, not typical API traffic.",
            "The eight routes and input vectors remain the existing synthetic probe fixtures.",
            "Seven of eight route groups are mixed; this is a diagnostic that emphasizes mixed execution.",
            "39/48 is not a production hit rate, and this isolated cache does not reproduce model-wide reuse distance.",
            "No model weights, model execution, library execution, GPU operation, or benchmark are included."]}
    meta_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    files = {"mixed_case.json": meta_bytes, "mixed_case_data.h": header, "inputs.bin": inputs,
             "source-plan.txt": plan_data}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
        require((output / name).read_bytes() == data, "Generated-file readback failed: " + name)
    receipt = {"complete": True, "python": platform.python_version(), "repo_commit_expected": COMMIT,
               "output_directory": str(output), "cached_of_six": counts, "masks": masks,
               "source_count": len(sources), "input_bytes": len(inputs), "input_sha256": INPUT_SHA,
               "gzip_read_budget_bytes": INPUT_BYTES + 1,
               "files": {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in files.items()},
               "scope": "Source verification and bounded evidence extraction only; no model/ELF/GPU/benchmark run"}
    with (output / "preparation_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Pinned repository or exact exported tree")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory outside the repository")
    args = parser.parse_args()
    try:
        receipt = prepare(args.repo, args.output)
    except (ValueError, KeyError, OSError, TypeError, EOFError) as error:
        print("prepare_mixed_case: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"complete": True, "output": receipt["output_directory"],
                      "cached_of_six": receipt["cached_of_six"], "input_sha256": receipt["input_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
