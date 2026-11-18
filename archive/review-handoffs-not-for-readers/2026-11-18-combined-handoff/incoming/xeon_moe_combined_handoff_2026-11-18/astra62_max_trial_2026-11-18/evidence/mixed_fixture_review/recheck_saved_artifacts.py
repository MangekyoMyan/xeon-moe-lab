#!/usr/bin/env python3
"""Reaggregate saved fixture results only. Never build or run engine/probe binaries."""
import argparse
import collections
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def patch_sections(text):
    sections = {}
    name = None
    for line in text.splitlines(keepends=True):
        if line.startswith("--- "):
            name = None
        elif line.startswith("+++ b/"):
            name = line[len("+++ b/"):].strip()
            sections[name] = []
        elif name is not None:
            sections[name].append(line)
    return sections


def apply_in_memory(original, lines):
    source = original.splitlines(keepends=True)
    out = []
    cursor = 0
    index = 0
    while index < len(lines):
        header = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
        if not header:
            raise AssertionError(f"Unexpected patch line: {lines[index]!r}")
        start = int(header.group(1)) - 1
        before_count = int(header.group(2) or 1)
        after_count = int(header.group(4) or 1)
        assert start >= cursor
        out.extend(source[cursor:start])
        cursor = start
        index += 1
        old = new = 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line.startswith(" "):
                assert source[cursor] == line[1:]
                out.append(line[1:]); cursor += 1; old += 1; new += 1
            elif line.startswith("-"):
                assert source[cursor] == line[1:]
                cursor += 1; old += 1
            elif line.startswith("+"):
                out.append(line[1:]); new += 1
            elif line.startswith("\\ No newline"):
                raise AssertionError("Not expected in reviewed patch")
            else:
                raise AssertionError(f"Unexpected hunk line: {line!r}")
            index += 1
        assert (old, new) == (before_count, after_count)
    out.extend(source[cursor:])
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/scratch/1f624f9ac929"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "parent_evaluation/mixed_fixture_review"
    out.mkdir(parents=True, exist_ok=True)
    outputs = root / "astra62_max_trial/outputs"
    fixture = outputs / "route37-fixture"
    raw = outputs / "raw/route37-fixture"
    work = root / "astra62_max_trial/work/mixed-run"
    packet = root / "parent_evaluation/requested_source_02"
    identities = {}

    def read(path):
        data = path.read_bytes()
        rel = str(path.relative_to(root))
        identities[rel] = {"bytes": len(data), "sha256": sha(data)}
        return data

    for path in sorted(fixture.rglob("*")):
        if path.is_file():
            read(path)
    for name in ("06-compact-signed-byte.patch", "graph_probe.cpp", "build_cpu.sh"):
        read(outputs / name)
    for path in sorted(raw.iterdir()):
        if path.is_file():
            read(path)
    read(outputs / "raw/fixture-comparison.csv")

    rows = csv_rows(raw / "cases.csv")
    key_fields = ["mask", "mode", "type", "k", "threads", "raw", "pattern"]
    grouped = collections.defaultdict(dict)
    duplicate_keys = []
    for row in rows:
        key = tuple(row[name] for name in key_fields)
        if row["fixture"] in grouped[key]:
            duplicate_keys.append([list(key), row["fixture"]])
        grouped[key][row["fixture"]] = row["hash"]
    missing = [list(key) for key, value in grouped.items() if set(value) != {"base", "mixed"}]
    unequal = [list(key) for key, value in grouped.items() if len(set(value.values())) != 1]
    field_counts = {field: dict(collections.Counter(row[field] for row in rows))
                    for field in ["fixture"] + key_fields}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "After independently frozen initial ideas and after requested_source_02 was supplied",
        "method": "Static reads, stored CSV aggregation, stored binary byte/hash comparison only; no engine/probe execution, compilation, or performance measurement",
        "case_rows": len(rows),
        "paired_cases": len(grouped),
        "duplicate_case_keys": duplicate_keys,
        "missing_pair_keys": missing,
        "case_hash_mismatch_keys": unequal,
        "field_counts": field_counts,
    }
    assert not duplicate_keys and not missing and not unequal

    comparisons = csv_rows(outputs / "raw/fixture-comparison.csv")
    reported = {(int(row["mask"]), row["rounding"]): row for row in comparisons}
    rounding = ["nearest", "downward", "upward", "towardzero"]
    hashes = {}
    for line in (raw / "hashes.txt").read_text().splitlines():
        digest, filename = line.split(None, 1)
        hashes[filename.strip()] = digest
    binary_checks = []
    per_case_binary_failures = []
    originals_agree = True
    for mask in [0, 1, 21, 31, 63]:
        for mode in range(4):
            stored = {}
            for variant in ["base", "mixed"]:
                filename = f"{mask}-{mode}-{variant}.bin"
                data = read(work / filename)
                assert len(data) == 48 * 49 * 6 * 4
                assert sha(data) == hashes[filename]
                stored[variant] = data
                selected = [row for row in rows if row["fixture"] == variant and
                            int(row["mask"]) == mask and int(row["mode"]) == mode]
                original_rows = csv_rows(work / "raw" / f"{mask}-{mode}-{variant}.csv")
                read(work / "raw" / f"{mask}-{mode}-{variant}.csv")
                assert len(selected) == len(original_rows) == 48
                for number, (selected_row, original_row) in enumerate(zip(selected, original_rows)):
                    originals_agree &= all(selected_row[key] == original_row[key]
                                           for key in ["mode", "type", "k", "threads", "raw", "pattern", "hash"])
                    piece = data[number * 1176 : (number + 1) * 1176]
                    if fnv(piece) != selected_row["hash"]:
                        per_case_binary_failures.append([filename, number])
            actual_differences = sum(stored["base"][i:i+4] != stored["mixed"][i:i+4]
                                     for i in range(0, len(stored["base"]), 4))
            report = reported[(mask, rounding[mode])]
            row = {
                "mask": mask, "mode": mode, "rounding": rounding[mode],
                "float_outputs": len(stored["base"]) // 4,
                "mismatched_float_bits": actual_differences,
                "all_bytes_equal": stored["base"] == stored["mixed"],
                "base_sha256": sha(stored["base"]),
                "mixed_sha256": sha(stored["mixed"]),
                "matches_reported_comparison": (
                    int(report["float_outputs"]) == len(stored["base"]) // 4 and
                    int(report["mismatched_float_bits"]) == actual_differences and
                    report["base_sha256"] == sha(stored["base"]) and
                    report["mixed_sha256"] == sha(stored["mixed"]))
            }
            binary_checks.append(row)
    assert originals_agree and not per_case_binary_failures
    assert all(row["matches_reported_comparison"] and row["all_bytes_equal"] for row in binary_checks)
    result["stored_binary_comparisons"] = binary_checks
    result["paired_float_outputs"] = sum(row["float_outputs"] for row in binary_checks)
    result["mismatched_float_bits"] = sum(row["mismatched_float_bits"] for row in binary_checks)
    result["original_per_run_csv_agrees_with_consolidated_csv"] = originals_agree
    result["per_case_fnv64_agrees_with_stored_binary"] = not per_case_binary_failures

    counts = []
    instrumented = []
    for variant in ["base", "mixed"]:
        for mask in [0, 1, 21, 31, 63]:
            path = raw / f"paths-{variant}-{mask}.csv"
            records = csv_rows(path)
            assert len(records) == 1
            counts.append({"variant": variant, "mask": mask,
                           **{name: int(value) for name, value in records[0].items()}})
            assert read(work / "raw" / path.name) == path.read_bytes()
            data = read(work / f"instrumented-{variant}-{mask}.bin")
            reference = read(work / f"{mask}-0-base.bin")
            instrumented.append({"variant": variant, "mask": mask,
                                 "bytes": len(data), "matches_uninstrumented_base": data == reference})
    assert all(row["matches_uninstrumented_base"] for row in instrumented)
    result["saved_path_counters_nearest_only"] = counts
    result["instrumented_stored_binary_comparisons"] = instrumented

    packet_checks = []
    for path in sorted(packet.iterdir()):
        if path.is_file():
            packet_checks.append({"file": path.name,
                                  "exactly_matches_source_02": read(path) == read(fixture / "excerpts" / path.name)})
    assert all(row["exactly_matches_source_02"] for row in packet_checks)
    result["source_packet_identity"] = packet_checks
    sections = patch_sections((fixture / "08-mixed-dispatch.patch").read_text())
    normalized = {}
    excerpts = []
    for variant in ["base", "mixed"]:
        source_dir = work / f"src-{variant}"
        cpu_path = source_dir / "ggml/src/ggml-cpu/ggml-cpu.c"
        cpu = read(cpu_path).decode()
        norm = cpu
        for name in ["mul_mat_id_route37.c", "graph_plan_route37.c"]:
            expected = (packet / name).read_text()
            if variant == "mixed":
                expected = apply_in_memory(expected, sections[name])
            actual = read(source_dir / "route37-test-extracts" / name).decode()
            row = {"variant": variant, "file": name, "sha256": sha(actual.encode()),
                   "matches_unmodified_or_in_memory_patched_source_02": actual == expected,
                   "exact_occurrences_in_prepared_cpu_source": cpu.count(actual)}
            assert row["matches_unmodified_or_in_memory_patched_source_02"] and cpu.count(actual) == 1
            norm = norm.replace(actual, "/* REMOVED VERIFIED FUNCTION EXCERPT */")
            excerpts.append(row)
        normalized[variant] = norm
        for name in ["iq3-cache.cpp", "iq3-cache.h", "folded-input.inc", "route37-test-fixture.cpp"]:
            read(source_dir / "xeon-local" / name)
        cdb_path = work / f"build-{variant}/compile_commands.json"
        cdb = json.loads(read(cdb_path))
        result[f"relevant_compile_commands_{variant}"] = [entry for entry in cdb if entry["file"].endswith((
            "/ggml-cpu.c", "/xeon-local/iq3-cache.cpp", "/xeon-local/route37-test-fixture.cpp"))]
        dso_name = f"build-{variant}/bin/libggml-cpu.so"
        assert sha(read(work / dso_name)) == hashes[dso_name]
    result["transplanted_excerpts"] = excerpts
    result["base_mixed_cpu_source_equal_outside_verified_excerpts"] = normalized["base"] == normalized["mixed"]
    assert result["base_mixed_cpu_source_equal_outside_verified_excerpts"]
    result["shared_folded_kernel_source_equal"] = (
        (work / "src-base/xeon-local/folded-input.inc").read_bytes() ==
        (work / "src-mixed/xeon-local/folded-input.inc").read_bytes())
    assert result["shared_folded_kernel_source_equal"]
    (out / "RECHECK_SAVED_ARTIFACTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "SOURCE_FILE_IDENTITIES.json").write_text(json.dumps({
        "created_utc": result["created_utc"], "root": str(root), "files": identities,
        "note": "Source files and stored binaries were only read. No engine binaries were executed."
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"case_rows": result["case_rows"], "paired_cases": result["paired_cases"],
                      "paired_float_outputs": result["paired_float_outputs"],
                      "mismatched_float_bits": result["mismatched_float_bits"],
                      "source_packet_files": len(packet_checks), "identified_files": len(identities)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
