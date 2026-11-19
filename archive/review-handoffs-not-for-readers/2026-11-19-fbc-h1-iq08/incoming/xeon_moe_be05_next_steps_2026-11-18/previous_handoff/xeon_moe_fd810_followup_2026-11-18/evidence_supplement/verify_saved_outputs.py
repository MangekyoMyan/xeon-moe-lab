#!/usr/bin/env python3
"""Read saved gzip/CSV/JSON only; do not import or execute any engine code.

Run from any directory: python3 /path/to/evidence_supplement/verify_saved_outputs.py
All paths used for reading are relative to this package. No DSO, compiler,
network request, model, or third-party Python dependency is used.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fnv64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()

    def read(name):
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe package path: {name}")
        path = root.joinpath(*relative.parts).resolve()
        require(path.is_relative_to(root), f"Path leaves package: {name}")
        return path.read_bytes()

    def load_json(name):
        return json.loads(read(name))

    def rows(name):
        return list(csv.DictReader(io.StringIO(read(name).decode("utf-8"))))

    index = load_json("INDEX.json")
    support = {item["path"]: item for item in index["support_files"]}
    for name, entry in support.items():
        data = read(name)
        require(len(data) == entry["bytes"] and digest(data) == entry["sha256"],
                f"Support file differs: {name}")

    run = load_json("records/RUN_MANIFEST.json")
    parent = load_json("records/FINAL_ARTIFACT_VERIFICATION.json")
    mixed_parent = load_json("records/RECHECK_SAVED_ARTIFACTS.json")
    mixed_identities = load_json("records/SOURCE_FILE_IDENTITIES.json")["files"]
    mixed_report = {(int(r["mask"]), r["rounding"]): r
                    for r in rows("csv/mixed/fixture-comparison.csv")}
    hash_lines = {}
    for line in read("records/mixed_hashes.txt").decode("utf-8").splitlines():
        sha, name = line.split(None, 1)
        hash_lines[name.strip()] = sha
    original_final = run["ordinary_and_rounding_final"]
    parent_final = {(group["suite"], item["variant"]): item
                    for group in parent["final_output_suites"]
                    for item in group["files"]}
    original_mixed = {(int(item["mask"]), int(item["mode"])): item
                      for item in mixed_parent["stored_binary_comparisons"]}
    consolidated = rows("csv/mixed/cases.csv")
    key_fields = ("mode", "type", "k", "threads", "raw", "pattern")
    grouped_csv = collections.defaultdict(list)
    case_keys = set()
    for row in consolidated:
        key = (row["fixture"], row["mask"], *(row[name] for name in key_fields))
        require(key not in case_keys, f"Duplicate mixed CSV key: {key}")
        case_keys.add(key)
        grouped_csv[(row["fixture"], int(row["mask"]), int(row["mode"]))].append(row)
    require(len(consolidated) == 1920 and len(grouped_csv) == 40,
            "Mixed CSV cardinality is not the recorded 1920 rows / 40 runs")

    data_by_id = {}
    rows_by_id = {}
    per_graph_hashes = 0
    final_grouped = collections.defaultdict(dict)
    mixed_grouped = collections.defaultdict(dict)
    instruments = []

    for entry in index["outputs"]:
        identifier = entry["id"]
        require(identifier not in data_by_id, f"Duplicate output id: {identifier}")
        compressed = read(entry["path"])
        require(len(compressed) == entry["gzip_bytes"] and
                digest(compressed) == entry["gzip_sha256"],
                f"Compressed file differs: {identifier}")
        data = gzip.decompress(compressed)
        require(len(data) == entry["uncompressed_bytes"] and
                digest(data) == entry["sha256"],
                f"Uncompressed file differs: {identifier}")
        require(len(data) == entry["float_outputs"] * 4,
                f"Not the expected F32 byte count: {identifier}")
        data_by_id[identifier] = data
        case_rows = rows(entry["case_csv"])
        require(len(case_rows) == entry["case_rows"], f"Case count differs: {identifier}")
        rows_by_id[identifier] = case_rows

        if entry["kind"] == "final":
            name = entry["original_filename"]
            require(digest(data) == original_final["sha256_by_dump"][name],
                    f"Final dump differs from prior RUN_MANIFEST: {identifier}")
            csv_name = PurePosixPath(entry["case_csv"]).name
            require(digest(read(entry["case_csv"])) == original_final["sha256_by_csv"][csv_name],
                    f"Final CSV differs from prior RUN_MANIFEST: {identifier}")
            previous = parent_final[(entry["suite"], entry["variant"])]
            require(previous["dump_sha256"] == digest(data) and
                    previous["dump_bytes"] == len(data) and
                    previous["case_rows"] == len(case_rows),
                    f"Final dump differs from prior parent verification: {identifier}")
            final_grouped[entry["suite"]][entry["variant"]] = identifier
        elif entry["kind"] == "mixed":
            require(digest(data) == hash_lines[entry["original_filename"]],
                    f"Mixed dump differs from prior hashes.txt: {identifier}")
            previous = original_mixed[(entry["mask"], entry["mode"])]
            require(previous[entry["variant"] + "_sha256"] == digest(data),
                    f"Mixed dump differs from prior parent verification: {identifier}")
            csv_group = grouped_csv[(entry["variant"], entry["mask"], entry["mode"])]
            require(len(csv_group) == len(case_rows), f"Mixed CSV rows differ: {identifier}")
            require(all(all(a[k] == b[k] for k in (*key_fields, "hash"))
                        for a, b in zip(case_rows, csv_group)),
                    f"Per-run vs consolidated CSV differs: {identifier}")
            mixed_grouped[(entry["mask"], entry["mode"])][entry["variant"]] = identifier
        elif entry["kind"] == "instrumented":
            expected = mixed_identities[entry["original_workspace_relative"]]
            require(expected["bytes"] == len(data) and expected["sha256"] == digest(data),
                    f"Instrumented dump differs from prior identities: {identifier}")
            counters = rows(entry["path_counter_csv"])
            require(len(counters) == 1, f"Counter row count differs: {identifier}")
            instruments.append(entry)
        else:
            raise ValueError(f"Unexpected kind: {entry['kind']}")

        offset = 0
        for i, row in enumerate(case_rows):
            if entry["case_layout"] == "rows_slots_tokens":
                require(int(row["case"]) == i, f"Nonsequential case number: {identifier}/{i}")
                size = int(row["rows"]) * int(row["slots"]) * int(row["tokens"]) * 4
                expected_hash = row["hash64"]
            else:
                require(entry["case_layout"] == "49_rows_6_slots_1_token",
                        f"Unknown graph layout: {identifier}")
                size = 49 * 6 * 4
                expected_hash = row["hash"]
            piece = data[offset:offset + size]
            require(len(piece) == size and fnv64(piece) == expected_hash,
                    f"Graph bytes vs CSV FNV64 differ: {identifier}/{i}")
            offset += size
            per_graph_hashes += 1
        require(offset == len(data), f"Unused or missing dump bytes: {identifier}")

    variants = {"baseline", "route2", "q8", "iqsafe", "combosafe"}
    require(set(final_grouped) == {"check", "edges", "round0", "round1", "round2", "round3"},
            "Final suites missing or unexpected")
    final_pairs = []
    shape_fields = ("case", "seed", "type", "k", "rows", "experts", "slots", "tokens",
                    "activations", "threads", "cache", "pad", "raw", "ref", "pattern")
    for suite, variants_to_ids in sorted(final_grouped.items()):
        require(set(variants_to_ids) == variants, f"Final variants differ: {suite}")
        baseline_id = variants_to_ids["baseline"]
        reference = data_by_id[baseline_id]
        fields = shape_fields if suite in ("check", "edges") else key_fields
        expected_keys = [tuple(row[k] for k in fields) for row in rows_by_id[baseline_id]]
        for variant in sorted(variants - {"baseline"}):
            identifier = variants_to_ids[variant]
            require(data_by_id[identifier] == reference, f"Full final outputs differ: {identifier}")
            actual_keys = [tuple(row[k] for k in fields) for row in rows_by_id[identifier]]
            require(actual_keys == expected_keys, f"Final case conditions differ: {identifier}")
            final_pairs.append({"suite": suite, "variant": variant,
                                "float_outputs": len(reference) // 4, "all_bytes_equal": True})

    expected_mixed = {(mask, mode) for mask in (0, 1, 21, 31, 63) for mode in range(4)}
    require(set(mixed_grouped) == expected_mixed, "Mixed groups missing or unexpected")
    mixed_pairs = []
    names = ["nearest", "downward", "upward", "towardzero"]
    for (mask, mode), variants_to_ids in sorted(mixed_grouped.items()):
        require(set(variants_to_ids) == {"base", "mixed"}, "Mixed pair incomplete")
        base = data_by_id[variants_to_ids["base"]]
        mixed = data_by_id[variants_to_ids["mixed"]]
        require(base == mixed, f"Full mixed outputs differ: {mask}/{mode}")
        reported = mixed_report[(mask, names[mode])]
        require(int(reported["float_outputs"]) == len(base) // 4 and
                int(reported["mismatched_float_bits"]) == 0 and
                reported["base_sha256"] == digest(base) and
                reported["mixed_sha256"] == digest(mixed),
                f"Prior fixture comparison differs: {mask}/{mode}")
        mixed_pairs.append({"mask": mask, "mode": mode,
                            "float_outputs": len(base) // 4, "all_bytes_equal": True})

    require(len(instruments) == 10, "Instrumented output count differs")
    for entry in instruments:
        reference = mixed_grouped[(entry["mask"], 0)][entry["variant"]]
        require(data_by_id[entry["id"]] == data_by_id[reference],
                f"Instrumented vs uninstrumented outputs differ: {entry['id']}")

    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Read existing compressed binary outputs, CSV, and prior JSON/hash records only",
        "new_engine_execution": False,
        "DSO_verification_performed": False,
        "compressed_output_files": len(data_by_id),
        "support_files_checked": len(support),
        "per_graph_fnv64_checks": per_graph_hashes,
        "final_output_files": sum(len(x) for x in final_grouped.values()),
        "final_graph_records": sum(len(rows_by_id[i]) for d in final_grouped.values() for i in d.values()),
        "final_graphs_per_variant": sum(len(rows_by_id[d["baseline"]]) for d in final_grouped.values()),
        "final_floats_per_variant": sum(len(data_by_id[d["baseline"]]) // 4 for d in final_grouped.values()),
        "final_full_output_pairs": final_pairs,
        "mixed_output_files": sum(len(x) for x in mixed_grouped.values()),
        "mixed_case_rows": len(consolidated),
        "mixed_paired_graphs": len(consolidated) // 2,
        "mixed_paired_floats": sum(r["float_outputs"] for r in mixed_pairs),
        "mixed_full_output_pairs": mixed_pairs,
        "instrumented_output_files": len(instruments),
        "instrumented_outputs_equal_uninstrumented": True,
        "missing_or_mismatched_files": [],
        "all_passed": True,
        "limits": "Saved evidence integrity/equality only. No rerun, performance measurement, model-quality proof, current-runtime integration proof, or attestation of historic process execution.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
