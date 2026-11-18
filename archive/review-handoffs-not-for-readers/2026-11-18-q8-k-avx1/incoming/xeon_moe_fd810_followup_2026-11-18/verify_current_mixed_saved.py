#!/usr/bin/env python3
"""Parent's independent, read-only audit of the new saved CPU trial.

Usage: python3 verify_current_mixed_saved.py [astra62_next_outputs_directory]
No engine, build, model, network or third-party modules are used.
"""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
import struct
import sys


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "astra62_next"
if not (root / "protocol.json").is_file() and (root / "outputs/protocol.json").is_file():
    root = root / "outputs"
protocol = json.loads((root / "protocol.json").read_text())
variants = protocol["variants"]
need(variants == ["current", "08", "09"], "unexpected comparison arms")
need(len(protocol["selection_masks"]) == 6, "expected six cache selections")
need(len(protocol["numerical_round_modes"]) == 4, "expected four rounding modes")
round_bits = {"nearest": 1, "down": 2, "up": 4, "toward_zero": 8}

dump_records = rows(root / "raw/dump-sha256.csv")
numerical = rows(root / "raw/numerical.csv")
blobs, descriptions = {}, {}
for row in dump_records:
    key = tuple(row[x] for x in ("selection", "mode", "variant"))
    need(key not in blobs, "duplicate dump")
    blob = gzip.decompress((root / "raw" / row["file"]).read_bytes())
    need(len(blob) == int(row["bytes"]), "dump length")
    need(hashlib.sha256(blob).hexdigest() == row["sha256"], "dump SHA256")
    blobs[key], descriptions[key] = blob, row
expected = {(s, m, v) for s in protocol["selection_masks"]
            for m in protocol["numerical_round_modes"] for v in variants}
need(set(blobs) == expected, "dump coverage")
cpu_hashes = {v: {r["cpu_sha256"] for r in dump_records if r["variant"] == v} for v in variants}
need(all(len(x) == 1 for x in cpu_hashes.values()), "CPU hash consistency")
need(len(set.union(*cpu_hashes.values())) == 3, "three distinct CPU DSOs")
for name in ("base_sha256", "probe_sha256", "counter_sha256"):
    need(len({r[name] for r in dump_records}) == 1, "common " + name)

by_dump = defaultdict(list)
for row in numerical:
    by_dump[tuple(row[x] for x in ("selection", "mode", "variant"))].append(row)
need(set(by_dump) == expected, "CSV coverage")
preparation_example = defaultdict(dict)
graphs = 0
for key, records in by_dump.items():
    selection, mode, variant = key
    mask = protocol["selection_masks"][selection]
    desc, data = descriptions[key], blobs[key]
    need(len(records) == int(desc["graphs"]) == 67, "67 graphs per dump")
    position = 0
    for r in records:
        start, length = int(r["offset"]), int(r["bytes"])
        need(start == position, "ordered, gapless dump ranges")
        position += length
        piece = data[start:position]
        need(len(piece) == length and fnv(piece) == r["hash"], "per-graph bytes/hash")
        need(r["cpu_sha256"] == desc["cpu_sha256"], "CSV/dump CPU identity")
        k, outputs, tokens, inputs, workers, phase = (int(r[x]) for x in
            ("k", "rows", "tokens", "inputs", "threads", "phase"))
        need(length == outputs * 6 * tokens * 4, "output dimensions")
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", piece)), "finite fixture output")
        routes = [protocol["routes"][(phase + t) % 5] for t in range(tokens)]
        for t, route in enumerate(routes):
            for slot, expert in enumerate(route):
                if expert == -1:
                    start_zero = (t * 6 + slot) * outputs * 4
                    need(piece[start_zero:start_zero + outputs * 4] == bytes(outputs * 4), "skipped slot zeros")
        iq = r["type"] in ("iq2_s", "iq3_xxs", "iq3_s")
        sparse = iq and mask != 0 and tokens == 1 and not int(r["ref"])
        valid = sum(e >= 0 for route in routes for e in route)
        compact = sum(e >= 0 and bool(mask & (1 << e)) for e in routes[0]) if sparse else 0
        used_columns = {slot % inputs for slot, e in enumerate(routes[0])
                        if sparse and e >= 0 and mask & (1 << e)}
        folding = sparse and not int(r["raw_q8"]) and variant != "current"
        folded_columns = inputs if folding and variant == "08" else len(used_columns) if folding else 0
        blocks = k // 256
        active_workers = sum((i + 1) * blocks // workers > i * blocks // workers for i in range(workers))
        calls = (workers * inputs if folding and variant == "08" else
                 active_workers * len(used_columns) if folding else 0)
        need(int(r["prepare_blocks"]) == folded_columns * blocks, "prepared blocks")
        need(int(r["prepare_calls"]) == calls, "prepare calls including empty workers")
        need(int(r["prepare_round_mask"]) == (round_bits[mode] if calls else 0), "worker rounding mode")
        need((int(r["fold_select"]) > 0) == (folding and compact > 0), "fold selector presence")
        # The small numerical shapes take one dot callback per output row.
        need(int(r["native_dots"]) == (valid - compact) * outputs if iq else int(r["native_dots"]) == 0,
             "native IQ callback count")
        legacy = compact * outputs if not folding else 0
        need(int(r["legacy_dots"]) == legacy, "legacy compact callback count")
        mx = r["case"] == "mxfp4_prepack"
        need(int(r["mx_prepare"]) == (inputs * workers if mx else 0), "MX prepare preserved")
        need(int(r["mx_dot"]) == (valid * outputs if mx else 0), "MX dot preserved")
        if mode == "nearest" and r["case"] == "strided_down" and phase == 0 and r["type"] == "iq3_xxs":
            preparation_example[selection][variant] = {"blocks": int(r["prepare_blocks"]), "calls": calls}
        graphs += 1
    need(position == len(data), "no unaccounted output bytes")

for selection in protocol["selection_masks"]:
    for mode in protocol["numerical_round_modes"]:
        current = blobs[(selection, mode, "current")]
        need(current == blobs[(selection, mode, "08")] == blobs[(selection, mode, "09")],
             "whole-output byte comparison")

timing = rows(root / "raw/timing.csv")
processes = defaultdict(list)
for r in timing:
    processes[(r["case"], r["selection"], int(r["round"]), r["variant"])].append(r)
cases = {(c["name"], mask): c for c in protocol["cases"] for mask in c["masks"]}
need(len(cases) == 10, "ten timing conditions")
need(set(processes) == {(c, s, n, v) for c, s in cases for n in range(6) for v in variants},
     "timing process coverage")
triplets, timings_summary = [], []
for (case, selection), shape in cases.items():
    medians_by_arm, ratios = defaultdict(list), defaultdict(list)
    for n in range(6):
        times, output_hashes = {}, set()
        for variant in variants:
            record = processes[(case, selection, n, variant)]
            need(len(record) == 5 and {int(r["sample"]) for r in record} == set(range(5)), "five samples")
            for r in record:
                need(int(r["seed"]) == protocol["seed_base"] + n, "paired seed")
                need(int(r["order"]) == protocol["timing_orders"][n].index(variant), "balanced order")
                for field in ("k", "rows", "inputs", "threads", "loops"):
                    need(int(r[field]) == shape[field], "timing shape/loop count")
                need(r["affinity"] == shape["affinity"], "recorded affinity")
                need(r["cpu_sha256"] in cpu_hashes[variant], "timing CPU identity")
                need(r["base_sha256"] == dump_records[0]["base_sha256"], "timing base identity")
                need(r["probe_sha256"] == dump_records[0]["probe_sha256"], "timing probe identity")
                output_hashes.add(r["output_hash"])
            times[variant] = median(float(r["ns_per_op"]) for r in record)
            medians_by_arm[variant].append(times[variant] / 1000)
        need(len(output_hashes) == 1, "timing output hash consistency")
        pair = {"08_current": times["08"] / times["current"],
                "09_current": times["09"] / times["current"], "09_08": times["09"] / times["08"]}
        for comparison, value in pair.items():
            ratios[comparison].append(value)
        triplets.append({"case": case, "selection": selection, "round": n, "ratios": pair})
    timings_summary.append({"case": case, "selection": selection, "triplets": 6,
        "process_medians_us": {v: median(values) for v, values in medians_by_arm.items()},
        "ratios": {name: {"median": median(values), "min": min(values), "max": max(values),
                          "wins": sum(x < 1 for x in values), "all": values} for name, values in ratios.items()}})

published = {(r["case"], r["selection"]): r for r in rows(root / "summary/timing-summary.csv")}
need(set(published) == set(cases), "published summary coverage")
for result in timings_summary:
    saved = published[(result["case"], result["selection"])]
    need(int(saved["triplets"]) == 6, "published triplets")
    for name, stats in result["ratios"].items():
        need(abs(float(saved[f"ratio_{name}_median"]) - stats["median"]) <= 0.00000051,
             "independent median matches published summary")
        need(int(saved[f"ratio_{name}_wins"]) == stats["wins"], "independent wins")
        ends = [float(x) for x in saved[f"ratio_{name}_range"].split("..")]
        need(all(abs(x - y) <= 0.00000051 for x, y in zip(ends, [stats["min"], stats["max"]])), "ratio range")

result = {
    "status": "pass",
    "method": "Parent-authored independent saved-output and timing recalculation; engine not executed",
    "source_files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in (
        "protocol.json", "raw/numerical.csv", "raw/dump-sha256.csv", "raw/timing.csv", "summary/timing-summary.csv")},
    "dump_files": len(blobs), "verified_graph_ranges": graphs, "comparison_triads": graphs // 3,
    "values_per_variant": {v: sum(len(b) // 4 for k, b in blobs.items() if k[2] == v) for v in variants},
    "all_corresponding_outputs_byte_equal": True,
    "all_expected_preparation_and_callback_checks_passed": True,
    "preparation_example": dict(preparation_example),
    "cpu_sha256": {v: next(iter(hashes)) for v, hashes in cpu_hashes.items()},
    "base_sha256": dump_records[0]["base_sha256"],
    "timing_rows": len(timing), "timing_processes": len(processes), "timing_triplets": len(triplets),
    "timing_summary": timings_summary,
    "limits": ["Synthetic inputs and weights, not model quality or the deployed runtime",
               "Saved path counters count selector invocations, not each folded dot",
               "Timing output hashes are checked; timing output binaries were not saved",
               "No independent engine execution or new performance run by this script"]
}
print(json.dumps(result, ensure_ascii=False, indent=2))
