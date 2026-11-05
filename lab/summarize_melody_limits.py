#!/usr/bin/env python3
"""Summarize the new 26B observations, preserving unlike measurement units."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from summarize_bottleneck_observations import summarize

ROOT = Path(__file__).resolve().parent / "results/bottleneck-melody-fine-20261020"


def category(dso, symbol):
    if "ggml_vec_dot_q6_K_q8_K" in symbol:
        return "Q6_K_dot"
    if "ggml_vec_dot_q8_0_q8_0" in symbol:
        return "Q8_0_dot"
    if "libgomp" in dso:
        return "OpenMP_runtime"
    return "other"


def generate_report(mode, args, filename):
    command = ["perf", "report", "-f", "--stdio", "--stdio-color", "never",
               "--no-children", "--percent-limit", "0", "-t", ";",
               "-i", str(ROOT / mode / "profile.data"), *args]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    (ROOT / filename).write_text(result.stdout)
    (ROOT / (filename + ".stderr")).write_text(result.stderr)
    text = result.stdout
    lost = re.search(r"Total Lost Samples:\s*(\d+)", text)
    assert lost and int(lost[1]) == 0, "Missing sample-loss accounting or lost samples"
    rows = []
    for row in csv.reader(text.splitlines(), delimiter=";"):
        if row and "%" in row[0] and not row[0].startswith("#"):
            rows.append([float(row[0].strip().rstrip("%")), int(row[1]), int(row[2]),
                         row[3].strip(), row[4].strip()])
    assert rows and abs(sum(r[0] for r in rows) - 100) < 2
    return command, text, rows


def grouped(rows, column, weight_label):
    total_samples = sum(r[1] for r in rows)
    total_weight = sum(r[2] for r in rows)
    groups = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = column(r)
        groups[key][0] += r[1]
        groups[key][1] += r[2]
    return [
        {"name": k, "samples": n, "sample_share": n / total_samples,
         weight_label: w, "weight_share": w / total_weight}
        for k, (n, w) in sorted(groups.items(), key=lambda pair: -pair[1][1])
    ]


def main():
    run = json.loads((ROOT / "run.json").read_text())
    assert run["complete"] and run["normal_artifacts_and_global_settings_unchanged"]
    captures = {mode: json.loads((ROOT / mode / "capture.json").read_text())
                for mode in ("stalls", "record", "memory")}
    assert all(cap["valid_sample"] for cap in captures.values())
    assert len({cap["boot_id"] for cap in captures.values()}) == 1
    assert len({cap["bench_sha256"] for cap in captures.values()}) == 1
    stalls = summarize(ROOT / "stalls")
    assert stalls["valid"] and stalls["original_capture_all_event_gate_passed"]
    cmd1, text1, cycles = generate_report(
        "record", ["-g", "none", "-s", "dso,symbol",
                   "-F", "overhead,sample,period,dso,symbol"], "cycles-symbols.csv")
    cmd2, text2, memory = generate_report(
        "memory", ["--mem-mode", "-s", "mem,symbol",
                   "-F", "overhead,sample,period,mem,symbol"], "memory-symbols.csv")
    declared = int(re.search(r"Total weight\s*:\s*(\d+)", text2)[1])
    assert sum(r[2] for r in memory) == declared
    libgomp = Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")
    disassembly = subprocess.run(
        ["objdump", "-d", "--start-address=0x24520", "--stop-address=0x247b0",
         str(libgomp)], capture_output=True, text=True, check=True
    ).stdout
    assert re.search(r"24560:.*pause", disassembly)
    assert re.search(r"24720:.*pause", disassembly)
    (ROOT / "libgomp-wait-loops.asm").write_text(disassembly)
    wait_cycles = 0
    for row in cycles:
        address = re.fullmatch(r"\[\.\]\s+0x([0-9a-f]+)", row[4])
        if "libgomp" not in row[3] or address is None:
            continue
        offset = int(address[1], 16)
        if 0x24560 <= offset < 0x24571 or 0x24720 <= offset < 0x24732:
            wait_cycles += row[2]
    result = {
        "valid": True,
        "model": str(captures["stalls"]["arguments"]["model"]),
        "configuration": "CPU24, single anonymous copy interleaved 4KiB, THP off for this process only",
        "stalls": stalls,
        "cycles": {
            "lost_samples": 0,
            "sample_count": sum(r[1] for r in cycles),
            "categories": grouped(cycles, lambda r: category(r[3], r[4]), "sampled_cycles"),
            "symbols": grouped(cycles, lambda r: r[3] + " " + r[4], "sampled_cycles"),
            "verified_pause_wait_loop_cycles_share": wait_cycles / sum(r[2] for r in cycles),
            "verified_wait_loop_ranges": ["0x24560 <= offset < 0x24571", "0x24720 <= offset < 0x24732"],
            "libgomp_sha256": hashlib.sha256(libgomp.read_bytes()).hexdigest(),
            "note": "Sampled CPU cycles include stalls; OpenMP runtime is not automatically all waiting.",
        },
        "memory": {
            "lost_samples": 0,
            "sample_count": sum(r[1] for r in memory),
            "categories": grouped(memory, lambda r: category("", r[4]), "sum_sampled_use_latency"),
            "symbols": grouped(memory, lambda r: r[4], "sum_sampled_use_latency"),
            "sources": grouped(memory, lambda r: r[3], "sum_sampled_use_latency"),
            "note": "Sampled loads above 30 cycles, period1000. Use-latency includes pipeline queueing; weight shares are not wall-time fractions. LFB samples hide the eventual backing source.",
        },
        "report_commands": [cmd1, cmd2],
        "measurement_notes": [
            "One 18-second window per measurement type; this is a diagnostic, not a new A/B speed comparison.",
            "Different counters and profiles overlap; do not sum their fractions or infer removable wall time.",
            "IMC traffic is system-wide. DeepSeek results cannot be substituted for these 26B measurements.",
            "Synthetic generation from shallow context, warm memory; not cold load or long conversation.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "metrics": stalls["metrics"],
        "cycles_categories": result["cycles"]["categories"],
        "cycles_top_symbols": result["cycles"]["symbols"][:10],
        "verified_wait_loop_cycles_share": result["cycles"]["verified_pause_wait_loop_cycles_share"],
        "memory_categories": result["memory"]["categories"],
        "memory_sources": result["memory"]["sources"][:4],
    }, indent=2))


if __name__ == "__main__":
    main()
