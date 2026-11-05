#!/usr/bin/env python3
"""Matched full-model screening of Q6 instructions and NUMA work scheduling."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import full_replica_experiment as bench

LAB = Path(__file__).resolve().parent
BASE = LAB / "review-full-replica-20261020/build"
CANDIDATE = LAB / "review-q6-scales-20261020/build"
OUT = LAB / "results/optimization-q6-xeon-20261020"


def compare_flags():
    def normalized(build):
        source = build.parent / "source"
        rows = json.loads((build / "compile_commands.json").read_text())
        result = {}
        for row in rows:
            if "/ggml-cpu/" in row["file"]:
                flags = shlex.split(row["command"])
                flags = flags[:flags.index("-o")]
                result[str(Path(row["file"]).relative_to(source))] = [
                    v.replace(str(source), "SOURCE").replace(str(build), "BUILD") for v in flags]
        return result
    a, b = normalized(BASE), normalized(CANDIDATE)
    assert a == b, "Compiler flags differ"
    report = {"identical": True, "cpu_compilation_units": len(a),
              "libraries": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (BASE/"bin/libggml-cpu.so", CANDIDATE/"bin/libggml-cpu.so")},
              "normal_artifacts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (LAB/"run_melody.sh", LAB/"run_deepseek_flash.sh",
                                             LAB/"build-avx-prefetch/bin/libggml-cpu.so")}}
    (OUT/"compiler-comparison.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "confirm-kernel", "confirm-numa", "confirm-joint"))
    args = parser.parse_args()
    assert "THP_enabled:\t0" in Path("/proc/self/status").read_text()
    bench.OUT = OUT
    compare_flags()
    labels = {"A": (BASE, {}), "B": (CANDIDATE, {}),
              "C": (CANDIDATE, {"GGML_CPU_Q6_STATIC_ROWS": "1"})}
    sequence = {"screen": "ABC", "confirm-kernel": "ABBA",
                "confirm-numa": "ACCA", "confirm-joint": "ABCCBA"}[args.stage]
    for i, label in enumerate(sequence):
        build, extra = labels[label]
        bench.run(f"{args.stage}-{i}-{label}", 2, threads=24, tokens=128,
                  repetitions=2 if args.stage == "screen" else 3,
                  build=build, extra=extra)


if __name__ == "__main__":
    main()
