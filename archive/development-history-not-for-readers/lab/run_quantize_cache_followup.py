#!/usr/bin/env python3
"""Lengthen stock cache tests to reduce short-run clock and timer effects."""
import fcntl
import json
from run_quantize_tradeoff import LAB, OUT, BIN, TYPES, run

with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert json.loads((OUT / "complete.json").read_text())["complete"]
    for repeat in range(2):
        args = [arg for kind in TYPES for arg in ("--type", kind)]
        run(f"quant-perf-cache-long-{repeat}", ["numactl", "--physcpubind=0", "--membind=0",
            str(BIN / "test-quantize-perf-fixed"), *args, "--op", "vec_dot_q",
            "--op", "dequantize_row_q", "--size", "4096", "-i", "500000"])
