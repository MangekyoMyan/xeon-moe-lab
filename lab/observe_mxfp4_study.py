#!/usr/bin/env python3
"""Reuse the prior steady-state CPU/DRAM counter observer."""
import fcntl
import json

import observe_quantize_tradeoff as observer
from run_mxfp4_study import LAB, OUT, BIN


def main():
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert json.loads((OUT / "complete.json").read_text())["complete"]
        observer.OUT = OUT
        observer.BIN = BIN
        for kind in ("q8_0", "q4_K", "mxfp4"):
            print("Capturing " + kind, flush=True)
            observer.capture(kind)


if __name__ == "__main__":
    main()
