#!/usr/bin/env python3
"""Check whether a larger prompt microbatch benefits the resident Q8 model."""
import fcntl
from glm_q8_experiment import OUT, run

if __name__ == "__main__":
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run("prefill-b512", threads="12,24", prompt=512, tokens=0,
            repetitions=2, batch=512, timeout=3600)
