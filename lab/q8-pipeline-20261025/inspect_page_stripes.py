#!/usr/bin/env python3
"""Read-only samples of page-cache NUMA runs, avoiding periodic sample aliasing."""
import argparse
import collections
import ctypes
import json
import mmap
from pathlib import Path
import random
import time

from page_placement import LAB, OUT, PAGE, LIBC, move

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    model = LAB / 'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00002-of-00002.gguf'
    rng = random.Random(20261025)
    results, counts, runs = [], collections.Counter(), collections.Counter()
    started = time.monotonic()
    with model.open('rb') as stream:
        memory = mmap.mmap(stream.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        anchor = ctypes.c_char.from_buffer(memory)
        base = ctypes.addressof(anchor)
        window_pages = 2048
        window_bytes = window_pages * PAGE
        for index in range(64):
            offset = rng.randrange((len(memory) - window_bytes) // PAGE) * PAGE
            resident = (ctypes.c_ubyte * window_pages)()
            assert LIBC.mincore(base + offset, window_bytes, resident) == 0
            assert all(v & 1 for v in resident), 'This inspector only samples already cached pages'
            addresses = []
            for page in range(window_pages):
                position = offset + page * PAGE
                _ = memory[position]
                addresses.append(base + position)
            nodes = move(addresses)
            assert all(n in [0, 1] for n in nodes)
            counts.update(nodes)
            run, previous = 0, None
            local_runs = []
            for node in nodes:
                if node != previous and run:
                    local_runs.append(run)
                    run = 0
                run += 1
                previous = node
            local_runs.append(run)
            # Exclude the clipped run on each end from run-length statistics.
            runs.update(local_runs[1:-1])
            results.append({'offset': offset, 'node_counts': dict(collections.Counter(nodes)),
                            'transitions': len(local_runs) - 1, 'first_256_nodes': nodes[:256]})
        del anchor
        memory.close()
    report = {'read_only': True, 'label': args.label, 'elapsed_s': time.monotonic() - started,
              'node_counts': dict(counts), 'run_lengths_pages': dict(runs), 'samples': results}
    (OUT / (args.label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)

if __name__ == '__main__':
    main()
