#!/usr/bin/env python3
"""Validate actual NUMA mapping and dispatch; this is not a speed sample."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

lab = Path(__file__).resolve().parent
root = lab / "review-deep-numa-20261019"
binary = root / "build/bin/llama-bench"
out = lab / "results" / ("numa-replica-placement-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir()
model = lab / "models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf"
command = ["numactl", "--physcpubind=0-23", "--interleave=0,1", str(binary), "-m", str(model),
           "-p", "0", "-n", "32", "-d", "0", "-t", "12", "--poll", "50", "-ngl", "0",
           "-r", "1", "-o", "json", "--offline", "--progress", "-b", "512", "-ub", "512",
           "-lm", "mmap", "--cpu-mask", "0xffffff", "--cpu-strict", "1"]
report = {"command": command, "diagnostic_only": True, "adopted": False,
          "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
          "cpu_library_sha256": hashlib.sha256((root / "build/bin/libggml-cpu.so").read_bytes()).hexdigest()}
environment = dict(os.environ, GGML_CPU_NUMA_REPLICA="1", GGML_CPU_NUMA_REPLICA_TRACE="1",
                   OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1")
with (out / "bench.json").open("w") as stdout, (out / "bench.log").open("w") as stderr:
    child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
    start = time.monotonic()
    try:
        captured = False
        while child.poll() is None:
            if time.monotonic() - start > 240:
                raise TimeoutError("NUMA diagnostic exceeded 240 seconds")
            log = (out / "bench.log").read_text()
            if not captured and "generation run 1/1" in log:
                for name in ("maps", "numa_maps", "status", "io"):
                    (out / ("proc-" + name + ".txt")).write_text(Path(f"/proc/{child.pid}/{name}").read_text())
                captured = True
            time.sleep(0.5)
        report["exit_code"] = child.returncode
        report["wall_seconds"] = time.monotonic() - start
        if child.returncode or not captured:
            raise RuntimeError("No complete NUMA diagnostic; inspect bench.log")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
log = (out / "bench.log").read_text()
regions = []
vmas = []
for line in (out / "proc-maps.txt").read_text().splitlines():
    lo, hi = (int(v, 16) for v in line.split()[0].split("-"))
    vmas.append((lo, hi))
numa_lines = {int(line.split()[0], 16): line for line in (out / "proc-numa_maps.txt").read_text().splitlines()}
for node_text, address_text, size_text in re.findall(r"NUMA_REPLICA_MAP node=(\d+) address=(0x[0-9a-f]+) bytes=(\d+)", log):
    node, address, size = int(node_text), int(address_text, 16), int(size_text)
    matches = [(lo, hi) for lo, hi in vmas if lo <= address and hi >= address + size]
    if len(matches) != 1: raise RuntimeError("Replica range is not present in one VMA")
    numa = numa_lines[matches[0][0]]
    pages = {int(n): int(count) for n, count in re.findall(r"\bN(\d+)=(\d+)", numa)}
    local = pages.get(node, 0) > 0 and all(n == node or count == 0 for n, count in pages.items())
    regions.append({"node": node, "address": address_text, "bytes": size, "numa_maps": numa,
                    "pages_by_node": pages, "all_resident_pages_local": local})
report["regions"] = regions
report["tensor_records"] = [dict(name=name, bytes=int(size), offset=int(offset)) for name, size, offset in
    re.findall(r"NUMA_REPLICA_TENSOR (\S+) bytes=(\d+) offset=(\d+)", log)]
report["calls"] = [dict(node0=int(a), node1=int(b), fallback=int(c)) for a, b, c in
    re.findall(r"NUMA_REPLICA_CALLS node0=(\d+) node1=(\d+) fallback=(\d+)", log)]
report["all_regions_local"] = bool(regions) and all(region["all_resident_pages_local"] for region in regions)
report["sum_replica_bytes"] = sum(region["bytes"] for region in regions)
report["tensor_count"] = len(report["tensor_records"])
report["selected_tensor_bytes"] = sum(t["bytes"] for t in report["tensor_records"])
(out / "placement.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"saved": str(out / "placement.json"), "tensor_count": report["tensor_count"],
                  "all_regions_local": report["all_regions_local"], "calls": report["calls"]}), flush=True)
assert report["all_regions_local"]
assert report["tensor_count"] == 107 and report["selected_tensor_bytes"] == 3253207040
assert sum(x["node0"] for x in report["calls"]) > 0 and sum(x["node1"] for x in report["calls"]) > 0
assert sum(x["fallback"] for x in report["calls"]) == 0
