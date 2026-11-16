#!/usr/bin/env python3
"""One read-only nvidia-smi XML snapshot. Never launches an engine or authorizes a run."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

MIB = 1024**2
GIB = 1024**3
MAX_XML_BYTES = 1024**2


def clock_pair() -> dict:
    lo = time.monotonic_ns()
    unix = time.time_ns()
    hi = time.monotonic_ns()
    return {"monotonic_before_ns": lo, "unix_ns": unix, "monotonic_after_ns": hi}


def mib_bytes(value: str | None) -> int | None:
    if value is None or value.strip() in ("N/A", "[N/A]", "Not Supported", ""):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s+MiB", value.strip())
    if not match:
        raise ValueError("Unexpected memory value/unit: " + repr(value))
    try:
        number = Decimal(match.group(1)) * MIB
    except InvalidOperation as exc:
        raise ValueError("Invalid memory number") from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError("Memory value cannot be represented in whole bytes")
    return int(number)


def parse_gpu_xml(data: bytes, gpu_id: str) -> dict:
    if len(data) > MAX_XML_BYTES:
        raise ValueError("Unexpectedly large nvidia-smi XML")
    root = ET.fromstring(data)
    devices = root.findall("gpu")
    if len(devices) != 1:
        raise ValueError("Expected exactly one selected GPU")
    gpu = devices[0]
    uuid = gpu.findtext("uuid")
    if not uuid or (gpu_id.startswith("GPU-") and gpu_id != uuid):
        raise ValueError("GPU UUID missing or mismatched")
    memory_raw = {k: gpu.findtext("fb_memory_usage/" + k) for k in ("total", "used", "free", "reserved")}
    memory = {k + "_bytes": mib_bytes(v) for k, v in memory_raw.items()}
    if memory["total_bytes"] is not None:
        for field in ("used_bytes", "free_bytes", "reserved_bytes"):
            if memory[field] is not None and memory[field] > memory["total_bytes"]:
                raise ValueError("Memory field exceeds reported total")
    # Preserve source names. Do not silently substitute one power concept for another.
    power = {p: gpu.findtext(p) for p in (
        "gpu_power_readings/current_power_limit", "gpu_power_readings/power_limit",
        "power_readings/power_limit", "gpu_power_readings/enforced_power_limit")}
    return {
        "requested_id": gpu_id, "uuid": uuid, "pci_id": gpu.get("id"),
        "product_name": gpu.findtext("product_name"),
        "driver_version": root.findtext("driver_version"),
        "memory_raw": memory_raw, "memory": memory,
        "power_fields_raw": power, "ecc_mode_current": gpu.findtext("ecc_mode/current_ecc"),
    }


def screen(free: int | None, payload: int | None) -> dict:
    # These are source-derived thresholds, not new authorization or admission logic.
    threshold = None if payload in (None, 0) else payload + 3072 * MIB
    return {
        "candidate_payload_bytes": payload,
        "positive_payload_requires_free_strictly_greater_than_payload_plus_bytes": 3072 * MIB,
        "reported_free_exceeds_3072_MiB": None if free is None else free > 3072 * MIB,
        "candidate_preallocation_threshold_bytes": threshold,
        "reported_free_passes_candidate_memory_inequality": None if free is None or threshold is None else free > threshold,
        "zero_payload_skips_this_positive_payload_preallocation_guard": payload == 0,
        "existing_postallocation_free_must_exceed_bytes": 2560 * MIB,
        "postallocation_guard_evaluated": False,
        "runtime_admission_or_launch_authorized": False,
        "interpretation": "NVML MiB reporting is a coarse snapshot. Actual cudaMemGetInfo, postallocation and existing guards remain authoritative. A true flag never starts or permits a run.",
    }


def host_memory() -> dict:
    before = time.monotonic_ns()
    try:
        text = Path("/proc/meminfo").read_text()
        entries = dict(re.findall(r"^(MemTotal|MemAvailable):\s+([0-9]+) kB$", text, re.MULTILINE))
        missing = [name for name in ("MemTotal", "MemAvailable") if name not in entries]
        values = {name + "_bytes": int(entries[name]) * 1024 if name in entries else None
                  for name in ("MemTotal", "MemAvailable")}
        result = {"values": values, "missing_fields": missing, "complete": not missing,
                  "error": "Required fields missing or unrecognized: " + ", ".join(missing) if missing else None}
    except OSError as exc:
        result = {"values": {"MemTotal_bytes": None, "MemAvailable_bytes": None},
                  "missing_fields": ["MemTotal", "MemAvailable"], "complete": False, "error": str(exc)}
    result.update({"read_begin_monotonic_ns": before, "read_end_monotonic_ns": time.monotonic_ns(),
                   "existing_RAM_reserve_bytes": 64 * GIB,
                   "interpretation": "Host MemAvailable only; does not establish process/cgroup allocation headroom or future reserves."})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Prefer the already verified GPU UUID; numeric index is allowed but is weaker identity.")
    parser.add_argument("--out", required=True, type=Path, help="New JSON path. Existing files are never overwritten.")
    parser.add_argument("--candidate-payload-bytes", type=int, help="Optional native GPU-bank payload only; not a compact CPU-cache byte count.")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", args.gpu):
        parser.error("Use a numeric GPU index or a GPU- UUID")
    if args.candidate_payload_bytes is not None and args.candidate_payload_bytes < 0:
        parser.error("Payload must be nonnegative")
    # Open exclusively before collecting. No model loading, settings writes or retry loop.
    with args.out.open("x", encoding="utf-8") as file:
        report = {
            "schema": 1, "purpose": "read_only_capacity_snapshot_before_designing_observations",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "clock_before": clock_pair(), "host_memory": host_memory(),
            "query": {}, "gpu": None, "error": None,
            "engine_or_API_actions": [], "GPU_settings_changed": False,
            "zero_payload_dispatcher_is_not_a_route_only_probe": True,
        }
        command = ["nvidia-smi", "--id=" + args.gpu, "-q", "-x"]
        report["query"] = {"argv": command, "begin_monotonic_ns": time.monotonic_ns()}
        free = None
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=15, check=False, env={**os.environ, "LC_ALL": "C"})
            report["query"].update({"end_monotonic_ns": time.monotonic_ns(), "returncode": result.returncode,
                "stdout_bytes": len(result.stdout), "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "stderr": result.stderr.decode("utf-8", errors="replace")[:4096]})
            if result.returncode:
                raise RuntimeError("nvidia-smi query failed; no CUDA fallback attempted")
            report["gpu"] = parse_gpu_xml(result.stdout, args.gpu)
            free = report["gpu"]["memory"]["free_bytes"]
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, ET.ParseError) as exc:
            report["error"] = type(exc).__name__ + ": " + str(exc)
            report["query"].setdefault("end_monotonic_ns", time.monotonic_ns())
        report["capacity_screen"] = screen(free, args.candidate_payload_bytes)
        report["clock_after"] = clock_pair()
        report["limitations"] = [
            "This performs one read-only nvidia-smi query, not a per-token profiler.",
            "The XML snapshot spans a query interval; its individual values are not guaranteed simultaneous.",
            "The script does not map an NVML GPU index to a CUDA_VISIBLE_DEVICES ordinal.",
            "It does not establish current API identity, GPU layer placement, monitor/fan state or complete guard compliance.",
            "No memory allocation, API request, restart, build, model download or GPU setting change is performed.",
            "Full XML is not saved; selected numeric/identity fields and a hash are retained.",
        ]
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"saved": str(args.out), "query_error": report["error"],
                      "reported_free_bytes": free, "launch_authorized": False}))
    return 0 if report["error"] is None and free is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
