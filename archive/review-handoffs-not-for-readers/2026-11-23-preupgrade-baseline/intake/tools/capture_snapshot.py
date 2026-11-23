#!/usr/bin/env python3
"""Read-only Linux snapshot against the schema-1 current/runtime.json reference.

Only the new --out directory is written. No service, inference API, sudo, build,
or profile command is executed. Exit 0 means collection completed, NOT a running
state attestation; 2 is invalid input/output; 3 means a comparison needs review.
Python standard library only. Run explicitly on the machine being documented.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


VERSION = "1.0"
SMALL_HASH_LIMIT = 2 * 1024 ** 3
JSON_LIMIT = 4 * 1024 ** 2
PERF_ENV = frozenset("""
OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_DYNAMIC OMP_MAX_ACTIVE_LEVELS OMP_PROC_BIND
OMP_PLACES OMP_WAIT_POLICY OMP_SCHEDULE GOMP_CPU_AFFINITY GOMP_SPINCOUNT
KMP_AFFINITY KMP_BLOCKTIME OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS
NUMEXPR_NUM_THREADS LD_LIBRARY_PATH LD_PRELOAD CUDA_VISIBLE_DEVICES
CUDA_DEVICE_ORDER CUDA_MODULE_LOADING GGML_CUDA_REGISTER_HOST
GGML_OP_OFFLOAD_MIN_BATCH GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH
GGML_CPU_IQ3_COMPACT_MIB GGML_CPU_IQ_COMPACT_MIB GGML_CPU_IQ_PLAN
GGML_CPU_IQ_ROW_LOCAL GGML_CPU_IQ_PACK_THREADS GGML_CPU_IQ_COMPACT_THREADS
GGML_CPU_IQ_MIN_FREE_BYTES GGML_CPU_IQ_COMPACT_MIN_FREE_BYTES
GGML_CPU_IQ_COMPACT_RESERVE_BYTES GGML_XEON_HOT_FFN GGML_XEON_HOT_PLAN
""".split())
VALUE_FLAGS = frozenset("""
-t --threads -tb --threads-batch --threads-draft --threads-batch-draft
--cpu-mask --cpu-mask-batch --cpu-range --cpu-range-batch --cpu-strict
--cpu-strict-batch --prio --prio-batch --poll --poll-batch -c --ctx-size
-b --batch-size -ub --ubatch-size -ngl --n-gpu-layers --gpu-layers --n-cpu-moe
-np --parallel --numa --load-mode --lazy-mode --interleave --membind
--cpunodebind --physcpubind --cache-type-k --cache-type-v --split-mode
--tensor-split --main-gpu --image-max-tokens --mtmd-batch-max-tokens
--cache-ram --ctx-checkpoints --fit -fa --flash-attn
""".split())
PATH_FLAGS = frozenset(("-m", "--model", "--mmproj"))
BOOL_FLAGS = frozenset("""
--no-repack --op-offload --no-op-offload --offline --no-cont-batching
--no-context-shift --cache-prompt --no-cache-prompt --mmproj-offload
--no-mmproj-offload --mlock --no-mmap --mmap --no-ui --no-agent --metrics
""".split())
SETTING_FLAGS = {
    "threads_decode": ("-t", "--threads"),
    "threads_prefill": ("-tb", "--threads-batch"),
    "decode_cpu_mask": ("--cpu-mask",),
    "prefill_cpu_mask": ("--cpu-mask-batch",),
    "context": ("-c", "--ctx-size"), "batch": ("-b", "--batch-size"),
    "ubatch": ("-ub", "--ubatch-size"), "parallel_slots": ("-np", "--parallel"),
    "load_mode": ("--load-mode",), "lazy_mode": ("--lazy-mode",),
}
SETTING_ENV = {
    "OMP_THREAD_LIMIT": "OMP_THREAD_LIMIT",
    "host_weight_registration": "GGML_CUDA_REGISTER_HOST",
    "offload_min_batch": "GGML_OP_OFFLOAD_MIN_BATCH",
    "mxfp4_offload_min_batch": "GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH",
}
SAFE_CONFIG_KEYS = frozenset("""
context slot_count cache_prompt cache_ram_mib ctx_checkpoints kv_type n_cpu_moe
gpu_resident_expert_layers cpu_moe
""".split())
REFERENCE_INFERENCE_KEYS = frozenset("""
context threads_decode threads_prefill decode_cpu_mask prefill_cpu_mask
OMP_THREAD_LIMIT batch ubatch kv_type GPU_routed_layers cpu_routed_count
host_weight_registration interleave_nodes THP_disabled_by_wrapper
offload_min_batch mxfp4_offload_min_batch packing_threads_at_load
compact_packing_host_reserve_bytes load_mode lazy_mode fan_mode
fan_limit_percent image_max_tokens parallel_slots
""".split())
REFERENCE_HARDWARE_KEYS = frozenset("""
cpu sockets physical logical one_cpu_per_physical_core board bios populated_dimms
dimm_sizes configured_memory_speeds ram_GiB_installed GPU GPU_limit_W gpu_ecc
""".split())
THREAD_STATUS_KEYS = frozenset("""
Name State Tgid Pid PPid TracerPid Threads Cpus_allowed Cpus_allowed_list
Mems_allowed Mems_allowed_list voluntary_ctxt_switches nonvoluntary_ctxt_switches
""".split())


def stamp():
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
            "monotonic_ns": time.monotonic_ns()}


def failure(exc):
    if isinstance(exc, FileNotFoundError):
        status = "missing"
    elif isinstance(exc, PermissionError):
        status = "permission_denied"
    else:
        status = "error"
    return {"status": status, "error_type": type(exc).__name__,
            "errno": getattr(exc, "errno", None)}


def read_text(path):
    try:
        return {"status": "ok", "value": Path(path).read_text(errors="replace").strip()}
    except OSError as exc:
        return failure(exc)


def fields(text):
    return {k.strip(): v.strip() for line in text.splitlines()
            if ":" in line for k, v in (line.split(":", 1),)}


def stat_record(value):
    return {"device": f"{os.major(value.st_dev):x}:{os.minor(value.st_dev):x}",
            "device_number": value.st_dev, "inode": value.st_ino,
            "size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "mode": stat.S_IFMT(value.st_mode)}


def fingerprint(path, expected=None, rehash=True, limit=SMALL_HASH_LIMIT):
    """Compare both the opened inode and the path before/after a content read."""
    expected = expected or {}
    result = {"path": str(path), "expected_sha256": expected.get("sha256"),
              "expected_size": expected.get("size"), "contents_rehashed": False,
              "content_matches_expected": None, "matches_expected": None,
              "collection_start": stamp()}
    try:
        before = os.stat(path)
        result["stat_before"] = stat_record(before)
        if not stat.S_ISREG(before.st_mode):
            result.update(status="not_regular_file", matches_expected=False)
            return result
        if expected.get("size") is not None:
            result["size_matches_expected"] = before.st_size == expected["size"]
        if rehash and limit is not None and before.st_size > limit:
            result.update(status="not_measured", reason="small_file_hash_limit_exceeded")
            return result
        if rehash:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    result.update(status="not_regular_file", matches_expected=False)
                    return result
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 ** 2), b""):
                    digest.update(chunk)
                finished = os.fstat(stream.fileno())
            result["fd_stat_before"] = stat_record(opened)
            result["fd_stat_after"] = stat_record(finished)
            result["sha256"] = digest.hexdigest()
            result["contents_rehashed"] = True
        after = os.stat(path)
        result["stat_after"] = stat_record(after)
        stable = stat_record(before) == stat_record(after)
        if rehash:
            stable = stable and stat_record(before) == stat_record(opened) == stat_record(finished)
        result["stable_during_read"] = stable
        if not stable:
            result.update(status="changed_during_read", matches_expected=False,
                          content_matches_expected=None)
        elif rehash:
            match = (result["sha256"] == expected["sha256"].lower()
                     if expected.get("sha256") else None)
            result["content_matches_expected"] = match
            result["matches_expected"] = (match and result.get("size_matches_expected", True)
                                          if match is not None else None)
            result["status"] = ("match" if result["matches_expected"] else "mismatch"
                                if match is not None else "hashed_without_reference")
        else:
            result["status"] = ("size_mismatch" if result.get("size_matches_expected") is False
                                else "metadata_only")
            if result.get("size_matches_expected") is False:
                result["matches_expected"] = False
    except OSError as exc:
        result.update(failure(exc))
    finally:
        result["collection_end"] = stamp()
    return result


def read_json_stable(path):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("JSON input must be a regular file")
        before = stat_record(os.fstat(stream.fileno()))
        data = stream.read(JSON_LIMIT + 1)
        after = stat_record(os.fstat(stream.fileno()))
    if len(data) > JSON_LIMIT:
        raise ValueError("JSON input exceeds the 4 MiB safety limit")
    if before != after or after != stat_record(os.stat(path)):
        raise ValueError("JSON input changed while being read")
    return json.loads(data), hashlib.sha256(data).hexdigest(), before


def profile_targets(profile):
    if not isinstance(profile, dict) or profile.get("schema") != 1:
        raise ValueError("Expected a schema-1 current/runtime.json reference")
    try:
        small = [("service_config", profile["service_config"]),
                 ("server_executable", profile["server_executable"])]
        small += [("libraries." + name, profile["libraries"][name])
                  for name in ("cpu", "llama", "cuda", "base")]
        small += [("plan", profile["plan"])]
        small += [(group + ".download_receipt", profile[group]["download_receipt"])
                  for group in ("model", "vision")]
        weights = [(group + ".files." + str(i), item)
                   for group in ("model", "vision")
                   for i, item in enumerate(profile[group]["files"])]
        if not profile["model"]["files"] or not profile["vision"]["files"]:
            raise ValueError("Both model and vision weight lists are required")
        for _, item in small + weights:
            path = Path(item["file"])
            if path.is_absolute() or ".." in path.parts or str(path) == ".":
                raise ValueError("Reference file paths must be relative to --lab-root without '..'")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"]):
                raise ValueError("Reference SHA256 must contain exactly 64 hexadecimal digits")
            if "size" in item and (type(item["size"]) is not int or item["size"] < 0):
                raise ValueError("Reference size must be a non-negative integer")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete or malformed reference target list") from exc
    return small, weights


def run_readonly(argv, timeout=15):
    result = {"argv": argv, "collection_start": stamp()}
    executable = shutil.which(argv[0])
    if executable is None:
        result.update(status="not_installed", collection_end=stamp())
        return result
    try:
        proc = subprocess.run([executable] + argv[1:], shell=False, check=False,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              env={**os.environ, "LC_ALL": "C"})
        result.update(status="ok" if proc.returncode == 0 else "command_failed",
                      returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        result.update(status="timeout", timeout_seconds=timeout)
    except OSError as exc:
        result.update(failure(exc))
    result["collection_end"] = stamp()
    return result


def cpulist(value):
    output = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo or hi - lo > 65536:
                raise ValueError("Invalid CPU list range")
            output.update(range(lo, hi + 1))
        else:
            output.add(int(part))
    return sorted(output)


def affinity(tid=0):
    try:
        return {"status": "ok", "cpus": sorted(os.sched_getaffinity(tid))}
    except (OSError, AttributeError) as exc:
        return failure(exc)


def memory_modules():
    command = run_readonly(["dmidecode", "--type", "17"])
    raw = command.pop("stdout", "")
    modules = []
    keep = {"Size", "Locator", "Bank Locator", "Type", "Type Detail", "Speed",
            "Configured Memory Speed", "Configured Clock Speed", "Total Width",
            "Data Width", "Rank", "Manufacturer", "Part Number"}
    for block in re.split(r"\nHandle ", raw):
        if "Memory Device\n" not in block:
            continue
        record = {k: v for k, v in fields(block).items() if k in keep}
        if record:
            modules.append(record)
    populated = [m for m in modules if m.get("Size", "").lower()
                 not in ("", "no module installed", "not installed", "unknown", "0 mb", "0 gb")]
    speeds = sorted({m[key] for m in populated for key in
                     ("Configured Memory Speed", "Configured Clock Speed")
                     if key in m and re.search(r"\d", m[key])})
    return {"status": "ok" if command["status"] == "ok" and populated and speeds else "partial",
            "command": command, "modules": modules, "populated_count": len(populated)
            if modules else None, "configured_speeds": speeds,
            "configured_speed_obtained": bool(speeds),
            "note": "SMBIOS reported configured rate, not measured memory bandwidth; no sudo used."}


def gpu_info():
    command = run_readonly(["nvidia-smi", "-q", "-x"])
    raw = command.pop("stdout", "")
    result = {"status": command["status"], "command": command, "devices": [],
              "note": "Temperature/clocks/power draw are snapshot readings, not load-effective frequency or a benchmark."}
    if command["status"] != "ok":
        return result
    wanted = {
        "name": ("product_name",), "bus_id": ("pci/pci_bus_id",),
        "vram_total": ("fb_memory_usage/total",),
        "power_limit": ("power_readings/power_limit", "gpu_power_readings/power_limit", "gpu_power_readings/requested_power_limit"),
        "enforced_power_limit": ("power_readings/enforced_power_limit", "gpu_power_readings/enforced_power_limit"),
        "default_power_limit": ("power_readings/default_power_limit", "gpu_power_readings/default_power_limit"),
        "minimum_power_limit": ("power_readings/min_power_limit", "gpu_power_readings/min_power_limit"),
        "maximum_power_limit": ("power_readings/max_power_limit", "gpu_power_readings/max_power_limit"),
        "power_draw": ("power_readings/power_draw", "gpu_power_readings/power_draw", "gpu_power_readings/instant_power_draw"),
        "ecc_current": ("ecc_mode/current_ecc",), "ecc_pending": ("ecc_mode/pending_ecc",),
        "temperature_gpu": ("temperature/gpu_temp",),
        "graphics_clock_snapshot": ("clocks/graphics_clock",),
        "sm_clock_snapshot": ("clocks/sm_clock",),
        "memory_clock_snapshot": ("clocks/mem_clock",), "performance_state": ("performance_state",),
    }
    try:
        root = ET.fromstring(raw)
        result["driver_version"] = root.findtext("driver_version")
        for gpu in root.findall("gpu"):
            item = {}
            for key, alternatives in wanted.items():
                value = next((gpu.findtext(p) for p in alternatives if gpu.findtext(p) is not None), None)
                item[key] = {"status": "not_available" if value in (None, "N/A", "Not Supported", "Unknown")
                             else "ok", "value": value}
            result["devices"].append(item)
        required = ("name", "bus_id", "vram_total", "power_limit", "ecc_current")
        if not result["devices"] or not result["driver_version"] or any(
                item[key]["status"] != "ok" for item in result["devices"] for key in required):
            result["status"] = "partial"
    except ET.ParseError:
        result["status"] = "parse_error"
    return result


def collect_hardware(proc_root=Path("/proc"), sys_root=Path("/sys")):
    cpu_root = sys_root / "devices/system/cpu"
    cpuinfo = read_text(proc_root / "cpuinfo")
    processors = []
    for block in cpuinfo.get("value", "").split("\n\n"):
        item = fields(block)
        if "processor" in item:
            processors.append({k: item[k] for k in ("processor", "model name", "vendor_id",
                               "cpu family", "model", "stepping", "microcode", "flags", "cpu MHz") if k in item})
    topology = []
    for path in sorted(cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        entry = {"cpu": int(path.name[3:]), "topology": {}, "cpufreq": {}}
        for key in ("physical_package_id", "die_id", "core_id", "thread_siblings_list", "core_siblings_list"):
            entry["topology"][key] = read_text(path / "topology" / key)
        entry["online_file"] = read_text(path / "online")
        for key in ("scaling_governor", "scaling_driver", "scaling_cur_freq", "cpuinfo_cur_freq", "scaling_min_freq", "scaling_max_freq"):
            entry["cpufreq"][key] = read_text(path / "cpufreq" / key)
        topology.append(entry)
    global_cpu = {key: read_text(cpu_root / key) for key in ("online", "possible", "present", "smt/active", "smt/control")}
    counts = {"logical_online": None, "physical_cores_online": None, "sockets_online": None}
    try:
        online = cpulist(global_cpu["online"]["value"])
        cores = {(e["topology"]["physical_package_id"]["value"], e["topology"]["core_id"]["value"])
                 for e in topology if e["cpu"] in online}
        counts.update(logical_online=len(online), physical_cores_online=len(cores),
                      sockets_online=len({s for s, _ in cores}))
    except (KeyError, ValueError):
        pass
    nodes_root = sys_root / "devices/system/node"
    nodes = [{"node": path.name, **{key: read_text(path / key)
              for key in ("cpulist", "distance", "meminfo")}}
             for path in sorted(nodes_root.glob("node[0-9]*"))]
    temperatures = []
    for path in sorted((sys_root / "class/hwmon").glob("hwmon*")):
        for sensor in sorted(path.glob("temp*_input")):
            temperatures.append({"hwmon": path.name, "device": read_text(path / "name"),
                                 "sensor": sensor.name, "label": read_text(path / sensor.name.replace("_input", "_label")),
                                 "millidegrees_celsius_snapshot": read_text(sensor)})
    dmi = {key: read_text(sys_root / "class/dmi/id" / key) for key in
           ("bios_vendor", "bios_version", "bios_date", "sys_vendor", "product_name",
            "product_version", "board_vendor", "board_name", "board_version")}
    memory = read_text(proc_root / "meminfo")
    if memory["status"] == "ok":
        memory["fields"] = fields(memory.pop("value"))
    return {
        "collection_scope": "Read-only snapshot on the host that executes this tool.",
        "uname": dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())),
        "cpu": {"cpuinfo_status": cpuinfo["status"], "processors": processors,
                "sysfs_topology": topology, "global": global_cpu, "counts": counts,
                "collector_affinity": affinity(), "lscpu": run_readonly(["lscpu", "--json"]),
                "clock_note": "cpu MHz and cpufreq are instantaneous reports; load-effective frequency was not measured."},
        "meminfo": memory, "numa": {"nodes": nodes, "online": read_text(nodes_root / "online")},
        "tuning": {"numa_balancing": read_text(proc_root / "sys/kernel/numa_balancing"),
                   "transparent_hugepage": {key: read_text(sys_root / "kernel/mm/transparent_hugepage" / key)
                                            for key in ("enabled", "defrag", "shmem_enabled", "use_zero_page")}},
        "bios_board": dmi, "dimm": memory_modules(), "gpu": gpu_info(),
        "temperatures": {"status": "ok" if temperatures else "not_available", "sensors": temperatures,
                         "note": "Snapshot values; no workload or thermal soak was run."},
    }


def safe_environment(environment):
    return {key: str(environment[key]) for key in sorted(PERF_ENV) if key in environment}


def sanitize_command(argv):
    """Keep known performance arguments; unknown/auth values are never emitted."""
    if not argv:
        return [], {}
    cleaned, options = [str(argv[0])], {}
    i = 1
    while i < len(argv):
        item = str(argv[i])
        flag, sep, inline = item.partition("=")
        if flag in VALUE_FLAGS or flag in PATH_FLAGS:
            if sep:
                value = inline
            elif i + 1 < len(argv):
                i += 1
                value = str(argv[i])
            else:
                value = "<missing>"
            # A URL or embedded credentials are not valid performance scalars.
            if re.search(r"(?i)(https?://|bearer\s|(?:token|password|api[_-]?key|secret)\s*[:=])", value):
                value = "<redacted>"
            cleaned.extend((flag, value))
            options[flag] = value
        elif flag in BOOL_FLAGS and not sep:
            cleaned.append(flag)
            options[flag] = True
        elif (flag.startswith("-") and re.search(r"(?i)(api[_-]?key|token|password|passwd|secret|auth|credential|header)", flag)) or flag in ("-k", "-H"):
            cleaned.append(flag + ("=<redacted>" if sep else ""))
            if not sep and i + 1 < len(argv):
                i += 1
                cleaned.append("<redacted>")
        elif item.startswith("-"):
            cleaned.append(flag + ("=<redacted>" if sep else ""))
            if not sep and i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                i += 1
                cleaned.append("<redacted>")
        else:
            cleaned.append("<unclassified argument redacted>")
        i += 1
    return cleaned, options


def settings_comparison(profile, options, environment):
    comparisons = []
    for key in list(SETTING_FLAGS) + list(SETTING_ENV):
        if key not in profile.get("inference", {}):
            continue
        expected = profile["inference"][key]
        value = next((options[k] for k in SETTING_FLAGS[key] if k in options), None) if key in SETTING_FLAGS else environment.get(SETTING_ENV[key])
        comparable = value is not None and value not in ("<redacted>", "<missing>")
        observed = value
        if comparable:
            try:
                if key.endswith("cpu_mask"):
                    matched = int(str(expected), 16) == int(str(value), 16)
                elif isinstance(expected, int):
                    observed = int(value)
                    matched = expected == observed
                else:
                    matched = str(expected) == str(value)
            except (ValueError, TypeError):
                matched = False
        else:
            matched = None
        comparisons.append({"setting": key, "expected": expected, "observed": observed,
                            "matches": matched, "status": "match" if matched else "mismatch"
                            if matched is False else "not_observed"})
    return comparisons


def config_observation(path, profile, hashed_config):
    result = {"path": str(path), "scope": "Configured values only; not proof of effective runtime settings."}
    try:
        data, digest, metadata = read_json_stable(path)
        if not isinstance(data, dict):
            raise ValueError("Expected config object")
        argv = data.get("command", [])
        if isinstance(argv, str):
            argv = shlex.split(argv)
        if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
            raise ValueError("Expected command list")
        environment = data.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("Expected environment object")
        command, options = sanitize_command(argv)
        selected_env = safe_environment(environment)
        result.update(status="ok", sha256_of_parsed_bytes=digest, stat=metadata,
                      same_bytes_as_file_fingerprint=digest == hashed_config.get("sha256"),
                      command_redacted=command, performance_options=options,
                      performance_environment=selected_env,
                      selected_config={k: data[k] for k in sorted(SAFE_CONFIG_KEYS) if k in data},
                      comparisons=settings_comparison(profile, options, selected_env))
        if not result["same_bytes_as_file_fingerprint"]:
            result["status"] = "changed_between_observations"
    except (OSError, ValueError, TypeError) as exc:
        result.update(failure(exc))
    return result


def proc_stat(path):
    result = read_text(path)
    if result["status"] != "ok":
        return result
    try:
        raw = result["value"]
        end = raw.rfind(")")
        rest = raw[end + 2:].split()
        return {"status": "ok", "pid": int(raw.split(" ", 1)[0]),
                "state": rest[0], "utime_ticks": int(rest[11]), "stime_ticks": int(rest[12]),
                "num_threads": int(rest[17]), "starttime_ticks": int(rest[19]),
                "last_cpu": int(rest[36]), "clock_ticks_per_second": os.sysconf("SC_CLK_TCK")}
    except (ValueError, IndexError):
        return {"status": "parse_error"}


def maps_entries(text):
    groups = {}
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), parts[5])
        clean_path = path.removesuffix(" (deleted)") if hasattr(str, "removesuffix") else re.sub(r" \(deleted\)$", "", path)
        if not re.search(r"\.so(?:\.|$)", Path(clean_path).name):
            continue
        key = (parts[3], parts[4], path)
        if key not in groups:
            groups[key] = {"path": clean_path, "deleted": path.endswith(" (deleted)"),
                           "mapped_device": parts[3], "mapped_inode": int(parts[4]), "segments": []}
        groups[key]["segments"].append({"range": parts[0], "permissions": parts[1], "offset": parts[2]})
    return list(groups.values())


def device_equal(left, right):
    try:
        return tuple(int(x, 16) for x in left.split(":")) == tuple(int(x, 16) for x in right.split(":"))
    except (AttributeError, ValueError):
        return False


def mapped_fingerprint(entry, base, expected):
    attempts = []
    candidates = [base / "map_files" / entry["segments"][0]["range"],
                  base / "root" / entry["path"].lstrip("/")]
    for path in candidates:
        try:
            observed = stat_record(os.stat(path))
            agrees = (observed["inode"] == entry["mapped_inode"]
                      and device_equal(observed["device"], entry["mapped_device"]))
            if not agrees:
                attempts.append({"path": str(path), "status": "mapped_inode_or_device_mismatch"})
                continue
            result = fingerprint(path, expected)
            actual = result.get("fd_stat_before", {})
            bound = actual.get("inode") == entry["mapped_inode"] and device_equal(actual.get("device"), entry["mapped_device"])
            result["matches_mapped_device_inode"] = bound
            result["attempts"] = attempts
            result["memory_bytes_hashed"] = False
            if not bound:
                result.update(status="mapped_identity_unverified", matches_expected=False)
            return result
        except OSError as exc:
            attempts.append({"path": str(path), **failure(exc)})
    return {"status": "mapped_identity_unverified", "matches_expected": False,
            "contents_rehashed": False, "attempts": attempts, "memory_bytes_hashed": False}


def process_observation(pid, profile, proc_root=Path("/proc")):
    base = proc_root / str(pid)
    result = {"pid": pid, "collection_start": stamp(), "stat_before": proc_stat(base / "stat"),
              "runtime_memory_attested": False, "decode_team_proven": False,
              "note": "Thread count, affinity and last_cpu do not identify the active decode team. File hashes do not hash mapped memory."}
    try:
        result["exe_link"] = os.readlink(base / "exe")
    except OSError as exc:
        result["exe_link_error"] = failure(exc)
    result["executable"] = fingerprint(base / "exe", profile["server_executable"])
    options, environment = {}, {}
    try:
        argv = (base / "cmdline").read_bytes().rstrip(b"\0").decode("utf-8", "replace").split("\0")
        command, options = sanitize_command(argv)
        result["command"] = {"status": "ok", "redacted_argv": command, "performance_options": options}
    except OSError as exc:
        result["command"] = failure(exc)
    try:
        raw_env = (base / "environ").read_bytes().split(b"\0")
        environment = {k.decode("utf-8", "replace"): v.decode("utf-8", "replace")
                       for item in raw_env if b"=" in item for k, v in (item.split(b"=", 1),)
                       if k.decode("utf-8", "replace") in PERF_ENV}
        result["performance_environment"] = {"status": "ok", "values": safe_environment(environment)}
    except OSError as exc:
        result["performance_environment"] = failure(exc)
    result["setting_comparisons"] = settings_comparison(profile, options, environment)
    maps = read_text(base / "maps")
    result["maps_status"] = maps["status"]
    result["mapped_dsos"] = maps_entries(maps.get("value", ""))
    result["profile_library_comparisons"] = []
    for name, expected in profile["libraries"].items():
        if name not in ("cpu", "llama", "cuda", "base"):
            continue
        family = Path(expected["file"]).name.split(".so", 1)[0] + ".so"
        matching = [m for m in result["mapped_dsos"] if Path(m["path"]).name == family
                    or Path(m["path"]).name.startswith(family + ".")]
        for item in matching:
            item["reference_library"] = name
            item["file_fingerprint"] = mapped_fingerprint(item, base, expected)
        maps_observed = maps["status"] == "ok"
        good = (len(matching) == 1 and matching[0]["file_fingerprint"].get("matches_expected") is True) if maps_observed else None
        result["profile_library_comparisons"].append({"library": name, "candidates": len(matching) if maps_observed else None,
            "expected_sha256": expected["sha256"], "matches": good,
            "status": "not_observed" if not maps_observed else "match" if good else "not_mapped" if not matching else "unverified_or_mismatch"})
    for item in result["mapped_dsos"]:
        if "file_fingerprint" not in item:
            item["file_fingerprint"] = {"status": "not_requested", "contents_rehashed": False,
                                        "reason": "No selected reference hash for this DSO"}
    threads = []
    try:
        tids = sorted(int(p.name) for p in (base / "task").iterdir() if p.name.isdigit())
        for tid in tids:
            task = base / "task" / str(tid)
            before = proc_stat(task / "stat")
            status_read = read_text(task / "status")
            observed_affinity = affinity(tid)
            after = proc_stat(task / "stat")
            stable = before.get("starttime_ticks") is not None and before.get("starttime_ticks") == after.get("starttime_ticks")
            threads.append({"tid": tid, "stat_before": before, "stat_after": after,
                "tid_identity_stable": stable, "affinity": observed_affinity,
                "status": {"status": status_read["status"], "fields": {
                    k: v for k, v in fields(status_read.get("value", "")).items() if k in THREAD_STATUS_KEYS}}})
        after_tids = sorted(int(p.name) for p in (base / "task").iterdir() if p.name.isdigit())
        result["thread_inventory"] = {"status": "ok" if tids == after_tids and all(
            t["tid_identity_stable"] and t["affinity"]["status"] == "ok" and t["status"]["status"] == "ok" for t in threads)
            else "partial", "tids_before": tids, "tids_after": after_tids, "thread_set_stable": tids == after_tids,
            "threads": threads}
    except OSError as exc:
        result["thread_inventory"] = {**failure(exc), "threads": threads}
    result["stat_after"] = proc_stat(base / "stat")
    result["pid_identity_stable"] = result["stat_before"].get("starttime_ticks") is not None and result["stat_before"].get("starttime_ticks") == result["stat_after"].get("starttime_ticks")
    result["status"] = "ok" if (result["pid_identity_stable"] and result["maps_status"] == "ok"
        and result["command"]["status"] == "ok" and result["performance_environment"]["status"] == "ok"
        and result["thread_inventory"]["status"] == "ok") else "partial"
    result["collection_end"] = stamp()
    return result


def hardware_comparisons(profile, hardware):
    expected = profile.get("hardware", {})
    counts = hardware["cpu"]["counts"]
    actual = {"sockets": counts["sockets_online"], "physical": counts["physical_cores_online"],
              "logical": counts["logical_online"]}
    models = sorted({p["model name"] for p in hardware["cpu"]["processors"] if "model name" in p})
    actual["cpu"] = models[0] if len(models) == 1 else models if models else None
    actual["bios"] = hardware["bios_board"].get("bios_version", {}).get("value")
    actual["populated_dimms"] = hardware["dimm"].get("populated_count")
    actual["configured_memory_speeds"] = hardware["dimm"].get("configured_speeds") or None
    gpu_devices = hardware["gpu"].get("devices", [])
    actual["GPU_limit_W"], actual["gpu_ecc"] = None, None
    if len(gpu_devices) == 1:
        gpu = gpu_devices[0]
        limit = gpu.get("power_limit", {})
        if limit.get("status") == "ok":
            try:
                actual["GPU_limit_W"] = float(limit["value"].split()[0])
            except (TypeError, ValueError, IndexError):
                pass
        ecc = gpu.get("ecc_current", {})
        if ecc.get("status") == "ok":
            actual["gpu_ecc"] = ecc.get("value")
    return [{"field": key, "expected": expected[key], "observed": actual[key],
             "matches": (sorted(actual[key]) == sorted(expected[key]) if key == "configured_memory_speeds"
                         else actual[key] == expected[key]) if actual[key] is not None else None,
             "comparison": "Reported values; a text-unit difference in DIMM speed needs manual interpretation."}
            for key in actual if key in expected]


def make_summary(report):
    small, weights = report["files"]["small"], report["files"]["weights"]
    findings = []
    for item in small:
        if item.get("matches_expected") is not True:
            findings.append({"kind": "small_file_unverified_or_different", "target": item["reference_key"], "status": item["status"]})
    for item in weights:
        if item.get("size_matches_expected") is False or item["status"] not in ("metadata_only", "match"):
            findings.append({"kind": "weight_unverified_or_different", "target": item["reference_key"], "status": item["status"]})
    for item in report["hardware_comparisons"]:
        if item["matches"] is False:
            findings.append({"kind": "hardware_difference", **item})
    config = report["config_observation"]
    if config["status"] != "ok":
        findings.append({"kind": "config_observation_incomplete", "status": config["status"]})
    for item in config.get("comparisons", []):
        if item["matches"] is False:
            findings.append({"kind": "configured_setting_difference", **item})
    process = report["process"]
    if process["status"] != "not_requested":
        if process["status"] != "ok" or process.get("pid_identity_stable_across_collection") is not True:
            findings.append({"kind": "process_observation_incomplete"})
        if process["executable"].get("matches_expected") is not True:
            findings.append({"kind": "running_executable_unverified_or_different", "status": process["executable"]["status"]})
        for item in process["profile_library_comparisons"]:
            if not item["matches"]:
                findings.append({"kind": "mapped_library_unverified_or_different", **item})
        for item in process["setting_comparisons"]:
            if item["matches"] is False:
                findings.append({"kind": "process_setting_difference", **item})
    optional_gaps = []
    for name in ("dimm", "gpu", "temperatures"):
        if report["hardware"][name]["status"] != "ok":
            optional_gaps.append(name)
    if any(v["status"] != "ok" for v in report["hardware"]["bios_board"].values()):
        optional_gaps.append("bios_board")
    if report["hardware"]["cpu"]["cpuinfo_status"] != "ok" or any(
            value is None for value in report["hardware"]["cpu"]["counts"].values()):
        optional_gaps.append("cpu_topology")
    if report["hardware"]["cpu"]["lscpu"]["status"] != "ok":
        optional_gaps.append("lscpu")
    if report["hardware"]["meminfo"]["status"] != "ok" or not report["hardware"]["numa"]["nodes"]:
        optional_gaps.append("memory_or_numa_topology")
    content_matches = [f.get("content_matches_expected") for f in weights]
    all_contents_match = (False if any(value is False for value in content_matches) else True
                          if content_matches and all(value is True for value in content_matches) else None)
    return {"status": "needs_review" if findings else "partial" if optional_gaps else "collected",
            "exit_code": 3 if findings else 0,
            "all_small_reference_files_match": all(f.get("matches_expected") is True for f in small),
            "small_files_count": len(small), "weights_count": len(weights),
            "weight_contents_rehashed": bool(weights) and all(f["contents_rehashed"] for f in weights),
            "all_weight_sizes_match": bool(weights) and all(f.get("size_matches_expected") is True for f in weights),
            "all_weight_contents_match": all_contents_match,
            "full_running_state_attested": False, "findings": findings,
            "optional_observation_gaps": optional_gaps,
            "note": "Exit 0 means collection completed. Metadata-only weights are NOT content-verified. No automatic repair was attempted."}


def capture(profile_path, lab_root, out, label, pid=None, full_weights=False):
    started = stamp()
    if out.exists() or out.is_symlink():
        raise ValueError("--out already exists; use a new directory (nothing was overwritten)")
    profile, profile_hash, profile_stat = read_json_stable(profile_path)
    small_targets, weight_targets = profile_targets(profile)
    if not lab_root.is_dir():
        raise ValueError("--lab-root must be an existing directory on the target machine")
    out.mkdir(parents=True, exist_ok=False)
    # Do not persist the whole profile: its service command is data, never executable input.
    before_pid = proc_stat(Path("/proc") / str(pid) / "stat") if pid else None
    files = {}
    for group, targets in (("small", small_targets), ("weights", weight_targets)):
        files[group] = []
        for key, expected in targets:
            item = fingerprint(lab_root / expected["file"], expected,
                               rehash=(group == "small" or full_weights),
                               limit=SMALL_HASH_LIMIT if group == "small" else None)
            item.update(reference_key=key, reference_file=expected["file"])
            files[group].append(item)
    hardware = collect_hardware()
    config = config_observation(lab_root / profile["service_config"]["file"], profile, files["small"][0])
    process = process_observation(pid, profile) if pid else {"status": "not_requested", "decode_team_proven": False}
    if pid:
        after_pid = proc_stat(Path("/proc") / str(pid) / "stat")
        process["stat_at_collection_start"] = before_pid
        process["stat_at_collection_end"] = after_pid
        process["pid_identity_stable_across_collection"] = (process["pid_identity_stable"]
            and before_pid.get("starttime_ticks") is not None
            and before_pid.get("starttime_ticks") == process["stat_before"].get("starttime_ticks")
            == after_pid.get("starttime_ticks"))
    report = {
        "schema": 1, "tool": "capture_snapshot.py", "version": VERSION, "label": label,
        "collection": {"start": started, "end": stamp(), "atomic": False,
                       "monotonic_clock_scope": "This process/boot only; not comparable across reboots."},
        "scope": {"single_snapshot_full_running_state_attestation": False,
                  "inference_or_benchmark_executed": False, "configuration_changed": False,
                  "load_effective_frequency_measured": False, "decode_team_proven": False,
                  "weight_contents_rehashed_by_default": False,
                  "no_pid_limit": "Without --pid, on-disk files/configuration do not establish what a running server loaded.",
                  "temporal_limit": "Files, hardware and threads are sampled at different instants. Other state may change between samples."},
        "reference": {"path": str(profile_path), "sha256": profile_hash, "stat": profile_stat,
                      "profile": profile.get("profile"), "verified_jst_date": profile.get("verified_jst_date"),
                      "hardware": {k: v for k, v in profile.get("hardware", {}).items() if k in REFERENCE_HARDWARE_KEYS},
                      "inference": {k: v for k, v in profile.get("inference", {}).items() if k in REFERENCE_INFERENCE_KEYS},
                      "lab_root": str(lab_root)},
        "full_weights_requested": full_weights, "files": files, "hardware": hardware,
        "hardware_comparisons": hardware_comparisons(profile, hardware),
        "config_observation": config,
        "collector_performance_environment": safe_environment(os.environ), "process": process,
    }
    report["summary"] = make_summary(report)
    with (out / "snapshot.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    summary = report["summary"]
    lines = ["READ-ONLY SNAPSHOT — " + label, "Started UTC: " + started["utc"],
             "Reference SHA256: " + profile_hash, "Status: " + summary["status"],
             "Small reference files all match: " + str(summary["all_small_reference_files_match"]).lower(),
             "Weight contents rehashed: " + str(summary["weight_contents_rehashed"]).lower(),
             "Weight sizes all match: " + str(summary["all_weight_sizes_match"]).lower(),
             "Weight contents all match: " + ("unknown" if summary["all_weight_contents_match"] is None
                                              else str(summary["all_weight_contents_match"]).lower()),
             "Full running state attested: false", "Decode team proven: false",
             "Optional observation gaps: " + (", ".join(summary["optional_observation_gaps"]) or "none"),
             "Findings: " + str(len(summary["findings"]))]
    lines.extend(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in summary["findings"])
    lines += ["", summary["note"], "See snapshot.json for expected/observed hashes, metadata and per-field status."]
    with (out / "summary.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Pinned schema-1 current/runtime.json reference")
    parser.add_argument("--lab-root", type=Path, required=True, help="Existing target-machine lab root")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--label", required=True, help="Example: old-original, old-p12, new-p24")
    parser.add_argument("--pid", type=int, help="Optional actual llama-server PID; no server is started")
    parser.add_argument("--full-weights", action="store_true", help="Explicitly hash all GGUF/projector contents (hundreds of GB of disk reads)")
    args = parser.parse_args(argv)
    if args.pid is not None and args.pid <= 0:
        parser.error("--pid must be a positive integer")
    if not args.label or len(args.label) > 128 or any(ord(c) < 32 for c in args.label):
        parser.error("--label must contain 1–128 printable characters")
    try:
        report = capture(args.profile.resolve(), args.lab_root.resolve(), args.out.absolute(),
                         args.label, args.pid, args.full_weights)
    except (OSError, ValueError) as exc:
        print("Snapshot not completed: " + str(exc), file=sys.stderr)
        return 2
    print(str(args.out.absolute() / "snapshot.json"))
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return report["summary"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
