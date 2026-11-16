"""Read current configuration/hardware once; never start or stop an API."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
ENGINE = LAB / "deepseek41-bringup-20261101"
PROFILE_KEYS = {
    "command", "environment", "model", "host", "port", "model_alias", "context",
    "slot_count", "cpu_moe", "cache_prompt", "cache_ram_mib", "ctx_checkpoints",
    "kv_type", "tested_on_real_model", "source_revision", "n_cpu_moe",
    "gpu_resident_expert_layers", "gpu_stream_profile", "cpu_iq3_compact_profile",
    "vision_profile", "study_libraries", "optimization_study", "study_files",
}


def run(args, timeout=10):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def safe_argv(argv):
    result = []
    hide_next = False
    for item in argv:
        if hide_next:
            result.append("<REDACTED>")
            hide_next = False
        elif item in ("--api-key", "--password", "--token", "--hf-token"):
            result.append(item)
            hide_next = True
        elif any(item.startswith(k + "=") for k in ("--api-key", "--password", "--token", "--hf-token")):
            result.append(item.split("=", 1)[0] + "=<REDACTED>")
        else:
            result.append(item)
    return result


def safe_env(env):
    return {k: v for k, v in env.items() if k.startswith(("GGML_", "OMP_", "GOMP_", "XEON_"))
            or k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "CUDA_MODULE_LOADING")}


def profile(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    result = {k: v for k, v in data.items() if k in PROFILE_KEYS}
    result["command"] = safe_argv(result.get("command", []))
    result["environment"] = safe_env(result.get("environment", {}))
    result["sharing_note"] = "API key file/value omitted; allowlisted configuration fields only."
    return {"original_path": str(path), "original_sha256": hashlib.sha256(raw).hexdigest(), "config": result}


def main():
    out = ROOT / "capture"
    out.mkdir(exist_ok=True)
    config = profile(ENGINE / "server-config.json")
    profiles = {"current": config}
    for name in ("final", "hot14-reviewed", "baseline"):
        profiles[name] = profile(LAB / "exploration-20261112/profiles" / (name + ".json"))
    (out / "PROFILES_REDACTED.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n")
    processes = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() != "llama-server":
                continue
            argv = (p / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            env = dict(part.split("=", 1) for part in (p / "environ").read_bytes().decode().split("\0") if "=" in part)
            status = dict(line.split(":", 1) for line in (p / "status").read_text().splitlines() if ":" in line)
            policies = {}
            for line in (p / "numa_maps").read_text().splitlines():
                fields = line.split()
                if len(fields) > 1:
                    policies[fields[1]] = policies.get(fields[1], 0) + 1
            expected = config["config"]
            configured_argv = expected["command"]
            actual = safe_argv(argv)
            suffix_match = any(configured_argv[i:] == actual for i in range(len(configured_argv)))
            processes.append({"pid": int(p.name), "argv": actual, "selected_environment": safe_env(env),
                              "configured_command_suffix_matches": suffix_match,
                              "configured_environment_matches": all(env.get(k) == v for k, v in expected["environment"].items()),
                              "status": {k: status[k].strip() for k in ("Threads", "VmRSS", "VmLck", "Cpus_allowed_list", "Mems_allowed_list") if k in status},
                              "numa_mapping_policy_counts": policies})
        except (OSError, ProcessLookupError):
            continue
    hardware = {"lscpu": run(["lscpu", "-J"]), "numa": run(["numactl", "--hardware"]),
                "storage": run(["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS"]),
                "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id,memory.total,power.limit,ecc.mode.current", "--format=csv,noheader"], 8)}
    hardware["dmi"] = {name: (Path("/sys/class/dmi/id") / name).read_text().strip()
                       for name in ("board_vendor", "board_name", "board_version", "bios_vendor", "bios_version", "bios_date")
                       if (Path("/sys/class/dmi/id") / name).exists()}
    dmi = run(["dmidecode", "-t", "17"])
    fields = {"Size", "Locator", "Bank Locator", "Type", "Type Detail", "Speed", "Configured Memory Speed", "Manufacturer", "Part Number", "Rank", "Configured Voltage"}
    hardware["memory_devices"] = [{k.strip(): v.strip() for line in block.splitlines() if ":" in line
                                    for k, v in [line.split(":", 1)] if k.strip() in fields}
                                   for block in dmi.get("stdout", "").split("Memory Device\n")[1:]]
    if "error" in dmi or dmi.get("returncode") != 0:
        hardware["memory_devices_error"] = dmi.get("error", dmi.get("stderr"))
    hardware["memory_kib"] = {k: v.strip() for line in Path("/proc/meminfo").read_text().splitlines()
                              for k, v in [line.split(":", 1)] if k in ("MemTotal", "MemAvailable", "Mlocked", "SwapTotal", "SwapFree")}
    hardware["disk_bytes"] = dict(zip(("total", "used", "free"), shutil.disk_usage(LAB)))
    services = {u: run(["systemctl", "show", u, "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "FragmentPath"])
                for u in ("llm-deepseek41-api.service", "xeon-watch.service", "xeon-watch-notify.service")}
    record = {"captured_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "kernel": os.uname().release,
              "scope": "One-time read-only capture; no inference, restart, IPMI calls, or configuration changes.",
              "services": services, "processes": processes, "hardware": hardware,
              "user_reported_client": {"Open_WebUI": "v0.11.1", "comparison_model": "Qwen3.8 27B",
                                       "same_application_and_endpoint": "unconfirmed", "client_settings": "unconfirmed",
                                       "display_issue": "Attached external review reports grouped output; no stream timing recorded in this capture."}}
    (out / "CURRENT_STATE.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"processes": [{k: x[k] for k in ("pid", "configured_command_suffix_matches", "configured_environment_matches")} for x in processes], "out": str(out)}))


if __name__ == "__main__":
    main()
