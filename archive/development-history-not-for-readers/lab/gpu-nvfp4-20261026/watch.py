#!/usr/bin/env python3
"""Bound GPU experiments with thermal, ECC, PCIe, and NIC observations."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / "results" / ROOT.name
GPU_ID = "GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3"
NIC_INTERFACES = ("enp4s0f0", "enp4s0f1")
sys.path.insert(0, str(LAB / "board-study-20261024"))
from hardware_run import counters, energy, ipmi
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment, temperatures
from bench_deepseek_light import snapshot


def gpu_sample():
    text = subprocess.check_output(["nvidia-smi", "-i", GPU_ID, "-q", "-x"], text=True, timeout=5)
    g = ET.fromstring(text).find("gpu")
    assert g.findtext("uuid") == GPU_ID
    def number(path):
        value = g.findtext(path)
        return None if value is None or value == "N/A" else float(value.split()[0])
    return {"bus_id": g.get("id"), "gpu_C": number("temperature/gpu_temp"), "hbm_C": number("temperature/memory_temp"),
            "power_W": number("gpu_power_readings/instant_power_draw"),
            "limit_W": number("gpu_power_readings/current_power_limit"),
            "ecc": g.findtext("ecc_mode/current_ecc"), "ecc_pending": g.findtext("ecc_mode/pending_ecc"),
            "sm_MHz": number("clocks/sm_clock"), "mem_MHz": number("clocks/mem_clock"),
            "gpu_util": number("utilization/gpu_util"), "memory_util": number("utilization/memory_util"),
            "pcie_rx_KB_s": number("pci/rx_util"), "pcie_tx_KB_s": number("pci/tx_util"),
            "vram_used_MiB": number("fb_memory_usage/used"),
            "ecc_single": number("ecc_errors/volatile/single_bit/total"),
            "ecc_double": number("ecc_errors/volatile/double_bit/total"),
            "replays": number("pci/replay_counter"),
            "throttling": {c.tag: c.text for c in g.findall("clocks_event_reasons/*")}}


def nic_sample():
    result = {}
    for name in NIC_INTERFACES:
        root = Path("/sys/class/net") / name
        result[name] = {"link": (root / "operstate").read_text().strip(),
                       **{k: int((root / "statistics" / k).read_text()) for k in ["rx_errors", "tx_errors", "rx_dropped", "tx_dropped"]}}
    return result


def fan_duty_sample():
    values = [int(ipmi("raw", "0x30", "0x90", "0x5a", "0x03", hex(reg), "0x01").strip(), 16) for reg in range(0x10, 0x14)]
    assert all(0 <= value <= 255 for value in values)
    return {"raw": values, "percent": [value * 100.0 / 255 for value in values], "mode": ipmi("raw", "0x30", "0x45", "0x00").strip()}

def bmc_sample():
    return {"unix": time.time(), "sensors": ipmi("sensor", "list"), "sel": ipmi("sel", "elist")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--overlap-preload", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments are stopped"
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command and "/" not in args.label and ".." not in args.label
    owner = LAB.stat()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chown(OUT, owner.st_uid, owner.st_gid)
    folder = OUT / args.label
    folder.mkdir(exist_ok=False)
    lock = None
    if not args.overlap_preload:
        lock = (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial_gpu, initial_nic, initial_counters = gpu_sample(), nic_sample(), counters()
    limits = {"gpu_C": 60, "hbm_C": 65,
              "cpu_C": 60, "peripheral_C": 45, "pch_C": 55}
    record = {"command": command, "started_unix": time.time(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "overlap_preload": args.overlap_preload, "samples": [], "phases": [], "bmc": [bmc_sample()],
              "limits": limits, "quiet_requested": args.quiet, "ambient_C_user_report": 20, "fan_PWM_limit_percent": 70,
              "quiet_fan_rpm_limits": {"FAN1": 3500, "FAN3": 3500, "FAN5": 3500, "FANB": 1125} if args.quiet else {},
              "fan_mode_before": ipmi("raw", "0x30", "0x45", "0x00").strip()}
    record["fan_duty_before"] = fan_duty_sample()
    assert max(record["fan_duty_before"]["raw"]) <= 178
    env = environment()
    for entry in args.env:
        key, value = entry.split("=", 1)
        assert key.startswith(("GGML_", "OMP_", "CUDA_"))
        env[key] = value
    record["extra_env"] = args.env
    child = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = None
    try:
        assert initial_gpu["limit_W"] <= 150.5 and initial_gpu["ecc"] == "Enabled"
        assert initial_gpu["gpu_C"] < 55 and initial_gpu["hbm_C"] < 55
        assert all(n["link"] == "up" for n in initial_nic.values())
        assert record["fan_mode_before"] == "00", "Keep Standard fan mode"
        assert initial_gpu["ecc_double"] == 0
        with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                     env=env, user=owner.st_uid, group=owner.st_gid, start_new_session=True)
            record["pid"] = child.pid
            start = time.monotonic()
            next_save, next_bmc, stderr_offset, next_duty = 0, 10, 0, 0
            while child.poll() is None:
                assert not (ROOT / "STOP_REQUESTED").exists(), "Stop requested"
                elapsed = time.monotonic() - start
                gpu, nic, cpu = gpu_sample(), nic_sample(), temperatures()
                row = {"elapsed_s": elapsed, "gpu": gpu, "nic": nic, "cpu_C": cpu,
                       "host_energy": energy(), "process": snapshot(child.pid)}
                if elapsed >= next_duty:
                    row["fan_duty"] = fan_duty_sample()
                    assert row["fan_duty"]["mode"] == "00", "Do not use Full fan mode"
                    assert max(row["fan_duty"]["raw"]) <= 178, "Fan PWM exceeded70percent; stop load"
                    next_duty = elapsed + 2
                record["samples"].append(row)
                with (folder / "stderr.txt").open() as log:
                    log.seek(stderr_offset)
                    for line in log:
                        if "llama-bench: benchmark" in line:
                            record["phases"].append({"elapsed_s": elapsed, "message": line.strip()})
                    stderr_offset = log.tell()
                assert gpu["gpu_C"] < limits["gpu_C"] and gpu["hbm_C"] < limits["hbm_C"], "GPU/HBM thermal cutoff"
                assert not cpu or max(cpu.values()) < limits["cpu_C"], "CPU thermal cutoff"
                assert gpu["limit_W"] <= 150.5 and gpu["ecc"] == "Enabled", "Operating policy changed"
                for key in ["ecc_single", "ecc_double", "replays"]:
                    assert gpu[key] == initial_gpu[key], ("GPU error counter changed", key)
                for name in NIC_INTERFACES:
                    for key in ["link", "rx_errors", "tx_errors"]:
                        assert nic[name][key] == initial_nic[name][key], "NIC link/error changed"
                # RX drops are recorded; background downloading can alter them without a PCIe fault.
                assert counters() == initial_counters, "Host ECC/throttle counter changed"
                assert elapsed < args.timeout, "Time limit"
                if row["process"]:
                    assert row["process"]["available_kib"] >= 3 * 1024 * 1024, "Low available host memory"
                if pending is not None and pending.done():
                    bmc = pending.result(); pending = None; record["bmc"].append(bmc)
                    assert "ECC" not in bmc["sel"], "BMC ECC event"
                    for line in bmc["sensors"].splitlines():
                        fields = [s.strip() for s in line.split("|")]
                        if fields[0] == "Peripheral Temp": assert float(fields[1]) < 45, "Peripheral thermal cutoff"
                        if fields[0] == "PCH Temp": assert float(fields[1]) < 55, "PCH thermal cutoff"
                        if fields[0] in record["quiet_fan_rpm_limits"] and fields[1] != "na":
                            assert float(fields[1]) <= record["quiet_fan_rpm_limits"][fields[0]], "Fan speed reached quiet-session cutoff"
                if elapsed >= next_bmc and pending is None:
                    pending = pool.submit(bmc_sample); next_bmc = elapsed + 10
                if elapsed >= next_save:
                    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
                    print(args.label, round(elapsed), "s", "GPU/HBM", gpu["gpu_C"], gpu["hbm_C"], "W", gpu["power_W"], flush=True)
                    next_save = elapsed + 30
                time.sleep(1)
            record["returncode"] = child.wait()
            assert record["returncode"] == 0, "Workload failed; inspect stderr"
    except BaseException as error:
        record["error"] = repr(error)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        # Preserve PCIe evidence before a user reboot clears the status bits.
        record["fault_pci"] = {}
        device_paths = [(Path("/sys/class/net") / name / "device").resolve() for name in NIC_INTERFACES]
        domain, bus, function = initial_gpu["bus_id"].split(":")
        device_paths.append((Path("/sys/bus/pci/devices") / f"{int(domain, 16):04x}:{bus}:{function}").resolve())
        bdfs = {p.name for d in device_paths for p in [d, d.parent] if (p / "vendor").exists()}
        for bdf in sorted(bdfs):
            try:
                record["fault_pci"][bdf] = subprocess.check_output(["lspci", "-Dvvnn", "-s", bdf], text=True, timeout=5)
            except Exception as probe_error:
                record["fault_pci"][bdf] = repr(probe_error)
        (ROOT / "STOP_REQUESTED").write_text(str(error) + "\n")
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        if child is not None: record["returncode"] = child.returncode
        pool.shutdown(wait=True)
        try: record["gpu_after"] = gpu_sample(); record["nic_after"] = nic_sample(); record["bmc"].append(bmc_sample())
        except Exception as error: record["final_probe_error"] = repr(error)
        record["elapsed_s"] = time.time() - record["started_unix"]
        try: record["fan_duty_after"] = fan_duty_sample()
        except Exception as error: record["fan_final_probe_error"] = repr(error)
        (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        for path in [folder, *folder.iterdir()]: os.chown(path, owner.st_uid, owner.st_gid)
        if lock is not None: lock.close()
    print("Completed", args.label, flush=True)


if __name__ == "__main__":
    main()
