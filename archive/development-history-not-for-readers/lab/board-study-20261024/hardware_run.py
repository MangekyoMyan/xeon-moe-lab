#!/usr/bin/env python3
"""Bounded hardware observations shared by the integrity and power-policy tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / "results" / ROOT.name
sys.path.insert(0, str(LAB))
from bench_deepseek_light import snapshot
from glm_q8_experiment import temperatures

ENV = os.environ.copy()
ENV.update({"LD_LIBRARY_PATH": str(ROOT / "diagnostics/usr/lib/x86_64-linux-gnu"), "LC_ALL": "C", "TZ": "UTC"})
IPMI = str(ROOT / "diagnostics/usr/bin/ipmitool")


def ipmi(*args):
    result = subprocess.run([IPMI, "-I", "open", *args], env=ENV, capture_output=True, text=True, timeout=40, check=True)
    return result.stdout


def counters():
    values = {}
    paths = list(Path("/sys/devices/system/edac/mc").glob("mc*/*count"))
    paths += list(Path("/sys/devices/system/cpu").glob("cpu*/thermal_throttle/*throttle_count"))
    for path in paths:
        values[str(path)] = int(path.read_text())
    return values


def energy():
    values = {}
    for path in Path("/sys/class/powercap").glob("intel-rapl:*"):
        name = (path / "name").read_text().strip()
        if name.startswith("package-") or name == "dram":
            values[path.name] = {"name": name, "uj": int((path / "energy_uj").read_text()),
                                 "range": int((path / "max_energy_range_uj").read_text())}
    return values


def sample(pid=None):
    row = {"monotonic": time.monotonic(), "unix_time": time.time(), "temperatures_C": temperatures(),
           "energy": energy(), "counters": counters(), "frequency_khz": {}}
    for path in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"):
        row["frequency_khz"][path.parent.name] = int(path.read_text())
    if pid is not None:
        row["process"] = snapshot(pid)
    return row


def run(label, command, timeout=900, env=None, keep_root=False):
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    initial = sample()
    report = {"command": command, "before": initial, "samples": [], "bmc": [], "start_unix": time.time()}
    child = None
    start = time.monotonic()
    next_bmc = 0
    next_progress = 0
    owner = LAB.stat()
    credentials = {} if keep_root else {"user": owner.st_uid, "group": owner.st_gid}
    try:
        if "ECC" in ipmi("sel", "elist"):
            raise RuntimeError("New ECC event already present; investigate before applying load")
        with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
            child = subprocess.Popen(command, env=env or ENV, stdout=stdout, stderr=stderr,
                stdin=subprocess.DEVNULL, start_new_session=True, **credentials)
            report["pid"] = child.pid
            while child.poll() is None:
                elapsed = time.monotonic() - start
                row = sample(child.pid)
                report["samples"].append(row)
                if row["temperatures_C"] and max(row["temperatures_C"].values()) >= 75:
                    raise RuntimeError("CPU temperature reached the test cutoff")
                if row["counters"] != initial["counters"]:
                    raise RuntimeError("ECC or thermal-throttle counters changed")
                if row.get("process") and row["process"]["available_kib"] < 8 * 1024 * 1024:
                    raise RuntimeError("Available memory below 8 GiB")
                if elapsed >= timeout:
                    raise RuntimeError("Test timeout")
                if elapsed >= next_bmc:
                    sensors = ipmi("sensor", "list")
                    events = ipmi("sel", "elist")
                    report["bmc"].append({"elapsed_s": elapsed, "host_unix": time.time(), "sensors": sensors, "events": events})
                    if "ECC" in events:
                        raise RuntimeError("New ECC event in BMC")
                    next_bmc = elapsed + 30
                if elapsed >= next_progress:
                    (folder / "run.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(label, round(elapsed), "s", "max_C", max(row["temperatures_C"].values()), flush=True)
                    next_progress = elapsed + 30
                time.sleep(1)
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            report["returncode"] = child.wait()
        report["after"] = sample()
        report["elapsed_s"] = time.monotonic() - start
        report["sel_after"] = ipmi("sel", "elist")
        (folder / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report.get("returncode") == 0, (folder / "stdout.txt").read_text()[-2000:]
    assert report["after"]["counters"] == initial["counters"] and "ECC" not in report["sel_after"]
    print("completed", label, flush=True)
    return report


def fix_owner():
    owner = LAB.stat()
    for path in [OUT, *OUT.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
