#!/usr/bin/env python3
"""Run existing small CUDA checks with bounded thermal/error monitoring."""
import datetime
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
OUT = LAB / "results" / ROOT.name / "short-checks"
OUT.mkdir(exist_ok=False)
sys.path.insert(0, str(LAB / "board-study-20261024"))
from hardware_run import counters, ipmi


def sample():
    xml = subprocess.check_output(["nvidia-smi", "-i", "0000:85:00.0", "-q", "-x"], text=True, timeout=5)
    gpu = ET.fromstring(xml).find("gpu")
    def number(path):
        return float(gpu.findtext(path).split()[0])
    nic = Path("/sys/class/net/enp130s0f0")
    return {"monotonic": time.monotonic(), "gpu_C": number("temperature/gpu_temp"),
            "hbm_C": number("temperature/memory_temp"),
            "power_W": number("gpu_power_readings/instant_power_draw"),
            "limit_W": number("gpu_power_readings/current_power_limit"),
            "ecc_current": gpu.findtext("ecc_mode/current_ecc"),
            "gpu_ecc_single": number("ecc_errors/volatile/single_bit/total"),
            "gpu_ecc_double": number("ecc_errors/volatile/double_bit/total"),
            "pci_replays": number("pci/replay_counter"),
            "nic_up": (nic / "operstate").read_text().strip(),
            "nic_counters": {k: int((nic / "statistics" / k).read_text())
                             for k in ["rx_errors", "tx_errors", "rx_dropped", "tx_dropped"]},
            "cpu_counters": counters()}


record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "before_sensors": ipmi("sensor", "list"), "before_sel": ipmi("sel", "elist"),
          "samples": [], "command": [sys.executable, str(LAB / "v100-prep-20261024/arrival_checks.py")],
          "limits": {"gpu_C": 60, "hbm_C": 65, "seconds": 90}}
initial = sample()
record["samples"].append(initial)
owner = LAB.stat()
child = None
try:
    assert initial["limit_W"] <= 150.5 and initial["ecc_current"] == "Enabled"
    assert initial["gpu_C"] < 40 and initial["hbm_C"] < 40
    assert initial["nic_up"] == "up"
    assert initial["gpu_ecc_single"] == 0 and initial["gpu_ecc_double"] == 0
    assert "ECC" not in record["before_sel"]
    with (OUT / "stdout.txt").open("w") as stdout, (OUT / "stderr.txt").open("w") as stderr:
        child = subprocess.Popen(record["command"], stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                 user=owner.st_uid, group=owner.st_gid)
        started = time.monotonic()
        while child.poll() is None:
            row = sample()
            record["samples"].append(row)
            assert row["gpu_C"] < 60 and row["hbm_C"] < 65, "Thermal stop"
            assert row["limit_W"] <= 150.5 and row["ecc_current"] == "Enabled"
            for key in ["gpu_ecc_single", "gpu_ecc_double", "pci_replays", "nic_counters", "cpu_counters", "nic_up"]:
                assert row[key] == initial[key], ("Counter/link change", key)
            assert time.monotonic() - started < 90, "Short-test time limit"
            time.sleep(0.5)
        record["returncode"] = child.wait()
        assert record["returncode"] == 0, "CUDA test failed; inspect logs"
except BaseException as error:
    record["error"] = repr(error)
    raise
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    record["samples"].append(sample())
    record["after_sensors"] = ipmi("sensor", "list")
    record["after_sel"] = ipmi("sel", "elist")
    (OUT / "monitor.json").write_text(json.dumps(record, indent=2) + "\n")
    for path in [OUT, *OUT.iterdir()]:
        os.chown(path, owner.st_uid, owner.st_gid)
print((OUT / "stdout.txt").read_text(), end="")
print("Peak GPU/HBM temperatures:", max(s["gpu_C"] for s in record["samples"]),
      max(s["hbm_C"] for s in record["samples"]))
print("Peak sampled power:", max(s["power_W"] for s in record["samples"]), "W")
