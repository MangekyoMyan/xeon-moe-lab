"""Record final service ownership, guard exits and counter comparisons without issuing inference."""
import json
import subprocess
import sys
from pathlib import Path

from study import CHECK, E, EXPECTED, NEW, R, adopt, fresh, read, sha


def main():
    windows = {}
    for key in ("TIMELINE", "OPERATOR", "VALIDATION", "CONTINUED", "PREFIX_QUALITY"):
        window = read(R / f"WINDOW_{key}.json")
        returned = read(R / f"RESTORED_{key}.json")
        assert window["restore_required"] is False and window["profile_sha256"] == NEW
        assert returned["pid"] == window["restored"]["pid"] and returned["API_health"] == "ok"
        windows[key] = {"from_pid": window["initial"]["pid"], "restored_pid": returned["pid"],
                        "started_unix": window["started_unix"], "finished_unix": window["finished_unix"]}

    assert sha(CHECK) == adopt.CHECKER
    dest = R / "FINAL_API.json"
    if not dest.exists():
        subprocess.run([sys.executable, "-I", str(CHECK), str(dest), "--expected", str(EXPECTED)], check=True, timeout=90)
    now = read(dest)
    who = adopt.identity(NEW)
    assert who["pid"] == now["pid"] == windows["PREFIX_QUALITY"]["restored_pid"]
    assert who["decode"] == who["prefill"] == 24 and who["decode_mask"] == who["prefill_mask"] == "0xffffff"
    config = read(E / "server-config.json")
    thresholds = ("GGML_OP_OFFLOAD_MIN_BATCH", "GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")
    assert all(config["environment"][name] == "64" for name in thresholds)
    current = adopt.snapshot_health()
    assert current["boot"] == who["boot"] and not current["alerts"] and not current["errors"]
    initial = read(R / "runs/control/health.json")
    host, gpu = initial["initial_host"], initial["initial_gpu"]
    assert current["host_counters"] == host and len(host) == 112
    assert current["gpu"]["power_limit_W"] == 150 and current["gpu"]["ecc_mode"] == "Enabled"
    ecc = ("ecc_single", "ecc_double")
    assert all(current["gpu"][key] == gpu[key] for key in ecc)
    checks = read(R / "CHECKS.json")["thermal_by_run"]
    receipts = {}
    for name, samples in checks.items():
        path = R / "runs" / name / "health.json"
        h = read(path)
        expected_exit = "stopped_by_request" if name.startswith("pp-") else "exited"
        assert h["boot_id"] == who["boot"] and h["status"] == expected_exit and h["child_returncode"] == 0
        assert h["initial_host"] == host and all(h["initial_gpu"][key] == gpu[key] for key in ecc)
        assert not h.get("error") and not samples["flagged_samples"]
        assert samples["min_available_GiB"] >= 64 and samples["fan_peak_pct"] <= 70
        receipts[name] = {"pid": h["pid"], "status": h["status"], "child_returncode": h["child_returncode"],
                          "started_unix": h["started_unix"], "finished_unix": h["finished_unix"],
                          "initial_host_counter_count": len(host), "initial_host_equal_to_control": True,
                          "initial_gpu_ecc": {k: h["initial_gpu"][k] for k in ecc}, "local_guard_sha256": sha(path)}
    assert len(receipts) == 13
    final_state = {
        "normal_pid": who["pid"], "normal_profile": NEW, "model_process": who,
        "production_threads_decode_prefill": [24, 24], "production_prefill_thresholds": [64, 64],
        "production_extra_layer7_bank": False, "model_maps_match_initial": now["mapped_runtime_hashes"] == read(R/"BEFORE.json")["mapped_runtime_hashes"],
        "provider_sha256": now["mapped_runtime_hashes"], "api_health": now["API_health"], "services": now["services"],
        "active_alerts": current["alerts"], "sensor_errors": current["errors"], "host_counters_compared": len(host),
        "host_counters_changed": 0, "gpu_ECC_delta": {k: current["gpu"][k] - gpu[k] for k in ecc},
        "windows": windows, "processes": receipts, "normal_profile_at_start_and_end": read(R/"BEFORE.json")["config_sha256"] == now["config_sha256"] == NEW,
        "ordinary_api_before_pid": read(R/"BEFORE.json")["pid"]
    }
    assert final_state["model_maps_match_initial"] and final_state["normal_profile_at_start_and_end"]
    fresh(R/"FINAL_STATE.json", final_state)
    print(json.dumps({"pid": who["pid"], "closed_windows": len(windows), "clean_child_exits": len(receipts),
                      "host_counters_compared": len(host), "api": now["API_health"]}))


if __name__ == "__main__":
    main()
