#!/usr/bin/env python3
"""Synthetic operations review for xeon-moe-lab public commit 4ff68eba.

No model, GPU, IPMI, service, notification, or real child process is started.
Only selected AST definitions from the reviewed files are evaluated. All command
execution and signals are replaced by fakes, and writes go to temporary folders.
Run: python operations_probes.py --repo /path/to/xeon-moe-lab --output new-result.json
The output must be a new file; existing paths are rejected before the probes run.
The tested source hashes are checked before execution to avoid silently testing
different source or importing arbitrary archived startup code.
"""
import argparse
import ast
import copy
import errno
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import tempfile
import types

COMMIT = "4ff68eba2d8c480692511dc6d219d27a6a102866"
LAB = "archive/development-history-not-for-readers/lab/"
PATHS = {
    "supervisor": LAB + "deepseek41-bringup-20261101/supervise.py",
    "retry_supervisor": LAB + "guard-retry-20261104/supervise.py",
    "old_supervisor": LAB + "api-server-20261030/supervise.py",
    "guard": LAB + "deepseek41-bringup-20261101/gpu_guard.py",
    "watch": LAB + "moe-engine-20261029/watch.py",
    "ipmi": LAB + "board-study-20261024/hardware_run.py",
    "monitor": LAB + "thermal-monitor-20261101/monitor.py",
    "unit": "deployment/llm-deepseek41-api.service.template",
}
# The public supervisor is also the applied guard-retry version, not the older
# api-server-20261030 supervisor.
PINNED_SUPERVISOR = "ae8f58d2cf9e70847ce219bb35fd8ac77544fdc0e435987b31316c11b904c5ed"
PINNED_HASHES = {
    "supervisor": PINNED_SUPERVISOR,
    "retry_supervisor": PINNED_SUPERVISOR,
    "old_supervisor": "117a6186f5c79dd9910e29f496b38b4f55a02a6508fcd7df98faf6f2187773e8",
    "guard": "b9a4f123b78c125e155a8cc2710c67436d6d05ef742559240519a40309492c2d",
    "watch": "b68bd1801403ef58fc21a3d0fc42fdff43947c953c78d5d9e99c4508b88fe3db",
    "ipmi": "47b0f10db706f4f0dc326194ce04aa5f8faea99cbd279e820b99882e97212e39",
    "monitor": "39df2f0ed0a5a5f8258d4599530650bde022dc0306610012794bb6559b4ed358",
    "unit": "bf69d30985bfa8e79e835b1fff9a58112418face8d1bee097539d3b59b94a3e6",
}


def definitions(path, names, namespace=None):
    """Load only named definitions; do not run imports or module startup code."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    assert not any(node.decorator_list for node in selected)
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


class Clock:
    def __init__(self):
        self.t = 0.0

    def mono(self):
        return self.t

    def wall(self):
        return 1000000.0 + self.t

    def advance(self, seconds):
        self.t += seconds


def healthy_gpu():
    return dict(gpu_C=28.0, hbm_C=30.0, limit_W=150.0, ecc="Enabled",
                ecc_single=0, ecc_double=0, replays=0)


HEALTHY_BMC = {"sel": "", "sensors":
               "Peripheral Temp | 30 | degrees C | ok\nPCH Temp | 40 | degrees C | ok\n"}


def load_guard(repo):
    namespace = {
        "json": json, "math": math, "Path": Path, "subprocess": subprocess,
        "time": types.SimpleNamespace(monotonic=lambda: 0, time=lambda: 0, sleep=lambda _: None),
        "ATTEMPTS": 3, "QUERY_SECONDS": 5, "RECOVERY_SECONDS": 20,
        "SNAPSHOT_SECONDS": 20, "EXPECTED_FANS": {"FAN1", "FAN3", "FAN5", "FANB"},
        "WATCH": Path("/synthetic-no-read-allowed"),
    }
    return definitions(repo / PATHS["guard"],
                       ["ProbeCancelled", "number", "temperatures", "validate_gpu",
                        "read_snapshot", "validate_snapshot", "GpuProbe"], namespace)


def run_supervisor(repo, mode):
    """Execute the actual main() control flow with exclusively fake hardware."""
    clock = Clock()
    gpu_times = []
    ipmi_calls = []
    signals = []
    shutdowns = []
    snapshot_reads = []
    stop_at = []
    sleep_count = 0
    child = types.SimpleNamespace(pid=98765, returncode=None)
    child.poll = lambda: child.returncode
    child.wait = lambda timeout=None: child.returncode
    guard = load_guard(repo)

    def gpu():
        gpu_times.append(clock.t)
        return healthy_gpu()

    def snapshot_forbidden():
        snapshot_reads.append(clock.t)
        raise AssertionError("A healthy direct GPU query must not need a snapshot")

    class Future:
        def done(self):
            return True

        def result(self):
            return copy.deepcopy(HEALTHY_BMC)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn):
            return Future()

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    def killpg(pid, sig):
        signals.append({"pid": pid, "signal": int(sig), "time": clock.t})
        if mode == "exit_signal_race":
            child.returncode = 0
            raise ProcessLookupError(errno.ESRCH, "Synthetic process group exited between poll and signal")
        child.returncode = -int(sig)

    def sleep(seconds):
        nonlocal sleep_count
        clock.advance(seconds)
        sleep_count += 1
        if mode in ("exit_signal_race", "normal_stop"):
            scope["stop_requested"] = True
            stop_at.append(clock.t)
        elif mode == "slow_fan" and sleep_count == 2:
            child.returncode = 0

    def fake_run(command, **kwargs):
        assert command[0] == "synthetic-ipmitool"
        # Preserve the real helper's timeout contract: successful 39 s is below 40 s.
        assert kwargs["timeout"] == 40 and kwargs["check"] is True
        duration = 39.0 if mode in ("slow_fan", "stop_during_fan") else 0.0
        clock.advance(duration)
        ipmi_calls.append({"args": command[3:], "timeout": kwargs["timeout"],
                           "seconds": duration, "end": clock.t})
        # First five calls are startup; the sixth is the first runtime fan read.
        if mode == "stop_during_fan" and len(ipmi_calls) == 6:
            scope["stop_requested"] = True
            stop_at.append(clock.t)
        return types.SimpleNamespace(stdout="00\n" if command[-1] == "0x00" else "66\n")

    ipmi_scope = definitions(repo / PATHS["ipmi"], ["ipmi"], {
        "IPMI": "synthetic-ipmitool", "ENV": {},
        "subprocess": types.SimpleNamespace(run=fake_run),
    })
    fan = definitions(repo / PATHS["watch"], ["fan_duty_sample"],
                      {"ipmi": ipmi_scope["ipmi"]})["fan_duty_sample"]
    watch = types.SimpleNamespace(gpu_sample=gpu, counters=lambda: {"ECC": 0},
                                  fan_duty_sample=fan, bmc_sample=lambda: copy.deepcopy(HEALTHY_BMC),
                                  environment=lambda: {}, nic_sample=lambda: {})
    with tempfile.TemporaryDirectory(prefix="xeon-ops-review-") as temporary:
        lab = Path(temporary)
        root = lab / "engine"
        root.mkdir()
        lock = lab / "results/glm-air-q8-20261020/.experiment.lock"
        lock.parent.mkdir(parents=True)
        lock.touch()
        (root / "server-config.json").write_text(json.dumps({
            "command": ["this-command-is-never-executed"], "environment": {}}))
        scope = definitions(repo / PATHS["supervisor"],
                            ["request_stop", "write_json", "check_bmc", "check_host", "main"], {
            "ROOT": root, "LAB": lab, "Path": Path, "watch": watch,
            "stop_requested": False, "json": json, "fcntl": fcntl,
            "os": types.SimpleNamespace(geteuid=lambda: 0, killpg=killpg),
            "signal": types.SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT,
                                             SIGKILL=signal.SIGKILL, signal=lambda *args: None),
            "subprocess": types.SimpleNamespace(check_output=lambda *a, **k: "",
                                                 Popen=lambda *a, **k: child,
                                                 DEVNULL=subprocess.DEVNULL,
                                                 TimeoutExpired=subprocess.TimeoutExpired),
            "concurrent": types.SimpleNamespace(futures=types.SimpleNamespace(ThreadPoolExecutor=Pool)),
            "time": types.SimpleNamespace(monotonic=clock.mono, time=clock.wall, sleep=sleep),
            "sys": types.SimpleNamespace(stderr=io.StringIO()),
            "ProbeCancelled": guard["ProbeCancelled"], "temperatures": guard["temperatures"],
        })
        scope["check_host"] = lambda initial: ({"CPU": 28.0}, 64 * 1024**2)
        scope["GpuProbe"] = lambda *args, **kwargs: guard["GpuProbe"](
            *args, **kwargs, monotonic=clock.mono, wall=clock.wall,
            sleep=clock.advance, snapshot=snapshot_forbidden)
        code = None
        error = None
        try:
            code = scope["main"]()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        health = json.loads((root / "health.json").read_text())
        return {
            "mode": mode, "exit_code": code, "raised": error,
            "reachability_note": ("ESRCH is explicitly injected. Natural reachability in this supervisor is not established; ordinary unreaped Linux child exit preserves the process group. This is not a confirmed production race."
                                  if mode == "exit_signal_race" else None),
            "gpu_sample_times_seconds": gpu_times,
            "runtime_gpu_gap_seconds": [b - a for a, b in zip(gpu_times[1:], gpu_times[2:])],
            "ipmi_calls": ipmi_calls,
            "signals": signals, "requested_stop_times": stop_at,
            "stop_to_sigterm_seconds": signals[0]["time"] - stop_at[0] if signals and stop_at else None,
            "pool_shutdown_calls": shutdowns, "snapshot_reads": snapshot_reads,
            "health_status": health["status"], "health_has_finished_unix": "finished_unix" in health,
            "stop_marker_exists": (root / "STOP_REQUESTED").exists(),
        }


def bmc_cases(repo):
    check = definitions(repo / PATHS["supervisor"], ["check_bmc"])["check_bmc"]
    cases = {
        "healthy": HEALTHY_BMC,
        "empty_sensor_output": {"sel": "", "sensors": ""},
        "both_unavailable": {"sel": "", "sensors": "Peripheral Temp | na | degrees C | na\nPCH Temp | na | degrees C | na"},
        "both_nonfinite": {"sel": "", "sensors": "Peripheral Temp | nan | degrees C | ok\nPCH Temp | NaN | degrees C | ok"},
        "missing_pch": {"sel": "", "sensors": "Peripheral Temp | 30 | degrees C | ok"},
        "bad_status_with_valid_values": {"sel": "", "sensors": "Peripheral Temp | 30 | degrees C | cr\nPCH Temp | 40 | degrees C | ok"},
        "peripheral_at_limit": {"sel": "", "sensors": "Peripheral Temp | 45 | degrees C | ok\nPCH Temp | 40 | degrees C | ok"},
        "pch_at_limit": {"sel": "", "sensors": "Peripheral Temp | 30 | degrees C | ok\nPCH Temp | 55 | degrees C | ok"},
        "ecc_in_sel": {"sel": "ECC fault", "sensors": HEALTHY_BMC["sensors"]},
    }
    results = {}
    for name, value in cases.items():
        try:
            check(value)
            results[name] = {"accepted": True}
        except Exception as exc:
            results[name] = {"accepted": False, "error": str(exc)}
    return results


def detector_config():
    # Explicit fixture; it is not claimed to be the user's current config.json.
    return {"limits": {"cpu_C": 60, "gpu_C": 60, "hbm_C": 65, "dimm_C": 70,
                       "pch_C": 55, "peripheral_C": 45, "fan_percent": 70},
            "recovery_margin": 3, "recovery_samples": 3, "sensor_failure_samples": 3,
            "expected_dimm_count": 16, "repeat_cooldown_seconds": 1800}


def detector_sample(now=100.0):
    return {"unix": now, "boot_id": "synthetic-boot", "errors": {}, "cpu": {"CPU": 28.0},
            "gpu": {"gpu_C": 28.0, "hbm_C": 30.0, "ecc_mode": "Enabled",
                    "ecc_single": 0, "ecc_double": 0, "replays": 0},
            "host_counters": {"synthetic_edac": 0},
            "bmc": {"observed_unix": now, "dimms_C": {f"DIMM{i}": 30 for i in range(16)},
                    "fan_rpm": {k: 1000 for k in ["FAN1", "FAN3", "FAN5", "FANB"]},
                    "fan_percent": [40] * 4, "fan_mode": "00", "pch_C": 40,
                    "peripheral_C": 30, "bad_sensor_status": {}}}


def detector_cases(repo):
    Detector = definitions(repo / PATHS["monitor"], ["Detector"])["Detector"]
    results = {}
    for mode in ["healthy", "bad_bmc_sensor_status", "missing_gpu_counters", "missing_board_temperature"]:
        detector = Detector(detector_config())
        history = []
        for i in range(4):
            sample = detector_sample(100 + i * 10)
            if mode == "bad_bmc_sensor_status":
                sample["bmc"]["bad_sensor_status"] = {"12V": "cr"}
            if mode == "missing_gpu_counters":
                sample["gpu"].update(ecc_single=None, ecc_double=None, replays=None)
            if mode == "missing_board_temperature":
                sample["bmc"]["pch_C"] = None
            events, recovered = detector.update(sample)
            history.append({"unix": sample["unix"], "events": events, "recovered": recovered})
        results[mode] = {"history": history, "active_alerts": sorted(
            key for key, value in detector.s["alarms"].items() if value["active"]),
            "failure_counts": detector.s["failures"]}
    detector = Detector(detector_config())
    history = []
    for now, count in [(100, 0), (110, 1), (120, 101), (1911, 101)]:
        sample = detector_sample(now)
        sample["gpu"]["ecc_single"] = count
        events, recovered = detector.update(sample)
        history.append({"unix": now, "counter": count, "events": events})
    results["counter_cooldown_policy"] = {
        "history": history, "last_seen": detector.s["last_counter"]["gpu:ecc_single"],
        "last_notified_at": detector.s["counter_notified"].get("gpu:ecc_single")}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New result JSON path; existing paths are rejected before execution")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"Output already exists; specify a new result file: {args.output}")
    repo = args.repo.resolve()
    hashes = {name: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
              for name, relative in PATHS.items()}
    if hashes != PINNED_HASHES:
        raise SystemExit("Unexpected source hash; review the differences before running")
    supervisor = {mode: run_supervisor(repo, mode) for mode in [
        "slow_fan", "stop_during_fan", "exit_signal_race", "normal_stop"]}
    bmc = bmc_cases(repo)
    monitor = detector_cases(repo)
    checks = {
        "slow_ipmi_creates_197_second_runtime_gpu_gap": supervisor["slow_fan"]["runtime_gpu_gap_seconds"] == [197.0],
        "slow_ipmi_delays_synthetic_stop_158_seconds": supervisor["stop_during_fan"]["stop_to_sigterm_seconds"] == 158.0,
        "gpu_snapshot_not_consulted_when_direct_queries_healthy": all(not x["snapshot_reads"] for x in supervisor.values()),
        "empty_na_nan_and_missing_board_sensors_pass": all(bmc[k]["accepted"] for k in [
            "empty_sensor_output", "both_unavailable", "both_nonfinite", "missing_pch"]),
        "real_threshold_and_ecc_controls_reject": all(not bmc[k]["accepted"] for k in [
            "peripheral_at_limit", "pch_at_limit", "ecc_in_sel"]),
        "injected_esrch_raises_not_a_proven_natural_race": supervisor["exit_signal_race"]["raised"]["type"] == "ProcessLookupError",
        "injected_esrch_leaves_running_health_and_skips_finalization": (
            supervisor["exit_signal_race"]["health_status"] == "running"
            and not supervisor["exit_signal_race"]["health_has_finished_unix"]
            and not supervisor["exit_signal_race"]["pool_shutdown_calls"]),
        "normal_stop_control_finalizes": (supervisor["normal_stop"]["exit_code"] == 0
            and supervisor["normal_stop"]["health_status"] == "stopped_by_request"
            and supervisor["normal_stop"]["health_has_finished_unix"]),
        "monitor_does_not_alert_on_bad_bmc_status": not monitor["bad_bmc_sensor_status"]["active_alerts"],
        "monitor_does_not_alert_on_missing_gpu_counters": not monitor["missing_gpu_counters"]["active_alerts"],
        "monitor_control_detects_missing_board_temperature": monitor["missing_board_temperature"]["active_alerts"] == ["sensor:board_temperature"],
        "counter_increments_during_cooldown_not_deferred": (
            len(monitor["counter_cooldown_policy"]["history"][1]["events"]) == 1
            and not monitor["counter_cooldown_policy"]["history"][2]["events"]
            and not monitor["counter_cooldown_policy"]["history"][3]["events"]),
    }
    result = {"reviewed_commit": COMMIT, "scope": "Public source; virtual time and synthetic hardware only",
              "source_files": {k: {"path": PATHS[k], "sha256": v} for k, v in hashes.items()},
              "supervisor_cases": supervisor, "bmc_cases": bmc,
              "monitor_cases": monitor, "assertions": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "limits": ["No real latency, temperature, GPU, service, or inference measurement",
                         "Fake immediate BMC future; does not exercise real thread shutdown timing",
                         "Monitor config is explicit fixture, not current private machine config",
                         "ESRCH fault injection does not establish natural reachability; excluded from confirmed findings",
                         "Counter cooldown suppression may be intentional; treated as a policy question",
                         "Source files remain unchanged"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"], "output": str(args.output)}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
