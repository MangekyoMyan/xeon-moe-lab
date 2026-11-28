"""Guarded, opt-in launcher on x9ram. A private Unix socket never takes LAN requests."""

import argparse
import contextlib
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import time
from native import Connection, Native, digest, proc_id
from checkpoint_policy import DEFAULT_INTERVAL
from branchctl import add_checkpoint_arguments

HERE = Path(__file__).resolve().parent
LAB = Path("/home/codex/projects/llm-lab")
DEFAULT = LAB / "engine-state-phase1-20261128"
sys.path.insert(0, str(LAB / "normal24-timeline-20261124"))
import study
import prefill


def load(p):
    return json.loads(p.read_bytes())


def record(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def private(path, directory=True):
    if directory:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner = LAB.stat()
    os.chown(path, owner.st_uid, owner.st_gid)
    path.chmod(0o700 if directory else 0o600)


def protected_worker(run):
    if os.geteuid() != 0 or study.adopt.unit()["ActiveState"] != "inactive":
        raise RuntimeError("the ordinary backend must be stopped by its maintenance owner")
    if any(
        p.exists()
        for p in (
            run / "STOP_REQUESTED",
            study.E / "STOP_REQUESTED",
            LAB / "moe-engine-20261029/STOP_REQUESTED",
        )
    ):
        raise RuntimeError("unresolved hardware stop")
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    sys.path.insert(0, str(study.E))
    spec = importlib.util.spec_from_file_location("branch_guard", study.E / "supervise.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.ROOT = run
    original = guard.check_host

    def reserve(initial):
        cpu, available = original(initial)
        if available < 64 * 1024**2:
            raise RuntimeError("private pilot host reserve below 64 GiB")
        return cpu, available

    guard.check_host = reserve
    rc = guard.main()
    if rc != 0:
        raise RuntimeError("inference guard stopped: " + str(rc))


@contextlib.contextmanager
def engine(workspace, store, run, observed=False, stock=False):
    private(store)
    private(store / "images")
    private(run)
    sock = run / "native.sock"
    if len(os.fsencode(sock)) >= 107:
        raise RuntimeError("private socket path too long")
    key = run / "http.key"
    key.write_text(secrets.token_urlsafe(40) + "\n")
    private(key, False)
    cfg = prefill.cfg_for(128)
    cmd = cfg["command"]
    cmd[cmd.index("--host") + 1] = str(sock)
    cfg["host"] = str(sock)
    cmd[cmd.index("--api-key-file") + 1] = str(key)
    cfg["api_key_file"] = str(key)
    cmd += ["--slot-save-path", str(store / "images")]
    b = load(workspace / "build/BUILD.json")
    impl = (study.E / "build/bin" if stock else workspace / "build") / "libllama-server-impl.so"
    impl_hash = digest(impl)
    if impl_hash != b["base_server_impl_sha256" if stock else "library_sha256"]:
        raise RuntimeError("unexpected private or stock HTTP adapter")
    if not stock:
        cfg["environment"]["LD_LIBRARY_PATH"] = (
            str(impl.parent) + ":" + cfg["environment"]["LD_LIBRARY_PATH"]
        )
    if observed:
        tr = run / "trace"
        private(tr)
        (tr / "phase").write_text("idle\n")
        obs = LAB / "warm-suffix-20261127/observer/observe.so"
        if digest(obs) != load(obs.parent / "BUILD.json")["binary_sha256"]:
            raise RuntimeError("unexpected observer")
        cfg["environment"].update(LD_PRELOAD=str(obs), XEON_SUFFIX_RECORD=str(tr))
    record(run / "server-config.json", cfg)
    (run / "server-config.json").chmod(0o600)
    with (run / "supervisor.log").open("x") as log:
        proc = subprocess.Popen(
            [sys.executable, "-B", str(HERE / "x9ram_pilot.py"), "worker", "--run", str(run)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            end = time.monotonic() + 1800
            while proc.poll() is None and time.monotonic() < end:
                study.adopt.monitor()
                if (run / "health.json").is_file():
                    h = load(run / "health.json")
                    pid = h.get("pid")
                    if pid and h["status"] == "running" and sock.is_socket():
                        epoch = proc_id(pid)
                        ticks = prefill.child_identity(proc.pid, pid, cfg)
                        conn = Connection(sock, epoch)
                        try:
                            conn.request(
                                "GET",
                                "/health",
                                headers={"Authorization": "Bearer " + key.read_text().strip()},
                            )
                            res = conn.getresponse()
                            good = res.status == 200 and json.loads(res.read())["status"] == "ok"
                        except (OSError, http.client.HTTPException):
                            good = False
                        finally:
                            conn.close()
                        if good:
                            break
                time.sleep(2)
            else:
                raise RuntimeError(f"private startup failed; guard status {proc.poll()}")
            ref = workspace / "BEFORE.json"
            before = load(ref if ref.exists() else workspace / "OFFLINE_REFERENCE.json")
            profile = load(LAB / "current/runtime.json")
            expected = {**before["mapped_runtime_hashes"], str(impl): impl_hash}
            backing = prefill.verify_backing(pid, expected)
            env = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
            assert all((k + "=" + v).encode() in env for k, v in cfg["environment"].items())
            assert before["config_sha256"] == digest(study.E / "server-config.json") == study.NEW
            identity = {
                "model_revision": before["model_revision"],
                "model_manifest": before["model_download_manifest_sha256"],
                "profile": before["config_sha256"],
                "expert_plan": before["plan_sha256"],
                "server_exe": profile["server_executable"]["sha256"],
                "arithmetic_libraries": {
                    Path(k).name: v for k, v in before["mapped_runtime_hashes"].items()
                },
                "server_impl": impl_hash,
                "slot_format": "GGSQ4/server_tokens1/DSV4-full/text",
                "context": 131072,
                "vocab": 129280,
                "offload_thresholds": [128, 128],
                "threads": [24, 24],
                "template_options": {"thinking_mode": "chat"},
            }
            binding = {
                "epoch": epoch,
                "identity": identity,
                "unix_socket": str(sock),
                "private_key_file": str(key),
                "image_dir": str(store / "images"),
            }
            record(run / "binding.json", binding)
            private(run / "binding.json", False)
            n = Native(run / "binding.json")
            props = n.call("GET", "/props")
            identity["template_sha256"] = hashlib.sha256(
                str(props.get("chat_template", "")).encode()
            ).hexdigest()
            record(run / "binding.json", binding)
            record(
                run / "READY.json",
                {
                    "pid": pid,
                    "start_ticks": ticks,
                    "identity": identity,
                    "mapped": backing,
                    "observed": observed,
                    "original_or_private_server": ("stock" if stock else "private"),
                    "only_unix_socket": True,
                },
            )
            yield run / "binding.json"
        finally:
            if proc.poll() is None:
                proc.terminate()
            rc = proc.wait(timeout=130)
            h = load(run / "health.json") if (run / "health.json").exists() else {}
            record(
                run / "STOP.json",
                {
                    "returncode": rc,
                    "status": h.get("status"),
                    "child_returncode": h.get("child_returncode"),
                },
            )
            if rc != 0 or h.get("status") != "stopped_by_request" or h.get("child_returncode") != 0:
                raise RuntimeError(f"private guard did not close normally: {rc}, {h.get('status')}")


@contextlib.contextmanager
def manager(
    store,
    binding,
    log_path,
    test_module=None,
    *,
    checkpoint_mode="off",
    checkpoint_interval=DEFAULT_INTERVAL,
):
    from branchctl import rpc

    owner = LAB.stat()
    path = HERE / "branchctl.py" if test_module is None else test_module
    with log_path.open("x") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(path),
                "--store",
                str(store),
                "serve",
                "--binding",
                str(binding),
                "--checkpoint-mode",
                checkpoint_mode,
                "--checkpoint-interval-tokens",
                str(checkpoint_interval),
            ],
            user=owner.st_uid,
            group=owner.st_gid,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    status = rpc(store, {"op": "status"})
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"manager failed to start; see {log_path}")
            yield proc
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=650)


@contextlib.contextmanager
def window(workspace, label):
    study.R = workspace
    with (LAB / "normal24-timeline-20261124/.study.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        label = label + "-" + str(time.time_ns())
        state = study.adopt.unit()["ActiveState"]
        if state == "active":
            with study.maintenance(label):
                yield
        elif state == "inactive":
            study.adopt.monitor()
            study.adopt.released()
            if digest(study.E / "server-config.json") != study.NEW:
                raise RuntimeError("normal profile has changed")
            path = workspace / ("WINDOW_" + label + ".json")
            info = {
                "initial_normal_state": "inactive",
                "started_unix": time.time(),
                "restore_required": False,
            }
            record(path, info)
            try:
                yield
            finally:
                info["final_normal_state"] = study.adopt.unit()["ActiveState"]
                info["finished_unix"] = time.time()
                record(path, info)
        else:
            raise RuntimeError("ordinary API is transitioning or needs attention: " + state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["open", "worker"])
    p.add_argument("--workspace", type=Path, default=DEFAULT)
    p.add_argument("--run", type=Path)
    add_checkpoint_arguments(p)
    a = p.parse_args()
    if os.geteuid() != 0 or not __debug__:
        raise SystemExit("run this hardware launcher with sudo and normal Python assertions")
    if a.action == "worker":
        return protected_worker(a.run)
    workspace = a.workspace.resolve()
    store = workspace / "user-store"
    label = "interactive-" + str(int(time.time()))
    run = workspace / "runs" / label
    if not (workspace / "BEFORE.json").exists():
        if study.adopt.unit()["ActiveState"] == "active":
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(study.CHECK),
                    str(workspace / "BEFORE.json"),
                    "--expected",
                    str(study.EXPECTED),
                ],
                check=True,
            )
        elif study.adopt.unit()["ActiveState"] == "inactive":
            profile = load(LAB / "current/runtime.json")
            subprocess.run(
                [sys.executable, "-I", str(LAB / "current/check.py"), "verify"], check=True
            )
            record(
                workspace / "OFFLINE_REFERENCE.json",
                {
                    **load(study.EXPECTED),
                    "model_revision": profile["model"]["revision"],
                    "model_download_manifest_sha256": profile["model"]["download_receipt"][
                        "sha256"
                    ],
                    "plan_sha256": profile["plan"]["sha256"],
                },
            )
    run.parent.mkdir(exist_ok=True)

    def stop(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        with window(workspace, "BRANCH-CLI"):
            with engine(workspace, store, run) as binding, manager(
                store,
                binding,
                run / "manager.log",
                checkpoint_mode=a.checkpoint_mode,
                checkpoint_interval=a.checkpoint_interval_tokens,
            ) as worker:
                print(
                    f"Branch CLI ready. In a second terminal:\npython3 {HERE/'branchctl.py'} --store {store} list",
                    flush=True,
                )
                print(
                    "Ctrl+C closes this session and restores the prior service state.", flush=True
                )
                while worker.poll() is None:
                    if load(run / "health.json").get("status") != "running":
                        raise RuntimeError("private hardware guard stopped; inspect its receipt")
                    time.sleep(3)
                raise RuntimeError("private branch manager exited")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
