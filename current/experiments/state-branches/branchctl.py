#!/usr/bin/env python3
"""Local CLI and single-writer service for one exclusive text inference slot."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import sys
import time
from manager import Manager

DEFAULT_INTERVAL = 1024
MODES = ("off", "fixed", "auto")
from native import Error, Native, Uncertain, encode
from storage import Store

MAX_FRAME = 16 * 1024**2


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        disconnected = False

        def send(v):
            nonlocal disconnected
            if not disconnected:
                try:
                    self.wfile.write((encode(v) + "\n").encode())
                    self.wfile.flush()
                except OSError:
                    disconnected = True  # finish the native operation before serving another client

        try:
            self.connection.settimeout(30)
            data = self.rfile.readline(MAX_FRAME + 1)
            if not data or len(data) > MAX_FRAME or not data.endswith(b"\n"):
                raise Error("request", "incomplete or oversized request")
            req = json.loads(data)
            if not isinstance(req, dict) or not isinstance(req.get("op"), str):
                raise Error("request", "operation must be a JSON object")

            def cancelled():
                if disconnected:
                    return True
                try:
                    return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
                except (BlockingIOError, TimeoutError):
                    return False
                except OSError:
                    return True

            def emit(t):
                send({"delta": t})
                if disconnected:
                    raise Uncertain(
                        "cancelled", "client left during generation; committed head retained"
                    )

            # select first: the handler's normal request timeout must not block a liveness check.
            import select

            emit.cancelled = lambda: disconnected or (
                bool(select.select([self.connection], [], [], 0)[0]) and cancelled()
            )
            answer = self.server.manager.handle(req, emit)
            send({"ok": True, "result": answer})
        except Error as e:
            send({"ok": False, "code": e.code, "error": str(e), "details": e.details})
        except (KeyError, TypeError, ValueError, OSError) as e:
            send({"ok": False, "code": "request", "error": str(e)})


def add_checkpoint_arguments(parser):
    parser.add_argument("--checkpoint-mode", choices=MODES, default="off")

    def interval(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError("checkpoint interval must be positive")
        return value

    parser.add_argument("--checkpoint-interval-tokens", type=interval, default=DEFAULT_INTERVAL)
    parser.add_argument("--resume-target-ms", type=float, default=5000)
    parser.add_argument(
        "--coverage-policy", choices=("all_history", "recent"), default="all_history"
    )
    parser.add_argument("--recent-window-tokens", type=interval, default=4096)
    parser.add_argument("--checkpoint-budget-mib", type=interval)


def serve(
    root,
    binding,
    budget,
    native_factory=Native,
    *,
    checkpoint_mode="off",
    checkpoint_interval=DEFAULT_INTERVAL,
    resume_target_ms=5000,
    coverage_policy="all_history",
    recent_window_tokens=4096,
    checkpoint_budget_mib=None,
):
    os.umask(0o077)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise Error("permissions", "the branch store must be private (mode 0700)")
    paths = json.loads(binding.read_bytes())
    backend = Path(paths["unix_socket"]).resolve()
    if (
        backend.parent.stat().st_mode & 0o077
        or Path(paths["image_dir"]).resolve() != root / "images"
    ):
        raise Error(
            "isolation", "use an exclusive private backend and this store's image directory"
        )
    sock = root / "manager.sock"
    with contextlib.ExitStack() as stack:
        for path in (backend.with_suffix(".writer.lock"), root / "manager.lock"):
            lock = stack.enter_context(path.open("a+b"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Error("in_use", "a manager already owns this backend or store") from exc
        n = native_factory(binding)
        s = Store(root, n, budget)
        stack.callback(s.db.close)
        if s.control()["mode"] != "uncertain":
            s.clean_files()
        manager = Manager(
            s,
            checkpoint_mode,
            checkpoint_interval,
            resume_target_ms=resume_target_ms,
            coverage_policy=coverage_policy,
            recent_window_tokens=recent_window_tokens,
        )
        if checkpoint_budget_mib is not None:
            if not manager.enabled:
                raise Error(
                    "budget", "the staging-inclusive checkpoint budget requires fixed or auto"
                )
            if s.control()["mode"] == "uncertain":
                raise Error(
                    "uncertain", "restart the private backend before changing its image budget"
                )
            requested = checkpoint_budget_mib * 1024**2
            if requested != s.control()["budget"]:
                with s.tx():
                    s.trim(requested, coverage=(manager.cp, s.branch()))
                    s.db.execute("UPDATE control SET budget=?,rev=rev+1", (requested,))
                s.clean_files()
        if s.control()["mode"] != "uncertain":
            # Even a clean manager restart re-establishes its own durable head.
            try:
                manager.handle({"op": "recover", "ticket": s.stamp()})
            except Error:
                if s.control()["mode"] not in ("invalid", "uncertain"):
                    raise
        if sock.is_socket():
            sock.unlink()

        class LocalServer(socketserver.UnixStreamServer):
            request_queue_size = 64

        server = stack.enter_context(LocalServer(str(sock), Handler))
        stack.callback(sock.unlink, missing_ok=True)
        server.manager = manager
        sock.chmod(0o600)
        stop = False

        def finish(*_):
            nonlocal stop
            stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, finish)
        server.timeout = 0.25
        while not stop:
            server.handle_request()


def rpc(root, req, emit=lambda _: None):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(900)
        sock.connect(str(root / "manager.sock"))
        sock.sendall((encode(req) + "\n").encode())
        with sock.makefile("rb") as reader:
            for line in reader:
                data = json.loads(line)
                if "delta" in data:
                    emit(data["delta"])
                elif data.get("ok"):
                    return data["result"]
                else:
                    raise Error(
                        data.get("code", "request"), data["error"], **data.get("details", {})
                    )
    raise Error(
        "manager_disconnected", "no commit receipt; inspect the current ticket before retrying"
    )


def checkpoint_text(v):
    def number(x, key, unit, factor=1):
        return (
            f"{x[key]/factor:,.2f} {unit} (p={x['p']}, c={x['c']})" if x else "未計測または部分計測"
        )

    c = v["capacity"]
    now = v["current"]
    future = v["after_plan_forecast"]
    lines = [
        f"枝 {v['branch']}  対象 {v['requested_range'][0]:,}–{v['requested_range'][1]:,} tokens ({v['config']['coverage']})",
        "現在の最大復帰時間の推定  " + number(now["estimated_max_resume"], "ms", "秒", 1000),
        f"ディスク画像 {c['retained_bytes']/2**20:.2f} MiB / {c['budget_bytes']/2**20:.2f} MiB（保護 {c['protected_bytes']/2**20:.2f} MiB）",
        "現在の最低換算速度  "
        + number(now["estimated_min_equivalent"], "tokens_per_second", "tokens/s"),
        f"現在の実在位置 {v['currently_available']}  時間目標 {now['target_met']}",
        f"計画間隔 {v['chosen_interval']}  次の候補 {v['next_position']}  計画後の時間目標 {future['forecast']['target_met']}",
        "計画後の最大復帰時間の推定  "
        + number(future["forecast"]["estimated_max_resume"], "ms", "秒", 1000),
        "計画後の画像保持 / ピーク  "
        + (
            "未計測"
            if future["retained_bytes"] is None
            else f"{future['retained_bytes']/2**20:.2f} / {future['peak_bytes']/2**20:.2f} MiB"
        ),
        "計画後の最低換算速度  "
        + number(future["forecast"]["estimated_min_equivalent"], "tokens_per_second", "tokens/s"),
    ]
    if v.get("return_route"):
        r = v["return_route"]
        lines.append(f"指定位置 {r['p']} に戻る起点 {r['c']}、再入力 {r['replay_tokens']} tokens")
    if v["reasons"]:
        lines.append("判定根拠  " + ", ".join(v["reasons"]))
    return "\n".join(lines)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--json", action="store_true", help="emit one machine-readable result")
    p.add_argument("--at", help="expected session/branch/generation/revision ticket as JSON")
    cmd = p.add_subparsers(dest="op", required=True)
    daemon = cmd.add_parser("serve")
    daemon.add_argument("--binding", type=Path, required=True)
    daemon.add_argument("--budget-mib", type=int, default=1024)
    add_checkpoint_arguments(daemon)
    for key in ("list", "status", "recover"):
        cmd.add_parser(key)
    cp = cmd.add_parser("checkpoint")
    cp.add_argument("view", choices=("preview", "status"))
    cp.add_argument("--branch", dest="name")
    cp.add_argument("--return-position", type=int)
    cp.add_argument("--empty-suffix", action="store_true")
    cp.add_argument("--history-tokens", dest="history", type=int)
    cp.add_argument("--mode", choices=MODES)
    cp.add_argument("--interval", type=int)
    cp.add_argument("--target-ms", type=float)
    cp.add_argument("--coverage", choices=("all_history", "recent"))
    cp.add_argument("--recent", type=int)
    t = cmd.add_parser("turn")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument(
        "--messages", type=Path, help="full normalized-input source as a JSON message array"
    )
    t.add_argument("--max-tokens", type=int, default=64)
    for key in ("mark", "fork", "checkout"):
        c = cmd.add_parser(key)
        c.add_argument("name")
        if key in ("mark", "fork"):
            c.add_argument("--pin", action="store_true")
        if key == "fork":
            c.add_argument("--from", dest="source", help="parent mark or immutable snapshot ID")
    c = cmd.add_parser("inspect")
    c.add_argument("kind", choices=["branch", "snapshot"])
    c.add_argument("name")
    for key in ("pin", "unpin", "release"):
        c = cmd.add_parser(key)
        c.add_argument("kind", choices=["branch", "mark"])
        c.add_argument("name")
    c = cmd.add_parser("budget")
    c.add_argument("bytes", type=int)
    a = p.parse_args()
    root = a.store.resolve()
    if a.op == "serve":
        serve(
            root,
            a.binding.resolve(),
            a.budget_mib * 1024**2,
            checkpoint_mode=a.checkpoint_mode,
            checkpoint_interval=a.checkpoint_interval_tokens,
            resume_target_ms=a.resume_target_ms,
            coverage_policy=a.coverage_policy,
            recent_window_tokens=a.recent_window_tokens,
            checkpoint_budget_mib=a.checkpoint_budget_mib,
        )
        return
    req = {
        k: v
        for k, v in vars(a).items()
        if k not in ("store", "json", "at", "source") and v is not None
    }
    if getattr(a, "source", None):
        req["from"] = a.source
    if getattr(a, "messages", None):
        req["messages"] = json.loads(a.messages.read_bytes())
    start = time.monotonic_ns()
    if a.op not in ("status", "list", "inspect", "checkpoint"):
        req["ticket"] = json.loads(a.at) if a.at else rpc(root, {"op": "ticket"})["ticket"]
    first = None

    def emit(s):
        nonlocal first
        if first is None:
            first = time.monotonic_ns()
        if not a.json:
            print(s, end="", flush=True)

    out = rpc(root, req, emit)
    if first is not None:
        out["client_first_ns"] = first
        out["client_first_ms"] = (first - start) / 1e6
    out["client_total_ms"] = (time.monotonic_ns() - start) / 1e6
    if a.op == "checkpoint" and not a.json:
        print(checkpoint_text(out))
    elif a.op == "turn" and not a.json:
        print()
        print(
            "processed",
            out["processed"],
            "sampled pending",
            out["pending_sampled_ids"],
            file=sys.stderr,
        )
        cp = out.get("checkpoint", {})
        skipped = [
            f"{e['position']} ({e.get('reason','unknown')})"
            for e in cp.get("events", [])
            if e["status"] == "skipped"
        ]
        if skipped:
            print("保存を見送った位置", ", ".join(skipped), file=sys.stderr)
        if cp.get("status") in ("failed", "unknown", "saved_cleanup_warning"):
            print(
                "Turn committed; checkpoint",
                cp["status"],
                cp.get("reason"),
                "slot",
                cp.get("slot_mode", "ready"),
                file=sys.stderr,
            )
    else:
        print(json.dumps(out, ensure_ascii=False, indent=None if a.json else 2))


if __name__ == "__main__":
    try:
        cli()
    except (Error, OSError, ValueError) as e:
        print(
            encode(
                {
                    "error": str(e),
                    "code": getattr(e, "code", "client"),
                    "details": getattr(e, "details", {}),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(2)
