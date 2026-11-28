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
from checkpoint_policy import MODES, DEFAULT_INTERVAL
from native import Error, Native, encode
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
            answer = self.server.manager.handle(req, lambda t: send({"delta": t}))
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


def serve(
    root,
    binding,
    budget,
    native_factory=Native,
    *,
    checkpoint_mode="off",
    checkpoint_interval=DEFAULT_INTERVAL
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
        manager = Manager(s, checkpoint_mode, checkpoint_interval)
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
        options = (
            {}
            if (a.checkpoint_mode, a.checkpoint_interval_tokens) == ("off", DEFAULT_INTERVAL)
            else {
                "checkpoint_mode": a.checkpoint_mode,
                "checkpoint_interval": a.checkpoint_interval_tokens,
            }
        )
        serve(root, a.binding.resolve(), a.budget_mib * 1024**2, **options)
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
    if a.op not in ("status", "list", "inspect"):
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
    if a.op == "turn" and not a.json:
        print()
        print(
            "processed",
            out["processed"],
            "sampled pending",
            out["pending_sampled_ids"],
            file=sys.stderr,
        )
        cp = out.get("checkpoint", {})
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
