"""The text-only adapter for one exclusively owned native llama-server slot."""

import array
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
import uuid


class Error(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


class Uncertain(Error):
    """A submitted native operation may still be running."""


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pack(ids):
    a = array.array("i", ids)
    if a.itemsize != 4:
        raise RuntimeError("32-bit token storage required")
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def unpack(raw):
    a = array.array("i")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tolist()


def image_tokens(path, limit):
    # GGSQ v4, server_tokens v1, text only. The remaining numerical state is opaque.
    with path.open("rb") as f:
        header = f.read(24)
        if len(header) != 24:
            raise Error("invalid_image", "short native header")
        magic, version, words, marker, codec, n = struct.unpack("<IIIiII", header)
        if (magic, version, marker, codec) != (0x67677371, 4, -1, 1) or n > limit or words != n + 4:
            raise Error("invalid_image", "unexpected native slot/token format")
        ids = f.read(4 * n)
        if len(ids) != 4 * n or f.read(4) != b"\0" * 4 or not f.read(1):
            raise Error("invalid_image", "incomplete or non-text native slot")
    return ids


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def proc_id(pid):
    ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    return f"{Path('/proc/sys/kernel/random/boot_id').read_text().strip()}:{pid}:{ticks}"


class Connection(http.client.HTTPConnection):
    def __init__(self, path, epoch):
        super().__init__("localhost", timeout=600)
        self.path, self.epoch = path, epoch

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(str(self.path))
            pid, _, _ = struct.unpack(
                "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if proc_id(pid) != self.epoch:
                raise Error(
                    "backend_changed",
                    "private backend identity changed; reopen the manager with its new binding",
                )
            self.sock = sock
        except BaseException:
            sock.close()
            raise


class Native:
    def __init__(self, binding):
        b = json.loads(Path(binding).read_bytes())
        self.socket = Path(b["unix_socket"])
        self.images = Path(b["image_dir"])
        self.epoch = b["epoch"]
        self.identity = b["identity"]
        self.compat = hashlib.sha256(encode(self.identity).encode()).hexdigest()
        self.limit = self.identity["context"]
        self.vocab = self.identity["vocab"]
        self.key = Path(b["private_key_file"]).read_text().strip()
        self.images.mkdir(mode=0o700, exist_ok=True, parents=True)
        health = self.call("GET", "/health")
        if health.get("status") != "ok":
            raise Error("not_ready", "private inference server is loading")
        slots = self.call("GET", "/slots")
        if len(slots) != 1 or slots[0]["n_ctx"] != self.limit or slots[0].get("speculative"):
            raise Error(
                "wrong_runtime",
                "one text slot with the matching context and no speculation is required",
            )

    def call(self, method, route, body=None, consume=None):
        con = Connection(self.socket, self.epoch)
        issued = False
        try:
            con.connect()
            data = encode(body).encode() if body is not None else None
            issued = True
            con.request(
                method,
                route,
                data,
                {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            )
            resp = con.getresponse()
            if resp.status != 200:
                msg = resp.read(4096).decode(errors="replace")
                raise Error("native_error", f"{route} returned {resp.status}", native=msg)
            if consume is not None:
                return consume(resp)
            return json.loads(resp.read(32 * 1024**2))
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if issued and (route == "/completion" or route.startswith("/slots/0?action=")):
                raise Uncertain(
                    "native_outcome_unknown", f"no complete native receipt for {route}"
                ) from exc
            raise Error(
                "native_unavailable", f"native read or connection failed for {route}"
            ) from exc
        finally:
            con.close()

    def render(self, messages):
        text = self.call(
            "POST",
            "/apply-template",
            {
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"thinking_mode": "chat"},
            },
        )["prompt"]
        return self.call(
            "POST", "/tokenize", {"content": text, "add_special": True, "parse_special": True}
        )["tokens"]

    def erase(self):
        self.call("POST", "/slots/0?action=erase", {})

    def save(self, expected):
        name = ".writing-" + uuid.uuid4().hex + ".slot"
        path = self.images / name
        reply = self.call("POST", "/slots/0?action=save", {"filename": name})
        try:
            if (
                reply["n_saved"] * 4 != len(expected)
                or reply["n_written"] != path.stat().st_size
                or image_tokens(path, self.limit) != expected
            ):
                raise Error(
                    "boundary_mismatch",
                    "native saved tokens differ from the recorded processed boundary",
                )
            with path.open("rb") as f:
                os.fsync(f.fileno())
            return path, digest(path), path.stat().st_size
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def restore(self, sha, tokens, size):
        reply = self.call("POST", "/slots/0?action=restore", {"filename": sha + ".slot"})
        if reply.get("n_restored") * 4 != len(tokens) or reply.get("n_read") != size:
            raise Error(
                "restore_receipt", "native restore returned a different size or token count"
            )

    def replay(self, tokens, cache):
        if not tokens:
            self.erase()
            return 0
        ids = unpack(tokens)
        v = self.call(
            "POST",
            "/completion",
            {
                "id_slot": 0,
                "prompt": ids,
                "n_predict": 1,
                "cache_prompt": cache,
                "temperature": 0,
                "return_tokens": True,
                "response_fields": ["timings", "tokens_cached"],
            },
        )
        t = v["timings"]
        if v["tokens_cached"] != len(ids) or t["prompt_n"] + t["cache_n"] != len(ids):
            raise Error(
                "replay_boundary", "native replay did not reach the recorded token boundary"
            )
        return t["prompt_n"]

    def generate(self, ids, maximum, emit):
        start = time.monotonic_ns()

        def consume(resp):
            pieces = []
            first = None
            last = None
            partial_ids = 0
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                v = json.loads(payload)
                if "error" in v:
                    raise Error(
                        "generation_failed", "native stream returned an error", native=v["error"]
                    )
                if v.get("stop"):
                    last = v
                else:
                    partial_ids += len(v.get("tokens", []))
                    if v.get("content"):
                        if first is None:
                            first = time.monotonic_ns()
                        pieces.append(v["content"])
                        emit(v["content"])
            if last is None:
                raise Uncertain("native_outcome_unknown", "stream ended without a final receipt")
            samples = last.get("branch_sampled_ids")
            tm = last["timings"]
            if (
                not samples
                or len(samples) != tm["predicted_n"]
                or tm["prompt_n"] + tm["cache_n"] != len(ids)
                or last["tokens_cached"] != len(ids) + len(samples) - 1
            ):
                raise Error(
                    "generation_boundary",
                    "final sampled-token receipt or processed count does not match",
                )
            return {
                "content": "".join(pieces),
                "sampled": samples,
                "first_ns": first,
                "engine_start_ns": start,
                "partial_stream_token_count": partial_ids,
                "sampled_count": len(samples),
                "finish": last["stop_type"],
                "timings": tm,
            }

        fields = [
            "stop",
            "content",
            "tokens",
            "branch_sampled_ids",
            "tokens_cached",
            "stop_type",
            "timings",
        ]
        return self.call(
            "POST",
            "/completion",
            {
                "id_slot": 0,
                "prompt": ids,
                "n_predict": maximum,
                "temperature": 0,
                "seed": 19,
                "cache_prompt": True,
                "return_tokens": True,
                "stream": True,
                "response_fields": fields,
            },
            consume,
        )
