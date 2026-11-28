"""Coordinate one native completion and its synchronous, bounded state exports on Linux."""

import concurrent.futures
import contextlib
import json
import os
import secrets
import selectors
import socket
import struct
import time

from native import Uncertain, proc_id


@contextlib.contextmanager
def boundaries(native, positions, captures, receive, *, diagnostic_logits=False):
    old = getattr(native, "boundary_request", None)
    native.boundary_request = (
        sorted(set(positions)),
        sorted(set(captures)),
        receive,
        diagnostic_logits,
    )
    try:
        yield
    finally:
        native.boundary_request = old


def exchange(native, request, operation):
    stops, captures, receive, diagnostic = request
    nonce = secrets.token_hex(16)
    plan = dict(stops=stops, captures=captures, channel=nonce, diagnostic_logits=diagnostic)
    allowed = set(stops)
    with contextlib.ExitStack() as stack:
        listener = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        listener.bind("\0xeon-prefix-" + nonce)
        listener.listen(1)
        done, writer = socket.socketpair()
        stack.enter_context(done)
        stack.enter_context(writer)
        sel = stack.enter_context(selectors.DefaultSelector())
        sel.register(listener, selectors.EVENT_READ)
        sel.register(done, selectors.EVENT_READ)
        pool = stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        future = pool.submit(operation, plan)
        future.add_done_callback(lambda _: writer.sendall(b"1"))
        while not future.done():
            for key, _ in sel.select():
                if key.fileobj is done:
                    return future.result()
                conn, _ = listener.accept()
                with conn:
                    pid, _, _ = struct.unpack(
                        "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                    )
                    if proc_id(pid) != native.epoch:
                        continue
                    conn.settimeout(120)
                    with conn.makefile("rb") as stream:
                        try:
                            line = stream.readline(4096)
                            if not line.endswith(b"\n"):
                                raise ValueError("incomplete native boundary")
                            event = json.loads(line)
                            pos = event["position"]
                            if pos not in allowed or (event["bytes"] > 0) != (pos in captures):
                                raise ValueError("unexpected native position or image")
                            allowed.remove(pos)
                            event["arrived_ns"] = time.monotonic_ns()
                            receive(event, conn, stream)
                        except Exception as exc:
                            # The HTTP operation can still be unwinding on the native server.
                            raise Uncertain(
                                "native_outcome_unknown", "prefix acknowledgement lost"
                            ) from exc
        return future.result()


def read_image(stream, path, size):
    """The native stream cannot grow this staging file beyond the reserved size."""
    remaining = size
    try:
        with path.open("xb") as out:
            while remaining:
                part = stream.read(min(remaining, 1024 * 1024))
                if not part:
                    raise EOFError("incomplete prefix image")
                remaining -= len(part)
                out.write(part)
            out.flush()
            os.fsync(out.fileno())
    except OSError as error:
        # Consume the already admitted native export so an ordinary disk failure
        # can be reported without interrupting the inference that produced it.
        while remaining:
            part = stream.read(min(remaining, 1024 * 1024))
            if not part:
                raise EOFError("native export interrupted") from error
            remaining -= len(part)
        raise
