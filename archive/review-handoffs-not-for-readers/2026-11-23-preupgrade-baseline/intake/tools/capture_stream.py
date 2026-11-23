#!/usr/bin/env python3
"""Explicit, single-request OpenAI-compatible SSE observation (stdlib only).

Importing this module never opens a connection. The outgoing Authorization header
is not logged; supplied JSON and response bodies are saved verbatim.
See README_capture_stream.md.
"""

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit


SCHEMA = "capture-stream/v1"
TEXT_FIELDS = ("content", "reasoning_content", "reasoning")
CAVEATS = [
    "Application read1 returns, TCP reads/packets, SSE events, and model tokens are different units.",
    "Timestamps are client perf_counter_ns immediately after read1 returns, not model generation times.",
    "All events completed in one read share its timestamp; zero event gaps are retained.",
    "TTFT means time to the first nonempty content delta received; event gaps are not token TPOT.",
    "usage/completion_tokens are server reports; event counts are never converted into token counts.",
    "Use the same client, payload, and observation point before/after; direct and proxy routes differ.",
    "This client clock is separate from internal engine/token-ID benchmark clocks.",
    "HTTP framing is decoded, but raw body bytes are preserved; these are not packet captures.",
    "Parsing and file writes can affect later read times; this client does not measure browser rendering.",
    "The outgoing Authorization header is not logged; supplied JSON and response bodies are saved verbatim.",
]


class InputError(ValueError):
    pass


class CaptureStop(Exception):
    pass


def json_loads(text):
    def reject_constant(_):
        raise ValueError("Non-finite JSON number")

    return json.loads(text, parse_constant=reject_constant)


def dump_json(path, value):
    with path.open("w", encoding="utf-8", errors="backslashreplace") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")


def gap_stats(times):
    gaps = sorted((b - a) / 1e6 for a, b in zip(times, times[1:]))
    return {
        "gap_count": len(gaps),
        "median_ms": statistics.median(gaps) if gaps else None,
        "p95_ms": gaps[math.ceil(0.95 * len(gaps)) - 1] if gaps else None,
        "max_ms": max(gaps) if gaps else None,
        "zero_gap_count": gaps.count(0),
        "p95_method": "nearest rank, ceil(0.95*N)",
    }


class SSEDecoder:
    """Frame UTF-8 SSE across arbitrary byte reads; do not dispatch an EOF tail.

    Every blank-line block is recorded. has_data distinguishes actual dispatched
    SSE data events (including data: with an empty value) from comments/blank blocks.
    CR terminates a line immediately; a following LF is consumed, even across reads.
    """

    def __init__(self, emit):
        self.emit = emit
        self.line = bytearray()
        self.lines = []
        self.after_cr = False
        self.first_line = True
        self.offset = 0
        self.last_id = ""
        self.invalid_utf8_lines = 0

    def feed(self, data, received_ns, read_index):
        for byte in data:
            self.offset += 1
            if self.after_cr:
                self.after_cr = False
                if byte == 10:
                    continue
            if byte in (10, 13):
                self.after_cr = byte == 13
                raw = bytes(self.line)
                self.line.clear()
                if self.first_line:
                    raw = raw.removeprefix(b"\xef\xbb\xbf")
                    self.first_line = False
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    self.invalid_utf8_lines += 1
                    line = raw.decode("utf-8", errors="replace")
                if line:
                    self.lines.append(line)
                else:
                    self._emit_block(received_ns, read_index)
            else:
                self.line.append(byte)

    def _emit_block(self, received_ns, read_index):
        data_lines, event_type, retry_raw = [], "message", None
        for line in self.lines:
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_type = value or "message"
            elif field == "id" and "\0" not in value:
                self.last_id = value
            elif field == "retry":
                retry_raw = value  # Recorded only: this client never reconnects.
        self.emit({
            "received_perf_counter_ns": received_ns,
            "read_index": read_index,
            "completion_body_byte_offset": self.offset,
            "raw_lines": self.lines,
            "has_data": bool(data_lines),
            "data": "\n".join(data_lines) if data_lines else None,
            "event_type": event_type,
            "last_event_id": self.last_id,
            "retry_raw_ignored": retry_raw,
        })
        self.lines = []

    def tail(self):
        return {
            "unfinished_block": bool(self.line or self.lines),
            "pending_line_bytes": len(self.line),
            "pending_complete_lines": len(self.lines),
            "invalid_utf8_lines": self.invalid_utf8_lines,
        }


def prepare(args):
    if any(ord(c) <= 32 or ord(c) == 127 for c in args.url):
        raise InputError("URL must not contain whitespace or control characters.")
    try:
        target = urlsplit(args.url)
        port = target.port
    except ValueError:
        raise InputError("Invalid URL or port.") from None
    if target.scheme not in ("http", "https") or not target.hostname:
        raise InputError("Use an absolute http(s) chat/completions URL.")
    if target.username is not None or target.password is not None or "@" in target.netloc:
        raise InputError("URL user information is forbidden; use --api-key-env.")
    if "?" in args.url or "#" in args.url:
        raise InputError("URL query strings and fragments are forbidden.")
    if not target.path.endswith("/chat/completions"):
        raise InputError("URL must contain the complete path ending in /chat/completions.")
    if port is not None and not (1 <= port <= 65535):
        raise InputError("Invalid port.")
    if not args.label.strip() or len(args.label) > 200 or any(ord(c) < 32 for c in args.label):
        raise InputError("Provide a nonempty observation-point label of at most 200 characters.")
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.max_seconds)):
        raise InputError("Timeout and max-seconds must be positive finite numbers.")
    if args.max_body_bytes < 1:
        raise InputError("max-body-bytes must be positive.")
    try:
        with Path(args.request).open("rb") as handle:
            request_bytes = handle.read(1024 * 1024 + 1)
        if len(request_bytes) > 1024 * 1024:
            raise InputError("Request file exceeds the 1 MiB limit.")
        payload = json_loads(request_bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("Cannot read a valid UTF-8 JSON request file.") from None
    if not isinstance(payload, dict) or payload.get("stream") is not True:
        raise InputError("Request JSON must explicitly contain stream: true; no payload is rewritten.")
    if type(payload.get("n")) is not int or payload["n"] != 1:
        raise InputError("Request JSON must explicitly contain n: 1.")
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise InputError("Request JSON must contain a nonempty model string.")
    if not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise InputError("Request JSON must contain a nonempty messages array.")
    if "stream_options" in payload and not isinstance(payload["stream_options"], dict):
        raise InputError("stream_options must be an object when supplied.")
    token = None
    if args.api_key_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
            raise InputError("api-key-env must be an environment variable name.")
        token = os.environ.get(args.api_key_env)
        if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise InputError("The named API-key variable is unset, empty, or not a printable ASCII token.")
        if token.encode() in request_bytes or token in args.url or token in args.label:
            raise InputError("Credential appears in material that would be saved; capture refused.")
    return target, request_bytes, payload, token


def capture(args):
    target, request_bytes, payload, token = prepare(args)
    out = Path(args.out)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        raise InputError("Output directory already exists; choose a new directory.") from None
    # Identical bytes are sent and saved; supplied JSON is never normalized/re-written.
    (out / "request.requested.json").write_bytes(request_bytes)
    (out / "request.effective.json").write_bytes(request_bytes)
    metadata = {
        "schema": SCHEMA, "label": args.label, "url": args.url,
        "python_version": sys.version.split()[0],
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "payload_modified": False, "validated_stream": True, "validated_n": 1,
        "usage_requested": payload.get("stream_options", {}).get("include_usage"),
        "api_key_env_name": args.api_key_env, "authorization_used": token is not None,
        "socket_timeout_seconds": args.timeout, "max_elapsed_seconds": args.max_seconds,
        "max_body_bytes": args.max_body_bytes, "read_method": "HTTPResponse.read1(<=65536)",
        "redirects": False, "retries": 0, "environment_proxies_used": False,
        "proxy_environment_names_ignored": [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy") if k in os.environ],
        "response_headers_saved": ["content-type", "content-length", "content-encoding", "transfer-encoding"],
        "caveats": CAVEATS,
    }
    dump_json(out / "metadata.json", metadata)
    stamps = {name: None for name in (
        "request_start", "response_headers_ready", "first_content", "first_reasoning",
        "first_reasoning_content", "first_reasoning_field", "last_content", "last_reasoning",
        "last_text", "done", "eof", "capture_end")}
    issues, usage_reports, finish_reasons = [], [], []
    all_event_times, content_times, reasoning_times = [], [], []
    counts = {"reads": 0, "body_bytes": 0, "sse_blocks": 0, "sse_data_events": 0,
              "json_events": 0, "content_events": 0, "reasoning_events": 0}
    text_chars = dict.fromkeys(TEXT_FIELDS, 0)
    http_info, failure, termination, api_error = None, None, None, False
    connection = response = active_socket = None
    phase, start_utc = "setup", None

    def issue(code):
        if code not in issues:
            issues.append(code)

    # Use ordinary buffering; context managers flush on normal timeout/Ctrl-C cleanup.
    with (out / "response_body.bin").open("wb") as raw, \
         (out / "reads.jsonl").open("w", encoding="utf-8") as reads, \
         (out / "events.jsonl").open("w", encoding="utf-8") as events, \
         (out / "content.txt").open("w", encoding="utf-8", errors="backslashreplace") as content, \
         (out / "reasoning_content.txt").open("w", encoding="utf-8", errors="backslashreplace") as rc, \
         (out / "reasoning.txt").open("w", encoding="utf-8", errors="backslashreplace") as reasoning:
        text_files = dict(zip(TEXT_FIELDS, (content, rc, reasoning)))

        def receive_event(event):
            nonlocal api_error
            counts["sse_blocks"] += 1
            event["block_index"] = counts["sse_blocks"]
            ns = event["received_perf_counter_ns"]
            if event["has_data"]:
                counts["sse_data_events"] += 1
                all_event_times.append(ns)
                data = event["data"]
                if data.strip() == "[DONE]":
                    event["kind"] = "done"
                    if stamps["done"] is None:
                        stamps["done"] = ns
                    else:
                        issue("multiple_done_events")
                elif not data:
                    event["kind"] = "empty_data"
                else:
                    try:
                        obj = json_loads(data)
                    except (ValueError, RecursionError):
                        event["kind"] = "invalid_json"
                        issue("invalid_json_event")
                    else:
                        event["kind"], event["json"] = "json", obj
                        counts["json_events"] += 1
                        if not isinstance(obj, dict):
                            issue("non_object_json_event")
                        else:
                            if obj.get("error") is not None:
                                api_error = True
                                issue("server_error_event")
                            if obj.get("usage") is not None:
                                usage_reports.append({"received_perf_counter_ns": ns, "usage": obj["usage"]})
                                if not isinstance(obj["usage"], dict):
                                    issue("non_object_usage")
                                elif "completion_tokens" not in obj["usage"]:
                                    issue("missing_completion_tokens")
                                elif (type(obj["usage"]["completion_tokens"]) is not int
                                      or obj["usage"]["completion_tokens"] < 0):
                                    issue("invalid_completion_tokens")
                            choices = obj.get("choices", [])
                            if not isinstance(choices, list):
                                issue("invalid_choices")
                                choices = []
                            present = set()
                            for choice in choices:
                                if (not isinstance(choice, dict)
                                        or type(choice.get("index")) is not int or choice["index"] != 0):
                                    issue("unexpected_choice_index_or_shape")
                                    continue
                                if choice.get("finish_reason") is not None:
                                    finish_reasons.append({"received_perf_counter_ns": ns,
                                                           "finish_reason": choice["finish_reason"], "index": 0})
                                    if (not isinstance(choice["finish_reason"], str)
                                            or not choice["finish_reason"].strip()):
                                        issue("invalid_finish_reason")
                                delta = choice.get("delta", {})
                                if not isinstance(delta, dict):
                                    issue("non_object_delta")
                                    continue
                                for field in TEXT_FIELDS:
                                    value = delta.get(field)
                                    if value is None or value == "":
                                        continue
                                    if not isinstance(value, str):
                                        issue("unsupported_" + field + "_shape")
                                        continue
                                    text_files[field].write(value)
                                    text_chars[field] += len(value)
                                    present.add(field)
                                    stamps["last_text"] = ns
                            if "content" in present:
                                content_times.append(ns)
                                counts["content_events"] += 1
                                if stamps["first_content"] is None:
                                    stamps["first_content"] = ns
                                stamps["last_content"] = ns
                            if present.intersection(("reasoning_content", "reasoning")):
                                reasoning_times.append(ns)
                                counts["reasoning_events"] += 1
                                if stamps["first_reasoning"] is None:
                                    stamps["first_reasoning"] = ns
                                stamps["last_reasoning"] = ns
                            for field, stamp in (("reasoning_content", "first_reasoning_content"),
                                                 ("reasoning", "first_reasoning_field")):
                                if field in present and stamps[stamp] is None:
                                    stamps[stamp] = ns
                            if stamps["done"] is not None:
                                issue("json_event_after_done")
            else:
                event["kind"] = "no_data_block"
            json_line(events, event)

        decoder = SSEDecoder(receive_event)

        def apply_timeout():
            remaining = args.max_seconds - (time.perf_counter_ns() - stamps["request_start"]) / 1e9
            if remaining <= 0:
                raise CaptureStop("elapsed_limit")
            seconds = min(args.timeout, remaining)
            if active_socket is not None:
                active_socket.settimeout(seconds)
            elif connection is not None:
                connection.timeout = seconds

        try:
            kind = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            connection = kind(target.hostname, target.port, timeout=min(args.timeout, args.max_seconds))
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
                       "Accept-Encoding": "identity", "Connection": "close"}
            if token is not None:
                headers["Authorization"] = "Bearer " + token
            start_utc = datetime.now(timezone.utc).isoformat()
            stamps["request_start"] = time.perf_counter_ns()
            phase = "send_request"
            connection.request("POST", target.path, body=request_bytes, headers=headers)
            active_socket = connection.sock
            apply_timeout()
            phase = "response_headers"
            response = connection.getresponse()
            stamps["response_headers_ready"] = time.perf_counter_ns()
            http_info = {"status": response.status, "version": response.version,
                         "headers": {name: response.getheader(name) for name in metadata["response_headers_saved"]}}
            mime = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
            if response.status != 200:
                issue("http_status_not_200")
            if mime != "text/event-stream":
                issue("content_type_not_event_stream")
            if encoding not in ("", "identity"):
                issue("unsupported_content_encoding")
            parse_sse = response.status == 200 and mime == "text/event-stream" and encoding in ("", "identity")
            phase = "read_body"
            while True:
                apply_timeout()
                remaining = args.max_body_bytes - counts["body_bytes"]
                if remaining <= 0:
                    raise CaptureStop("body_limit")
                read_start = time.perf_counter_ns()
                try:
                    data = response.read1(min(65536, remaining))
                except (Exception, KeyboardInterrupt) as exc:
                    json_line(reads, {"read_start_perf_counter_ns": read_start,
                        "returned_perf_counter_ns": time.perf_counter_ns(), "bytes": None,
                        "error_type": type(exc).__name__})
                    raise
                received = time.perf_counter_ns()
                counts["reads"] += 1
                offset = counts["body_bytes"]
                counts["body_bytes"] += len(data)
                json_line(reads, {"read_index": counts["reads"], "read_start_perf_counter_ns": read_start,
                    "received_perf_counter_ns": received, "bytes": len(data),
                    "body_offset_start": offset, "body_offset_end": counts["body_bytes"]})
                if not data:
                    stamps["eof"] = received
                    termination = "eof"
                    if response.length is not None and response.length > 0:
                        issue("truncated_content_length")
                    break
                raw.write(data)
                if parse_sse:
                    decoder.feed(data, received, counts["reads"])
        except CaptureStop as exc:
            termination = str(exc)  # Only our fixed, credential-free codes.
            issue(termination)
        except KeyboardInterrupt:
            termination = "interrupted"
            issue(termination)
        except Exception as exc:
            termination = "timeout" if isinstance(exc, TimeoutError) else "transport_or_client_error"
            issue(termination)
            # Arbitrary exception text can contain credentials. Save only class and phase.
            failure = {"type": type(exc).__name__, "phase": phase}
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            stamps["capture_end"] = time.perf_counter_ns()

    tail = decoder.tail()
    if tail["unfinished_block"]:
        issue("unfinished_sse_block")
    if tail["invalid_utf8_lines"]:
        issue("invalid_utf8_replaced_in_parsed_events")
    if stamps["done"] is None:
        issue("missing_done")
    if not usage_reports:
        issue("missing_usage")
    if not finish_reasons:
        issue("missing_finish_reason")
    fatal_response = http_info is not None and (http_info["status"] != 200 or
        "content_type_not_event_stream" in issues or "unsupported_content_encoding" in issues)
    status = "error" if fatal_response or api_error or (failure and counts["body_bytes"] == 0) else (
        "partial" if issues else "complete")
    start = stamps["request_start"]
    elapsed = {k + "_ms": (v - start) / 1e6 if v is not None and start is not None else None
               for k, v in stamps.items()}
    summary = {
        "schema": SCHEMA, "label": args.label, "status": status, "termination": termination,
        "request_start_utc": start_utc, "perf_counter_ns": stamps, "elapsed_from_request_start": elapsed,
        "ttft_content_receive_ms": elapsed["first_content_ms"],
        "first_reasoning_receive_ms": elapsed["first_reasoning_ms"],
        "counts": counts, "text_characters": text_chars, "http": http_info, "failure": failure,
        "issues": issues, "sse_tail": tail,
        "sse_data_event_gap_ms": gap_stats(all_event_times),
        "sse_content_event_gap_ms": gap_stats(content_times),
        "sse_reasoning_event_gap_ms": gap_stats(reasoning_times),
        "usage_reports": usage_reports, "server_usage": usage_reports[-1]["usage"] if usage_reports else None,
        "finish_reasons": finish_reasons,
        "finish_reason": finish_reasons[-1]["finish_reason"] if finish_reasons else None,
        "caveats": CAVEATS,
    }
    dump_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full http(s) URL ending in /chat/completions; no query or credentials")
    parser.add_argument("--request", required=True, help="Path to nonsecret fixed UTF-8 JSON (stream:true, n:1 required)")
    parser.add_argument("--out", required=True, help="New output directory; an existing directory is refused")
    parser.add_argument("--label", required=True, help="Observation point and CPU/run label, e.g. pre-direct-01")
    parser.add_argument("--api-key-env", help="Name of the environment variable containing a Bearer token")
    parser.add_argument("--timeout", type=float, default=60.0, help="Socket I/O timeout in seconds (default 60)")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="Elapsed budget in seconds (default 300; DNS resolution is OS-controlled)")
    parser.add_argument("--max-body-bytes", type=int, default=16777216, help="Raw body cap (default 16 MiB; reaching the cap is partial)")
    args = parser.parse_args(argv)
    try:
        summary = capture(args)
    except InputError as exc:
        parser.error(str(exc))
    except OSError as exc:
        print(json.dumps({"status": "error", "phase": "artifact_setup_or_save", "type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "termination": summary["termination"],
        "body_bytes": summary["counts"]["body_bytes"],
        "ttft_content_receive_ms": summary["ttft_content_receive_ms"],
        "first_reasoning_receive_ms": summary["first_reasoning_receive_ms"],
        "issues": summary["issues"]}, ensure_ascii=True))
    if summary["termination"] == "interrupted":
        return 130
    return {"complete": 0, "partial": 3, "error": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
