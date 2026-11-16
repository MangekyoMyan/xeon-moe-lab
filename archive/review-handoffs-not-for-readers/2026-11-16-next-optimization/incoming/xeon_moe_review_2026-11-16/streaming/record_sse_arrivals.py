#!/usr/bin/env python3
"""Send one explicitly requested streaming POST and record bounded arrival metadata.

No automatic retry or redirect.  Raw request/response text and API keys are not
written to the capture.  Native llama /completion and OpenAI choices are parsed.
Run this only when you intend to send a real generation request to --url.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import ssl
import time
from urllib.parse import urlsplit


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_meta(value):
    if not isinstance(value, str):
        return None
    encoded = value.encode("utf-8")
    return {"characters": len(value), "utf8_bytes": len(encoded), "sha256": sha(encoded)}


def strict_json(data):
    def invalid_constant(_):
        raise ValueError("Non-finite JSON constant")
    return json.loads(data, parse_constant=invalid_constant)


def nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1


def numbers(value, keys):
    if not isinstance(value, dict):
        return {}
    return {k: value[k] for k in keys if isinstance(value.get(k), (int, float))
            and not isinstance(value[k], bool) and 0 <= value[k] <= 1e100}


def intervals(timestamps):
    values = sorted((b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:]))
    def percentile(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        lo = int(index); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"intervals": len(values), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "max_ms": max(values) if values else None,
            "percentile_definition": "linear interpolation at (n-1)*p"}


@dataclass(frozen=True)
class Limits:
    body_bytes: int = 8 * 1024 * 1024
    event_bytes: int = 256 * 1024
    events: int = 4096
    reads: int = 32768
    timeout_seconds: float = 30.0
    duration_seconds: float = 300.0


class CaptureFailure(Exception):
    pass


class SSEParser:
    """SSE lines are buffered as bytes, so UTF-8 may span arbitrary body reads."""
    def __init__(self, limit, clock):
        self.limit = limit; self.clock = clock
        self.line = bytearray(); self.data = []; self.data_bytes = 0
        self.event_type = "message"; self.first_line = True; self.skip_lf = False
        self.comments = 0; self.ignored_fields = 0

    def append(self, fragment):
        if len(self.line) + self.data_bytes + len(fragment) + len(self.event_type.encode()) > self.limit:
            raise CaptureFailure("sse_event_size_limit")
        self.line.extend(fragment)

    def process_line(self, read_index):
        raw = bytes(self.line); self.line.clear()
        if self.first_line:
            raw = raw.removeprefix(b"\xef\xbb\xbf"); self.first_line = False
        line = raw.decode("utf-8")
        if not line:
            frame = None
            if self.data:
                frame = {"data": "\n".join(self.data), "event_type": self.event_type or "message",
                         "completed_monotonic_ns": self.clock(), "body_read_index": read_index}
            self.data = []; self.data_bytes = 0; self.event_type = "message"
            return frame
        if line.startswith(":"):
            self.comments += 1
            return None
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self.data.append(value); self.data_bytes += len(value.encode("utf-8")) + 1
        elif name == "event":
            self.event_type = value
        else:
            self.ignored_fields += 1  # id/retry are not used: this recorder never reconnects.
        return None

    def feed(self, chunk, read_index):
        if self.skip_lf:
            if chunk.startswith(b"\n"):
                chunk = chunk[1:]
            self.skip_lf = False
        position = 0
        for match in re.finditer(rb"\r\n|\r|\n", chunk):
            self.append(chunk[position:match.start()])
            frame = self.process_line(read_index)
            if frame is not None:
                yield frame
            position = match.end()
            if match.group() == b"\r" and position == len(chunk):
                self.skip_lf = True
        self.append(chunk[position:])

    def finish(self):
        bytes(self.line).decode("utf-8")  # Detect an incomplete UTF-8 suffix too.
        return not self.line and not self.data and self.event_type == "message"


class Observations:
    def __init__(self, started, limits):
        self.started = started; self.limits = limits
        self.events = []; self.errors = []; self.markers = []
        self.native_counts = []; self.native_timings = {}; self.usage = {}
        self.native_identity = {}
        self.supported_events = 0; self.content_times = []; self.reasoning_times = []
        self.last_content_event = self.last_reasoning_event = None
        self.content_hash = hashlib.sha256(); self.content_chars = 0

    def error(self, kind):
        if kind not in self.errors:
            self.errors.append(kind)

    def content(self, value, event, field, reasoning=False):
        if value is None:
            return
        if not isinstance(value, str):
            self.error("non_string_text_field")
            event[field] = {"type": type(value).__name__}
            return
        event[field] = text_meta(value)
        if value:
            event_index = len(self.events) - 1
            previous = self.last_reasoning_event if reasoning else self.last_content_event
            if previous != event_index:
                (self.reasoning_times if reasoning else self.content_times).append(event["completed_offset_ns"])
                if reasoning:
                    self.last_reasoning_event = event_index
                else:
                    self.last_content_event = event_index
            if not reasoning:
                self.content_hash.update(value.encode("utf-8")); self.content_chars += len(value)

    def observe(self, frame):
        if len(self.events) >= self.limits.events:
            raise CaptureFailure("sse_event_count_limit")
        data = frame["data"]
        event = {"index": len(self.events), "body_read_index": frame["body_read_index"],
                 "completed_monotonic_ns": frame["completed_monotonic_ns"],
                 "completed_offset_ns": frame["completed_monotonic_ns"] - self.started,
                 "data": text_meta(data),
                 "sse_event_type": frame["event_type"] if frame["event_type"] in ("message", "error", "ping") else "other",
                 "sse_event_type_metadata": text_meta(frame["event_type"])}
        self.events.append(event)
        if frame["event_type"] == "error":
            self.error("sse_error_event")
        if data.strip() == "[DONE]":
            event["kind"] = "done_marker"; self.markers.append(event["index"])
            return
        try:
            value = strict_json(data)
        except (ValueError, TypeError):
            event["kind"] = "non_json"; self.error("non_json_event_payload")
            return
        if not isinstance(value, dict):
            event["kind"] = "non_object_json"; self.error("non_object_json_payload")
            return
        if value.get("error") is not None and value.get("error") is not False:
            event["server_error_present"] = True; self.error("server_error_event")
        if self.markers:
            self.error("json_event_after_completion_marker")
        if isinstance(value.get("choices"), list) or (isinstance(value.get("usage"), dict) and "content" not in value):
            event["kind"] = "openai_choices" if "choices" in value else "openai_usage"
            self.supported_events += 1
            event["choices"] = []
            choices = value.get("choices", [])
            if not isinstance(choices, list):
                self.error("invalid_choices_shape"); choices = []
            for position, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    self.error("invalid_choice_shape"); continue
                item = {"index": choice.get("index") if nonnegative_int(choice.get("index")) else position,
                        "completed_offset_ns": event["completed_offset_ns"]}
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    self.error("invalid_delta_shape"); delta = {}
                self.content(delta.get("content", choice.get("text")), item, "content")
                for field in ("reasoning_content", "reasoning"):
                    self.content(delta.get(field), item, field, reasoning=True)
                if "role" in delta:
                    item["role"] = delta["role"] if delta["role"] in ("assistant", "tool") else "other"
                calls = delta.get("tool_calls") or []
                if not isinstance(calls, list):
                    self.error("invalid_tool_calls_shape"); calls = []
                item["tool_calls"] = []
                for call in calls:
                    if not isinstance(call, dict):
                        self.error("invalid_tool_call_shape"); continue
                    function = call.get("function") or {}
                    if not isinstance(function, dict):
                        self.error("invalid_function_shape"); function = {}
                    item["tool_calls"].append({"index": call.get("index") if nonnegative_int(call.get("index")) else None,
                                               "id": text_meta(call.get("id")), "function_name": text_meta(function.get("name")),
                                               "arguments": text_meta(function.get("arguments"))})
                finish = choice.get("finish_reason")
                item["finish_reason"] = finish if finish in (None, "stop", "length", "tool_calls", "function_call", "content_filter") else "other"
                event["choices"].append(item)
            usage = value.get("usage")
            if isinstance(usage, dict):
                self.usage = numbers(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
                self.usage["completion_tokens_details"] = numbers(usage.get("completion_tokens_details"),
                    ("reasoning_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"))
                event["usage"] = self.usage
        elif any(key in value for key in ("content", "tokens_predicted", "stop", "timings")):
            event["kind"] = "native_completion"; self.supported_events += 1
            conflict = False
            for field in ("index", "id_slot"):
                if nonnegative_int(value.get(field)):
                    event["native_" + field] = value[field]
                    if field in self.native_identity and self.native_identity[field] != value[field]:
                        conflict = True
                    self.native_identity.setdefault(field, value[field])
            if conflict:
                self.error("multiple_native_streams_not_supported")
                event["native_stream_conflict"] = True
                return  # Do not combine another generation's counter/text with the first.
            self.content(value.get("content"), event, "content")
            if isinstance(value.get("tokens"), list):
                event["token_ids_present_in_this_event"] = len(value["tokens"])  # Not total generated tokens.
            if "tokens_predicted" in value:
                count = value["tokens_predicted"]
                if nonnegative_int(count):
                    previous = self.native_counts[-1] if self.native_counts else None
                    observation = {"event_index": event["index"], "count": count,
                                   "delta_from_previous_observation": count - previous["count"] if previous else None,
                                   "completed_offset_ns": event["completed_offset_ns"]}
                    if previous and count < previous["count"]:
                        self.error("native_counter_decreased")
                    self.native_counts.append(observation); event["tokens_predicted"] = count
                else:
                    self.error("invalid_native_counter")
            timing = numbers(value.get("timings"), ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second"))
            if timing:
                event["timings"] = timing; self.native_timings = timing
            if value.get("stop") is True:
                event["native_stop"] = True; self.markers.append(event["index"])
        else:
            event["kind"] = "other_json"


def parse_url(url):
    value = urlsplit(url)
    if value.scheme not in ("http", "https") or not value.hostname or value.username is not None or value.password is not None or value.fragment:
        raise CaptureFailure("invalid_url_or_userinfo_or_fragment")
    port = value.port if value.port is not None else (443 if value.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        raise CaptureFailure("invalid_port")
    target = value.path or "/"
    if value.query:
        target += "?" + value.query
    return value, port, target


def capture(url, body, api_key, limits, connection_factory=None, clock=time.monotonic_ns):
    value, port, target = parse_url(url)
    started = clock(); observed = Observations(started, limits)
    parser = SSEParser(limits.event_bytes, clock); reads = []; total = 0
    stages = {"request_started_offset_ns": 0}; connection = response = None
    body_hash = hashlib.sha256(); eof = False; http_status = None; requests_attempted = 0
    started_utc = datetime.now(timezone.utc).isoformat()
    def remaining():
        seconds = limits.duration_seconds - (clock() - started) / 1e9
        if seconds <= 0:
            raise CaptureFailure("capture_duration_limit")
        return min(limits.timeout_seconds, seconds)
    try:
        if connection_factory is None:
            connection = (http.client.HTTPSConnection(value.hostname, port, timeout=remaining(), context=ssl.create_default_context())
                          if value.scheme == "https" else http.client.HTTPConnection(value.hostname, port, timeout=remaining()))
        else:
            connection = connection_factory(value.hostname, port, remaining())
        headers = {"Accept": "text/event-stream", "Accept-Encoding": "identity", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        requests_attempted += 1
        connection.request("POST", target, body=body, headers=headers)
        stages["request_sent_offset_ns"] = clock() - started
        sock = connection.sock
        sock.settimeout(remaining())
        response = connection.getresponse()
        stages["response_headers_completed_offset_ns"] = clock() - started
        http_status = response.status
        if not 200 <= http_status < 300:
            raise CaptureFailure("http_status_failure")
        if (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower() != "text/event-stream":
            raise CaptureFailure("non_sse_content_type")
        if (response.getheader("Content-Encoding") or "identity").strip().lower() != "identity":
            raise CaptureFailure("unsupported_content_encoding")
        content_length = response.getheader("Content-Length")
        expected_length = int(content_length) if content_length and content_length.isdecimal() and not response.chunked else None
        while True:
            sock.settimeout(remaining())
            chunk = response.read1(min(4096, limits.body_bytes - total + 1))
            arrived = clock()
            if not chunk:
                stages["response_eof_offset_ns"] = arrived - started; eof = True
                if expected_length is not None and total != expected_length:
                    raise CaptureFailure("truncated_content_length")
                if not parser.finish():
                    raise CaptureFailure("unfinished_sse_frame_at_eof")
                break
            if len(reads) >= limits.reads:
                raise CaptureFailure("body_read_count_limit")
            read_index = len(reads)
            reads.append({"index": read_index, "completed_monotonic_ns": arrived, "completed_offset_ns": arrived - started,
                          "body_offset": total, "bytes": len(chunk), "sha256": sha(chunk)})
            stages.setdefault("first_body_observed_offset_ns", arrived - started)
            total += len(chunk); body_hash.update(chunk)
            if total > limits.body_bytes:
                raise CaptureFailure("body_size_limit")
            for frame in parser.feed(chunk, read_index):
                observed.observe(frame)
    except CaptureFailure as error:
        observed.error(str(error))
    except (OSError, http.client.HTTPException, ValueError, RecursionError) as error:
        observed.error("transport_or_encoding_error:" + type(error).__name__)
    except KeyboardInterrupt:
        observed.error("interrupted")
    finally:
        for resource in (response, connection):
            if resource is not None:
                try:
                    resource.close()
                except OSError:
                    observed.error("connection_close_error")
    if not observed.markers:
        observed.error("no_application_completion_marker")
    if not eof:
        observed.error("http_body_eof_not_observed")
    if not observed.supported_events:
        observed.error("no_supported_generation_event")
    native_count_minus_timing_n = (observed.native_counts[-1]["count"] - observed.native_timings["predicted_n"]
                                  if observed.native_counts and "predicted_n" in observed.native_timings else None)
    stages["capture_finished_offset_ns"] = clock() - started
    if observed.content_times:
        stages["first_content_text_observed_offset_ns"] = observed.content_times[0]
    if observed.reasoning_times:
        stages["first_reasoning_text_observed_offset_ns"] = observed.reasoning_times[0]
    if observed.markers:
        stages["application_completion_offset_ns"] = observed.events[observed.markers[0]]["completed_offset_ns"]
    by_read = Counter(event["body_read_index"] for event in observed.events)
    visible_by_read = Counter(event["body_read_index"] for event in observed.events if
        (event.get("content") or {}).get("characters", 0) or any((c.get("content") or {}).get("characters", 0) for c in event.get("choices", [])))
    return {
        "schema_version": 1, "started_at_utc": started_utc, "clock": "time.monotonic_ns",
        "origin_monotonic_ns": started,
        "endpoint": {"scheme": value.scheme, "host": value.hostname, "port": port, "path": value.path or "/", "query_present": bool(value.query)},
        "request": {"sent_json_bytes": len(body), "sent_json_sha256": sha(body), "stream": True, "authorization_header_used": bool(api_key), "http_requests_attempted": requests_attempted},
        "http": {"status": http_status, "chunked": bool(response.chunked) if response is not None else None, "body_eof_observed": eof},
        "success": not observed.errors, "errors": observed.errors,
        "application_completion_marker_seen": bool(observed.markers), "stages_ns": stages,
        "body_reads": reads, "sse_events": observed.events,
        "summary": {
            "body_bytes_returned_by_read1": total, "body_sha256": body_hash.hexdigest(),
            "body_read_count": len(reads), "sse_event_count": len(observed.events), "comments": parser.comments,
            "body_read_intervals": intervals([r["completed_offset_ns"] for r in reads]),
            "sse_completion_intervals": intervals([e["completed_offset_ns"] for e in observed.events]),
            "content_event_intervals": intervals(observed.content_times), "reasoning_event_intervals": intervals(observed.reasoning_times),
            "body_reads_with_multiple_sse_events": sum(n > 1 for n in by_read.values()),
            "body_reads_with_multiple_content_events": sum(n > 1 for n in visible_by_read.values()),
            "largest_sse_events_in_one_body_read": max(by_read.values(), default=0),
            "native_cumulative_observations": observed.native_counts,
            "native_cumulative_jumps_over_one": [r for r in observed.native_counts if r["delta_from_previous_observation"] is not None and r["delta_from_previous_observation"] > 1],
            "native_final_timings": observed.native_timings, "openai_final_usage": observed.usage,
            "native_final_counter_minus_timing_predicted_n": native_count_minus_timing_n,
            "content_characters": observed.content_chars, "concatenated_content_sha256": observed.content_hash.hexdigest(),
        },
        "limits": vars(limits),
        "interpretation": [
            "read1 returns HTTP body bytes after transfer decoding; a read is not a TCP packet or a generation step.",
            "SSE completion timestamps include client parsing. Events with the same body_read_index arrived in one observed read.",
            "Headers-completed and first-body-read are observation proxies, not the exact first HTTP wire byte (TTFB).",
            "First content text is a visibility proxy; no browser/Open WebUI render timestamp is measured. Reasoning is recorded separately.",
            "SSE events, characters and token IDs present in a partial event are not total generated tokens.",
            "Native cumulative counts and OpenAI usage are server reports; they do not reveal each internal token completion time.",
            "tokens_predicted and timings.predicted_n remain separate; their definitions can differ by server version.",
            "The content hash concatenates observed content fields; with multiple choices it interleaves choices in arrival order.",
            "Raw content, reasoning, tool arguments, request prompts, query values and API keys are omitted.",
            "Limits stop this client; closing the connection does not guarantee that a remote model stops immediately.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--request-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new JSON file; never overwritten")
    ap.add_argument("--timeout-seconds", type=float, default=30)
    ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--max-body-mib", type=int, default=8)
    args = ap.parse_args(argv)
    if not (math.isfinite(args.timeout_seconds) and math.isfinite(args.max_seconds) and
            0 < args.timeout_seconds <= 3600 and 0 < args.max_seconds <= 3600 and 1 <= args.max_body_mib <= 64):
        ap.error("Timeout/duration must be finite in (0,3600]; max-body-mib must be 1..64")
    try:
        parse_url(args.url)
        with args.request_json.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise CaptureFailure("request_file_size_limit_1MiB")
        request = strict_json(raw)
        if not isinstance(request, dict):
            raise CaptureFailure("request_json_must_be_object")
        changed_stream = request.get("stream") is not True
        request["stream"] = True
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        # Reserve the new output before making a request; no overwrite or retry.
        with args.output.open("x", encoding="utf-8") as output:
            result = capture(args.url, body, os.environ.get("XEON_REVIEW_API_KEY"),
                             Limits(body_bytes=args.max_body_mib * 1024 * 1024, timeout_seconds=args.timeout_seconds,
                                    duration_seconds=args.max_seconds))
            result["request"].update(request_file_sha256=sha(raw), request_file_bytes=len(raw), stream_changed_to_true=changed_stream)
            json.dump(result, output, ensure_ascii=False, indent=2); output.write("\n")
        print(json.dumps({"output": str(args.output), "success": result["success"], "sse_events": len(result["sse_events"]), "errors": result["errors"]}))
        return 0 if result["success"] else 2
    except (OSError, ValueError, RecursionError, CaptureFailure) as error:
        # Do not print request bodies, full URLs, response text or credentials.
        print(json.dumps({"success": False, "error_type": type(error).__name__, "request_not_retried": True}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
