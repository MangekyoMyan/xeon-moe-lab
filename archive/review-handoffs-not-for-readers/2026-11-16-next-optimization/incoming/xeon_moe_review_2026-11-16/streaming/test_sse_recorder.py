"""Synthetic-only tests. No socket connections, model calls, sleeps or services."""
from contextlib import redirect_stdout
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import record_sse_arrivals as recorder


class FakeClock:
    def __init__(self):
        self.value = 1_000_000_000

    def __call__(self):
        self.value += 1_000
        return self.value


class FakeSocket:
    def __init__(self, wire=b""):
        self.wire = wire

    def settimeout(self, _):
        pass

    def makefile(self, _):
        return io.BytesIO(self.wire)


class FakeResponse:
    chunked = False

    def __init__(self, chunks, status=200, headers=None):
        self.chunks = list(chunks); self.status = status
        self.headers = {"Content-Type": "text/event-stream", **(headers or {})}

    def getheader(self, name):
        return self.headers.get(name)

    def read1(self, size):
        if not self.chunks:
            return b""
        item = self.chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        if len(item) > size:
            self.chunks.insert(0, item[size:])
        return item[:size]

    def close(self):
        pass


class FakeConnection:
    def __init__(self, response, sock=None):
        self.response = response; self.sock = sock or FakeSocket(); self.requests = []

    def request(self, method, target, body, headers):
        self.requests.append((method, target, body, headers))

    def getresponse(self):
        if isinstance(self.response, http.client.HTTPResponse):
            self.response.begin()
        return self.response

    def close(self):
        pass


def sse(value, newline=b"\n"):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return b"data: " + data.encode("utf-8") + newline + newline


def native(text="", count=1, stop=False):
    return sse({"content": text, "tokens_predicted": count, "stop": stop})


class RecorderTests(unittest.TestCase):
    def setUp(self):
        # Fail before any real DNS/connection if a test accidentally drops its fake.
        guard = patch("socket.create_connection", side_effect=AssertionError("Network forbidden in these tests"))
        guard.start(); self.addCleanup(guard.stop)

    def run_capture(self, chunks=None, *, response=None, limits=None, body=b'{"prompt":"secret-prompt","stream":true}', key=None, sock=None):
        connection = FakeConnection(response or FakeResponse(chunks), sock)
        result = recorder.capture("http://synthetic.invalid/completion?private=query-secret", body, key,
                                  limits or recorder.Limits(), connection_factory=lambda *args: connection,
                                  clock=FakeClock())
        self.assertEqual(len(connection.requests), 1)
        return result, connection

    def test_native_crlf_split_utf8_multiline_comments_and_bom(self):
        payload = (b"\xef\xbb\xbf: heartbeat\r\n\r\n"
                   b"data: {\"content\": \"" + "あい".encode() + b"\",\r\n"
                   b"data: \"tokens_predicted\": 1, \"stop\": false}\r\n\r\n"
                   b"id: ignored\r\n" + native(count=1, stop=True).replace(b"\n", b"\r\n"))
        # Every byte separately splits UTF-8, CRLF, BOM and JSON in all relevant places.
        result, _ = self.run_capture([bytes([byte]) for byte in payload])
        self.assertTrue(result["success"], result["errors"])
        self.assertEqual(result["summary"]["comments"], 1)
        self.assertEqual(result["summary"]["content_characters"], 2)
        self.assertEqual(result["summary"]["sse_event_count"], 2)
        self.assertEqual(result["summary"]["body_read_count"], len(payload))
        event = result["sse_events"][0]
        self.assertGreaterEqual(event["completed_monotonic_ns"], result["body_reads"][event["body_read_index"]]["completed_monotonic_ns"])

    def test_native_same_read_grouping_counter_jump_and_timing_difference(self):
        first = native("甲", 1) + native("乙丙", 3)
        last = sse({"content": "", "tokens_predicted": 3, "stop": True,
                    "timings": {"predicted_n": 2, "predicted_ms": 200, "prompt_ms": 80}})
        result, _ = self.run_capture([first, last])
        self.assertTrue(result["success"], result["errors"])
        summary = result["summary"]
        self.assertEqual(summary["body_reads_with_multiple_content_events"], 1)
        self.assertEqual(summary["largest_sse_events_in_one_body_read"], 2)
        self.assertEqual(summary["native_cumulative_jumps_over_one"][0]["delta_from_previous_observation"], 2)
        self.assertEqual(summary["native_final_counter_minus_timing_predicted_n"], 1)
        self.assertEqual(summary["content_event_intervals"]["intervals"], 1)

    def test_openai_text_reasoning_tools_usage_no_raw_text_or_key(self):
        secret_content = "secret-content-unique"; secret_reason = "secret-reasoning-unique"
        payload = b"".join([
            sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            sse({"choices": [{"index": 0, "delta": {"reasoning_content": secret_reason}}]}),
            sse({"choices": [{"index": 0, "delta": {"content": secret_content}},
                             {"index": 1, "delta": {"content": "other-choice-unique"}}]}),
            sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "secret-call-id",
                "function": {"name": "secret-function-name", "arguments": "secret-tool-arguments"}}]}, "finish_reason": "tool_calls"}]}),
            sse({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                "completion_tokens_details": {"reasoning_tokens": 2}}}), sse("[DONE]")])
        result, connection = self.run_capture([payload], key="secret-api-key")
        self.assertTrue(result["success"], result["errors"])
        encoded = json.dumps(result)
        for secret in (secret_content, secret_reason, "other-choice-unique", "secret-call-id", "secret-function-name",
                       "secret-tool-arguments", "secret-prompt", "secret-api-key", "query-secret"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(connection.requests[0][3]["Authorization"], "Bearer secret-api-key")
        self.assertEqual(result["summary"]["openai_final_usage"]["completion_tokens"], 4)
        self.assertEqual(result["summary"]["content_event_intervals"]["intervals"], 0)  # Two choices, one event.
        self.assertLess(result["stages_ns"]["first_reasoning_text_observed_offset_ns"], result["stages_ns"]["first_content_text_observed_offset_ns"])
        self.assertEqual(result["summary"]["sse_event_count"], 6)

    def test_non_json_server_error_and_missing_terminal_are_not_success(self):
        for payload, expected in [
            (native("a") + sse("invalid-sensitive-payload") + sse("[DONE]"), "non_json_event_payload"),
            (b"event: error\n" + sse({"error": {"message": "secret-error-body"}}) + sse("[DONE]"), "sse_error_event"),
            (native("a"), "no_application_completion_marker"),
            (native("a", stop=True) + b"data: {\"content\": \"unfinished", "unfinished_sse_frame_at_eof"),
        ]:
            with self.subTest(expected=expected):
                result, _ = self.run_capture([payload])
                self.assertFalse(result["success"])
                self.assertIn(expected, result["errors"])
                self.assertNotIn("invalid-sensitive-payload", json.dumps(result))
                self.assertNotIn("secret-error-body", json.dumps(result))

    def test_transport_and_invalid_utf8_fail_after_completion(self):
        for chunks in ([native("a", stop=True), OSError("secret-socket-error")],
                       [native("a", stop=True) + b"data: \xe3"]):
            with self.subTest(chunks=repr(chunks)):
                result, _ = self.run_capture(chunks)
                self.assertFalse(result["success"])
                self.assertTrue(result["application_completion_marker_seen"])
                self.assertNotIn("secret-socket-error", json.dumps(result))

    def test_http_status_redirect_content_type_and_encoding_fail_without_retry(self):
        cases = [(FakeResponse([], status=status), "http_status_failure") for status in (302, 401, 500)]
        cases += [(FakeResponse([], headers={"Content-Type": "application/json"}), "non_sse_content_type"),
                  (FakeResponse([], headers={"Content-Encoding": "gzip"}), "unsupported_content_encoding")]
        for response, expected in cases:
            with self.subTest(expected=expected, status=response.status):
                result, connection = self.run_capture(response=response)
                self.assertFalse(result["success"])
                self.assertIn(expected, result["errors"])
                self.assertEqual(len(connection.requests), 1)

    def test_real_httpresponse_chunked_decode_and_truncation(self):
        payload = native("synthetic", 1) + native(count=1, stop=True)
        def wire(chunks, terminator):
            return (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" +
                    b"".join(format(len(c), "x").encode() + b"\r\n" + c + b"\r\n" for c in chunks) + terminator)
        for terminator, success in ((b"0\r\n\r\n", True), (b"", False)):
            with self.subTest(success=success):
                sock = FakeSocket(wire([payload[:17], payload[17:]], terminator))
                response = http.client.HTTPResponse(sock)
                result, _ = self.run_capture(response=response, sock=sock)
                self.assertEqual(result["success"], success, result["errors"])
                self.assertTrue(result["http"]["chunked"])
                self.assertTrue(result["application_completion_marker_seen"])
                if not success:
                    self.assertIn("transport_or_encoding_error:IncompleteRead", result["errors"])

    def test_real_httpresponse_content_length_truncation(self):
        payload = native("x", stop=True)
        sock = FakeSocket(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: " +
                          str(len(payload) + 10).encode() + b"\r\n\r\n" + payload)
        result, _ = self.run_capture(response=http.client.HTTPResponse(sock), sock=sock)
        self.assertFalse(result["success"])
        self.assertIn("truncated_content_length", result["errors"])

    def test_bounded_body_event_count_event_size_read_count_and_duration(self):
        cases = [
            (recorder.Limits(body_bytes=10), [native("a", stop=True)], "body_size_limit"),
            (recorder.Limits(event_bytes=15), [native("a", stop=True)], "sse_event_size_limit"),
            (recorder.Limits(events=1), [native("a") + native(stop=True)], "sse_event_count_limit"),
            (recorder.Limits(reads=1), [native("a"), native(stop=True)], "body_read_count_limit"),
            (recorder.Limits(duration_seconds=.000005), [native("a", stop=True)], "capture_duration_limit"),
        ]
        for limits, chunks, expected in cases:
            with self.subTest(expected=expected):
                result, _ = self.run_capture(chunks, limits=limits)
                self.assertFalse(result["success"])
                self.assertIn(expected, result["errors"])

    def test_interval_percentiles(self):
        result = recorder.intervals([0, 100_000_000, 300_000_000, 700_000_000])
        self.assertEqual(result["p50_ms"], 200)
        self.assertEqual(result["p95_ms"], 380)
        self.assertEqual(result["max_ms"], 400)
        self.assertIsNone(recorder.intervals([0])["p50_ms"])

    def test_native_multiplex_is_reported_not_merged(self):
        payload = b"".join(sse(value) for value in [
            {"index": 0, "id_slot": 0, "content": "a", "tokens_predicted": 1, "stop": False},
            {"index": 1, "id_slot": 1, "content": "bc", "tokens_predicted": 3, "stop": False},
            {"index": 0, "id_slot": 0, "content": "", "tokens_predicted": 1, "stop": True},
        ])
        result, _ = self.run_capture([payload])
        self.assertFalse(result["success"])
        self.assertIn("multiple_native_streams_not_supported", result["errors"])
        self.assertEqual(result["summary"]["native_cumulative_jumps_over_one"], [])
        self.assertEqual(result["summary"]["content_characters"], 1)

    def test_cli_new_output_only_forced_stream_and_env_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); request = root / "request.json"; output = root / "result.json"
            request.write_text('{"prompt":"CLI-secret-prompt","stream":false}', encoding="utf-8")
            argv = ["--url", "http://synthetic.invalid/completion", "--request-json", str(request), "--output", str(output)]
            calls = []
            def fake_capture(url, body, api_key, limits):
                calls.append((url, body, api_key))
                return {"request": {}, "success": True, "sse_events": [], "errors": []}
            with patch.object(recorder, "capture", side_effect=fake_capture), patch.dict("os.environ", {"XEON_REVIEW_API_KEY": "CLI-secret-key"}), redirect_stdout(io.StringIO()):
                self.assertEqual(recorder.main(argv), 0)
                original = output.read_bytes()
                self.assertEqual(recorder.main(argv), 2)
                self.assertEqual(output.read_bytes(), original)
            self.assertEqual(len(calls), 1)
            self.assertTrue(json.loads(calls[0][1])["stream"])
            self.assertEqual(calls[0][2], "CLI-secret-key")
            self.assertNotIn("CLI-secret-prompt", original.decode())
            self.assertNotIn("CLI-secret-key", original.decode())

    def test_url_userinfo_and_fragment_rejected(self):
        for url in ("http://user:password@example.invalid/completion", "http://user@example.invalid/completion",
                    "https://example.invalid/completion#secret", "file:///tmp/request", "http://example.invalid:0"):
            with self.subTest(url=url):
                with self.assertRaises((recorder.CaptureFailure, ValueError)):
                    recorder.parse_url(url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
