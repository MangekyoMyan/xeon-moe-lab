"""Reconstruct this single local SSE capture from body bytes and application read times."""

import argparse
import bisect
import hashlib
import json
import re
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def timing_gaps(stamps):
    gaps = sorted((new - old) / 1e6 for old, new in zip(stamps, stamps[1:]))
    return {
        "gap_count": len(gaps),
        "median_ms": statistics.median(gaps) if gaps else None,
        "p95_ms": gaps[(len(gaps) * 95 + 99) // 100 - 1] if gaps else None,
        "max_ms": max(gaps, default=None),
        "zero_gap_count": gaps.count(0),
    }


def replay(directory):
    raw = (directory / "response_body.bin").read_bytes()
    reads = [json.loads(line) for line in (directory / "reads.jsonl").read_bytes().splitlines()]
    recorded_events = [json.loads(line) for line in (directory / "events.jsonl").read_bytes().splitlines()]
    summary = load(directory / "summary.json")
    metadata = load(directory / "metadata.json")
    requested = (directory / "request.requested.json").read_bytes()
    require(requested == (directory / "request.effective.json").read_bytes(), "effective request")
    require(hashlib.sha256(requested).hexdigest() == metadata["request_sha256"], "request digest")
    require(metadata["redirects"] is False and metadata["retries"] == 0, "single request")

    ends, times = [], []
    offset = 0
    for index, read in enumerate(reads, 1):
        require(read["read_index"] == index and read["body_offset_start"] == offset, "read sequence")
        require(read["read_start_perf_counter_ns"] <= read["received_perf_counter_ns"], "read time")
        require(not times or times[-1] <= read["read_start_perf_counter_ns"], "overlapping reads")
        offset += read["bytes"]
        require(offset == read["body_offset_end"], "read interval")
        ends.append(offset)
        times.append(read["received_perf_counter_ns"])
    require(reads[-1]["bytes"] == 0 and offset == len(raw), "EOF/size")
    require(summary["perf_counter_ns"]["eof"] == times[-1], "EOF timestamp")

    blocks = []
    lines = []
    consumed = 0
    for match in re.finditer(rb"([^\r\n]*)(\r\n|\r|\n)", raw):
        line = match[1].decode("utf-8")
        if match.start() == 0:
            line = line.removeprefix("\ufeff")
        consumed = match.end()
        if line:
            lines.append(line)
            continue
        data = [field.partition(":")[2].removeprefix(" ") for field in lines if field.partition(":")[0] == "data"]
        offset = match.start(2) + 1
        which = bisect.bisect_left(ends, offset)
        blocks.append((lines, "\n".join(data) if data else None, offset, which + 1, times[which]))
        lines = []
    require(consumed == len(raw) and not lines and len(blocks) == len(recorded_events), "SSE framing")

    samples = {key: [] for key in ("data", "content", "reasoning")}
    texts = {key: [] for key in ("content", "reasoning", "reasoning_content")}
    usages, reasons, done = [], [], []
    for block, event in zip(blocks, recorded_events, strict=True):
        lines, data, offset, read_index, stamp = block
        require(event["raw_lines"] == lines and event["data"] == data, "event body")
        require((event["completion_body_byte_offset"], event["read_index"], event["received_perf_counter_ns"])
                == (offset, read_index, stamp), "byte completion and read times")
        if data is None:
            continue
        samples["data"].append(stamp)
        if data.strip() == "[DONE]":
            done.append(stamp)
            continue
        if not data:
            continue
        obj = json.loads(data)
        if obj.get("usage") is not None:
            usages.append(obj["usage"])
        seen = set()
        for choice in obj.get("choices", []):
            require(choice["index"] == 0, "choice index")
            if choice.get("finish_reason") is not None:
                reasons.append(choice["finish_reason"])
            for key in texts:
                value = choice.get("delta", {}).get(key)
                if value:
                    texts[key].append(value)
                    seen.add("content" if key == "content" else "reasoning")
        for key in seen:
            samples[key].append(stamp)
    require(done == [summary["perf_counter_ns"]["done"]] and done[0] <= times[-1], "DONE")
    require(usages and usages[-1] == summary["server_usage"] and reasons[-1] == summary["finish_reason"], "final fields")
    require(summary["counts"]["reads"] == len(reads) and summary["counts"]["body_bytes"] == len(raw), "body and reads")
    gaps = {}
    for key, stamps in samples.items():
        name = "sse_data_events" if key == "data" else key + "_events"
        require(summary["counts"][name] == len(stamps), "event count")
        gap = gaps[key] = timing_gaps(stamps)
        require(all(summary["sse_" + key + "_event_gap_ms"][k] == v for k, v in gap.items()), "gap statistics")
    for key, parts in texts.items():
        require("".join(parts) == (directory / (key + ".txt")).read_text(), "response field reconstruction")
    start = summary["perf_counter_ns"]["request_start"]
    for key in ("content", "reasoning"):
        first = samples[key][0] if samples[key] else None
        require(summary["perf_counter_ns"]["first_" + key] == first, "first field receive timestamp")
        ms_label = "ttft_content_receive_ms" if key == "content" else "first_reasoning_receive_ms"
        require(summary[ms_label] == ((first - start) / 1e6 if first is not None else None), "first field elapsed")
    return {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "reads_including_EOF": len(reads), "reframed_blocks": len(blocks),
        "data_events": len(samples["data"]), "content_events": len(samples["content"]),
        "reasoning_events": len(samples["reasoning"]), "server_usage": usages[-1],
        "finish_reason": reasons[-1], "done_and_eof_observed": True,
        "first_content_receive_ms": summary["ttft_content_receive_ms"], "receive_event_gaps_ms": gaps,
        "scope": "One direct ordinary API request. Per-event clocks mark application read completion.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    print(json.dumps(replay(parser.parse_args().stream), indent=2, allow_nan=False))
