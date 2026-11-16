#!/usr/bin/env python3
"""Validate a proposed T=1 MoE trace and summarize observed intervals.

This is an offline analysis prototype, not engine instrumentation. It never
opens a model, runs CUDA, sends a request, or predicts a placement speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

HOST_INTERVALS = ("cpu_setup", "gpu_submit_host", "cpu_compute",
                  "cpu_output_copy", "gpu_join", "scatter")
GPU_INTERVALS = ("h2d", "graph", "d2h")


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, low=0, high=None):
    return isinstance(value, int) and not isinstance(value, bool) and value >= low and (high is None or value <= high)


def span(value, label):
    require(isinstance(value, list) and len(value) == 2 and all(integer(x) for x in value), label + ": expected [start_ns,end_ns]")
    require(value[1] >= value[0], label + ": reversed host interval")
    return value


def key(value):
    require(isinstance(value, dict), "Pass/record must be an object")
    require(isinstance(value.get("request"), str) and bool(value["request"]), "Missing request identity")
    require(integer(value.get("pass")), "Invalid model-pass identity")
    require(isinstance(value.get("phase"), str) and bool(value["phase"]), "Missing explicit phase")
    return value["request"], value["pass"], value["phase"]


def descriptive(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"n": len(values), "mean": math.fsum(values) / len(values),
            "p50": percentile(.5), "p95": percentile(.95), "max": values[-1]}


def analyze(data):
    require(isinstance(data, dict) and data.get("schema_version") == 1, "Unsupported schema")
    manifest = data.get("manifest")
    require(isinstance(manifest, dict), "Missing expected manifest")
    require(manifest.get("complete") is True and manifest.get("dropped_records") == 0,
            "Incomplete/dropped trace cannot pass")
    require(integer(manifest.get("dropped_records")), "Invalid drop count")
    require(manifest.get("layers") == 40 and manifest.get("top_k") == 6 and manifest.get("experts") == 384,
            "This prototype requires the pinned 40-layer/6-slot/384-expert geometry")
    require(manifest.get("host_clock") in ("CLOCK_MONOTONIC_RAW", "CLOCK_MONOTONIC"), "Unknown host clock")
    require(isinstance(manifest.get("clock_domain"), str) and bool(manifest["clock_domain"]), "Missing clock domain")
    require(re.fullmatch(r"[0-9a-f]{64}", manifest.get("config_sha256", "")) is not None, "Missing exact config SHA")
    trace_kind = manifest.get("trace_kind", "route_census")
    require(trace_kind in ("route_census", "hot_latency"), "Unknown trace kind")
    if trace_kind == "route_census":
        expected_layers = list(range(40))
        require(manifest.get("expected_layers", expected_layers) == expected_layers,
                "Route census requires every layer, 0 through 39")
    else:
        expected_layers = manifest.get("expected_layers")
        require(isinstance(expected_layers, list) and bool(expected_layers)
                and all(integer(v, high=39) for v in expected_layers)
                and expected_layers == sorted(set(expected_layers)),
                "Hot latency requires independently declared sorted unique instrumented layers")
    expected = manifest.get("expected_passes")
    require(isinstance(expected, list) and bool(expected), "Expected passes must be a nonempty independent inventory")
    expected_keys = [key(row) for row in expected]
    require(len(set(expected_keys)) == len(expected_keys), "Duplicate expected pass")
    inventory = {(k, layer) for k in expected_keys for layer in expected_layers}
    records = data.get("records")
    require(isinstance(records, list) and len(records) == len(inventory), "Record count differs from independent inventory")
    seen = set(); by_layer = defaultdict(list); host = defaultdict(list); gpu = defaultdict(list)
    query_counts = Counter(); phases = Counter(); host_wall_sum = 0; join_sum = 0
    for row in records:
        k = key(row)
        require(integer(row.get("layer"), high=39), "Invalid layer")
        identity = k, row["layer"]
        require(identity in inventory and identity not in seen, "Unexpected or duplicate layer/pass")
        seen.add(identity)
        require(row.get("n_tokens") == 1 and not isinstance(row.get("n_tokens"), bool), "Only explicitly T=1 records supported")
        require(row.get("clock_domain") == manifest["clock_domain"], "Mixed host clock domains")
        route = row.get("route")
        require(isinstance(route, list) and len(route) == 6 and all(integer(v, high=383) for v in route)
                and len(set(route)) == 6, "Route must preserve six unique ordered global IDs")
        backend = row.get("backend")
        require(backend in ("cpu", "gpu", "hot"), "Unknown FFN backend")
        require(trace_kind != "hot_latency" or backend == "hot", "Hot-only trace contains a different backend")
        mask = row.get("gpu_hit_mask")
        require(integer(mask, high=63), "Invalid current-placement GPU slot mask")
        require(backend != "cpu" or mask == 0, "CPU-only row has GPU hits")
        require(backend != "gpu" or mask == 63, "Whole-layer GPU row must have six GPU slots")
        h = mask.bit_count()
        phases[row["phase"]] += 1
        by_layer[row["layer"]].append((backend, h, tuple(route)))
        times = row.get("host_ns")
        require(trace_kind != "hot_latency" or times is not None,
                "Hot latency record is missing host timestamps")
        if times is None:
            require(row.get("gpu_stream_ms") is None, "GPU intervals without a timed host record")
            continue
        require(backend == "hot", "This MVP accepts detailed host intervals only for the hot dispatcher")
        require(isinstance(times, dict), "Invalid host_ns")
        wall = span(times.get("wall"), "wall")
        host_wall_sum += wall[1] - wall[0]
        host["wall"].append((wall[1] - wall[0]) / 1e6)
        required = {"cpu_setup", "cpu_compute", "cpu_output_copy"} if h < 6 else set()
        if h > 0:
            required |= {"gpu_submit_host", "gpu_join", "scatter"}
        active = []
        for name in HOST_INTERVALS:
            value = times.get(name)
            if name not in required:
                require(value is None, "Unused branch interval must be null: " + name)
                continue
            a, b = span(value, name)
            require(wall[0] <= a <= b <= wall[1], "Host interval outside wall: " + name)
            active.append((a, b, name))
            host[name].append((b - a) / 1e6)
            if name == "gpu_join":
                join_sum += b - a
        require(all(left[1] <= right[0] for left, right in zip(active, active[1:])), "Overlapping/reordered host intervals would double count")
        order = [name for _, _, name in active]
        expected_order = [name for name in HOST_INTERVALS if name in required]
        require(order == expected_order, "Host sequence differs from current dispatcher")
        covered = sum(b - a for a, b, _ in active)
        host["unassigned_host"].append((wall[1] - wall[0] - covered) / 1e6)
        observation = row.get("gpu_done_query", "not_measured")
        require(observation in ("ready", "not_ready", "not_measured", "not_applicable"), "Invalid event query status")
        if observation in ("ready", "not_ready"):
            require(0 < h < 6, "CPU-end query requires both CPU and GPU branches")
            query_time = row.get("gpu_done_query_host_ns")
            require(integer(query_time), "Missing query observation timestamp")
            require(times["cpu_output_copy"][1] <= query_time <= times["gpu_join"][0], "Query is outside its declared observation window")
        query_counts[observation] += 1
        streams = row.get("gpu_stream_ms")
        if streams is not None:
            require(h > 0 and isinstance(streams, dict), "GPU stream data without a GPU branch")
            require(row.get("gpu_events_completed") is True, "Event durations require completed recorded events")
            for name in GPU_INTERVALS:
                value = streams.get(name)
                require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
                        "Invalid GPU elapsed interval: " + name)
                gpu[name].append(value)
    require(seen == inventory, "Missing expected layer/pass")
    layers = []
    for layer, values in sorted(by_layer.items()):
        all_h = Counter(h for _, h, _ in values)
        hot_h = Counter(h for backend, h, _ in values if backend == "hot")
        layers.append({"layer": layer, "routes": len(values), "ordered_route_tuples": len({route for _, _, route in values}),
                       "current_gpu_slot_H_histogram": [all_h[h] for h in range(7)],
                       "hot_dispatcher_H_histogram": [hot_h[h] for h in range(7)]})
    return {"validated": True, "trace_kind": trace_kind, "expected_layers": expected_layers,
            "is_complete_40_layer_census": trace_kind == "route_census",
            "passes": len(expected), "records": len(records), "phase_record_counts": dict(phases),
            "layers": layers, "hot_host_interval_ms": {name: descriptive(values) for name, values in sorted(host.items())},
            "gpu_stream_elapsed_ms": {name: descriptive(values) for name, values in sorted(gpu.items())},
            "gpu_query_observations": dict(query_counts),
            "exposed_join_fraction_of_observed_hot_wall": join_sum / host_wall_sum if host_wall_sum else None,
            "interpretation": [
                "Host intervals are nonoverlapping observed intervals in one clock domain.",
                "GPU event elapsed times are not added to host CPU, submit or join times.",
                "Ready means complete at the query observation, not an exact GPU finish timestamp.",
                "Join fraction is descriptive for this execution, not an attainable speedup bound.",
                "Missing/dropped passes fail validation; this does not prove engine instrumentation correctness.",
                "Passes are explicit T=1 graph executions, not automatically generated tokens.",
                "The script does not infer new-placement latency from marginal hit frequencies."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be new")
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    result = analyze(json.loads(args.input.read_text(), parse_constant=invalid))
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False); out.write("\n")
    print(json.dumps({"output": str(args.output), "validated": result["validated"], "records": result["records"]}))


if __name__ == "__main__":
    main()
