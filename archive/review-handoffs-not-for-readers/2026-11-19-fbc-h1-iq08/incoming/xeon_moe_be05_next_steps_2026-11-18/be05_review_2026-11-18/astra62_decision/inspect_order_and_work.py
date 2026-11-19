"""Read the saved timing runs and calculate process-order ranges and static work."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe_work(block):
    """Counts for this T=1 graph, assuming the unmodified CPU MUL_MAT_ID source."""
    result = {"cell": block["cell"], "compact_allocation_bytes": block["compact_bytes"]}
    cached_bytes = 0
    native_bytes = 0
    dot_rows = 0
    compact_rows = 0
    weight_blocks = 0
    compact_weight_blocks = 0
    folded_span = []
    quantized_blocks = 0
    call_counts = {12: Counter(), 24: Counter()}
    routes = block["ids"]
    assert len(set(routes)) == len(routes) == 6

    for p, weight in enumerate(block["weights"]):
        k, m, experts, tokens = weight["ne"]
        assert experts == 384 and tokens == 1
        assert [v["expert"] for v in weight["selected_native"]] == routes
        cached = set(weight["compact_experts"])
        native_bytes += sum(s["bytes"] for s in weight["selected_native"] if s["expert"] not in cached)
        dot_rows += len(routes) * m
        if weight["type"] == "mxfp4":
            assert not cached
            folded_span.append(0)
            continue
        assert weight["type"] in {"iq2_s", "iq3_xxs", "iq3_s"}
        assert k % 256 == 0
        nblocks = k // 256
        rows_input = 1 if p < 2 else len(routes)
        quantized_blocks += rows_input * nblocks
        compact_rows += len(cached) * m
        weight_blocks += len(routes) * m * nblocks
        compact_weight_blocks += len(cached) * m * nblocks
        cached_bytes += len(cached) * m * nblocks * 144
        folded_span.append(rows_input * nblocks * 400 if cached else 0)
        for workers, counts in call_counts.items():
            for worker in range(workers):
                block_count = (worker + 1) * nblocks // workers - worker * nblocks // workers
                counts[block_count] += rows_input

    assert cached_bytes == block["compact_bytes"]
    result.update(
        logical_dot_rows=dot_rows,
        compact_dot_rows=compact_rows,
        logical_IQ_weight_blocks=weight_blocks,
        logical_compact_IQ_weight_blocks=compact_weight_blocks,
        logical_Q8_K_blocks=quantized_blocks,
        Q8_K_calls_by_threads_and_blocks_per_call=call_counts,
        logical_bytes_in_selected_compact_plus_uncached_weights=cached_bytes + native_bytes,
        selected_uncached_native_bytes=native_bytes,
        expected_08_folded_strided_span_by_projection_bytes=folded_span,
    )
    return result


def audit(source):
    q8 = source / "q8-avx1"
    design = read_json(q8 / "FFN_DESIGN.json")
    stages = [s for s in design["stages"] if s["stage"] == "timing"]
    rows = {}
    arm_orders = []
    pair_indices = [(0, 1), (3, 2), (5, 4), (6, 7)]
    geometry = {}
    for stage in stages:
        data = read_json(q8 / "runs-v2" / stage["name"] / "RUN.json")
        assert all(data["spec"][key] == value for key, value in stage.items())
        arm_orders.append((stage["mode"], stage["cells"], stage["threads"]))
        key_order = [(c, t) for c in stage["cells"] for t in stage["threads"]]
        assert [(b["cell"], b["threads"]) for b in data["blocks"]] == key_order
        for b in data["blocks"]:
            work = describe_work(b)
            assert geometry.setdefault(b["cell"], work) == work
            samples = b["samples"]
            assert len(samples) == stage["samples"] == 5
            assert all(s["end_ns"] > s["begin_ns"] and s["calls"] == 8 for s in samples)
            process_median_ms = median((s["end_ns"] - s["begin_ns"]) / (8 * 1e6) for s in samples)
            rows.setdefault((b["cell"], b["threads"]), []).append(process_median_ms)

    schedules = {"forward": ([1, 5, 6], [12, 24]), "reverse": ([6, 5, 1], [24, 12])}
    order_indices = {
        key: [i for i, (_, cells, threads) in enumerate(arm_orders) if (cells, threads) == order]
        for key, order in schedules.items()
    }
    assert all(len(indices) == 4 for indices in order_indices.values())
    for control, candidate in pair_indices:
        assert arm_orders[control][0] == "control" and arm_orders[candidate][0] == "candidate"
        assert arm_orders[control][1:] == arm_orders[candidate][1:]

    comparison = []
    for (cell, threads), values in sorted(rows.items()):
        ranges = {}
        for order, indices in order_indices.items():
            selected = [values[i] for i in indices]
            ranges[order] = [min(selected), max(selected)]
        ratios = [values[candidate] / values[control] for control, candidate in pair_indices]
        comparison.append({
            "cell": cell, "threads": threads, "process_medians_ms": values,
            "range_of_process_medians_ms_by_schedule": ranges,
            "pair_ratios_candidate_over_control": ratios, "median_of_four_ratios": median(ratios),
        })
    return {
        "commit": "be05f5fb10438e83ff0201deef2ddfd21bf4a111",
        "timing_stages": [s["name"] for s in stages], "order_process_indices": order_indices,
        "control_candidate_process_indices": pair_indices,
        "observed": comparison, "computed_from_shapes_and_dispatch_source": list(geometry.values()),
        "scope": "Order comparisons are descriptive. Work, byte spans and call counts are static logical quantities, not traffic, timing or path-counter observations.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted 2026-11-18-q8-k-avx1 directory")
    print(json.dumps(audit(parser.parse_args().source), indent=2, ensure_ascii=False))
