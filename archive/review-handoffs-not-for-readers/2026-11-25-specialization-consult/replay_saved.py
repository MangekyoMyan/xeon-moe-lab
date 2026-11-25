"""Inspect the saved November 24 traces. Uses only Python's standard library."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def summarize_graph(traces):
    tensor = {r["tensor"]: r for r in table(traces / "trace169-tensors.tsv")}
    splits = table(traces / "trace169-splits.tsv")
    nodes = defaultdict(list)
    for r in table(traces / "trace169-nodes.tsv"):
        nodes[r["split"]].append(r)
    per_call = defaultdict(dict)
    weights_by_layer = {}
    for split in splits:
        if split["backend"] != "CPU":
            continue
        group = nodes[split["split"]]
        gate = re.fullmatch(r"ffn_moe_gate-(\d+)", tensor[group[0]["tensor"]]["name"])
        if gate is None:
            continue
        layer = int(gate[1])
        names = [tensor[n["tensor"]]["name"] for n in group]
        expect = [f"ffn_moe_{kind}-{layer}" for kind in ("gate", "up", "swiglu_limited", "down")]
        require(names == expect, f"unexpected CPU graph {names}")
        g, u, act, down = group
        require(g["src1"] == u["src1"] and g["src2"] == u["src2"] == down["src2"], "operand IDs")
        require([g["tensor"], u["tensor"]] == [act["src0"], act["src1"]] and down["src1"] == act["tensor"], "dependencies")
        x, ids = tensor[g["src1"]], tensor[g["src2"]]
        require((x["type"], x["ne0"], x["ne1"], ids["type"], ids["ne0"], ids["ne1"]) ==
                ("f32", "5120", "1", "i32", "6", "1"), "input shapes")
        types = []
        for n, role in ((g, "gate"), (u, "up"), (down, "down")):
            w = tensor[n["src0"]]
            shape = tuple(int(w[f"ne{i}"]) for i in range(4))
            require(w["name"] == f"blk.{layer}.ffn_{role}_exps.weight", "weight name")
            require(shape == ((2304,5120,384,1) if role == "down" else (5120,2304,384,1)), "weight shape")
            types.append(w["type"])
        require(types[0] == types[1], "gate/up weight types differ")
        require(weights_by_layer.setdefault(layer, types) == types, "type changed across calls")
        require(layer not in per_call[split["call"]], "duplicate layer")
        per_call[split["call"]][layer] = types[0]
    require(len(per_call) == 8 and all(set(p) == set(range(37)) for p in per_call.values()), "trace coverage")
    return {
        "T1": len(per_call), "splits": len(splits), "CPU_gate_up_pairs_with_same_x_ids_and_dtype": 37*len(per_call),
        "per_T1_gate_up_types": dict(sorted(Counter(per_call["1"].values()).items())),
        "cpu_layers_gate_up_down_types": {str(layer): types for layer, types in sorted(weights_by_layer.items())},
    }


def expert_fields(row):
    # id, inputs, compact, rows, chunks, tiled, start ns, end ns
    return [tuple(map(int, item.split(","))) for item in row["experts"].split(";")]


def summarize_workers(folder):
    files = sorted(folder.glob("worker-*-*.tsv"))
    require(len(files) == 16, "expected 16 sampled decode calls")
    samples = []
    membership = Counter()
    differing_slots = []
    for path in files:
        layers = defaultdict(lambda: defaultdict(list))
        for row in table(path):
            match = re.fullmatch(r"ffn_moe_(gate|up|swiglu_limited|down)-(\d+)", row["name"])
            if match:
                layers[int(match[2])][match[1]].append(row)
        require(set(layers) == set(range(37)), "worker layer coverage")
        total_gap = 0
        total_after_arrival = 0
        total_before_expert = 0
        total_up_to_down = 0
        total_glu = 0
        for layer, pair in sorted(layers.items()):
            require(pair.keys() == {"gate", "up", "swiglu_limited", "down"}, "unexpected CPU split")
            gate, up = pair["gate"], pair["up"]
            glu, down = pair["swiglu_limited"], pair["down"]
            for rows in (gate, up, down, glu):
                require(len(rows) == 24 and {int(r["ith"]) for r in rows} == set(range(24)), "worker coverage")
                op = "GLU" if rows is glu else "MUL_MAT_ID"
                require(all(int(r["nth"]) == 24 and r["op"] == op for r in rows), "operator")
            def experts(rows):
                all_lists = [expert_fields(r) for r in rows]
                reference = [(e[0], e[2]) for e in all_lists[0]]
                require(len(set(k for k, _ in reference)) == 6 and len(reference) == 6, "expert count")
                require(all([(e[0],e[2]) for e in lst] == reference for lst in all_lists), "per-worker membership")
                return reference, all_lists
            gm, gate_experts = experts(gate)
            um, up_experts = experts(up)
            dm, down_experts = experts(down)
            require([k for k, _ in gm] == [k for k, _ in um] == [k for k, _ in dm], "per-projection IDs")
            diff = sum(a != b for (_, a), (_, b) in zip(gm, um))
            membership["pairs"] += 1
            membership["pairs_with_different_compact_membership"] += int(diff > 0)
            membership["expert_slots"] += len(gm)
            membership["slots_with_different_compact_membership"] += diff
            differing_slots.extend({"file": path.name, "layer": layer, "expert": expert,
                                    "gate_compact": a, "up_compact": b}
                                   for (expert, a), (_, b) in zip(gm, um) if a != b)
            # All timestamps below come from one CLOCK_MONOTONIC capture and one CPU subgraph.
            end_gate_expert = max(e[7] for group in gate_experts for e in group)
            start_up_expert = min(e[6] for group in up_experts for e in group)
            last_gate_arrival = max(int(r["finish"]) for r in gate)
            first_up_node = min(int(r["start"]) for r in up)
            require(end_gate_expert <= last_gate_arrival <= first_up_node <= start_up_expert, "barrier timeline")
            total_gap += start_up_expert - end_gate_expert
            total_after_arrival += first_up_node - last_gate_arrival
            total_before_expert += start_up_expert - first_up_node
            end_up_expert = max(e[7] for group in up_experts for e in group)
            start_glu = min(int(r["start"]) for r in glu)
            end_glu = max(int(r["finish"]) for r in glu)
            start_down_expert = min(e[6] for group in down_experts for e in group)
            require(end_up_expert <= start_glu <= end_glu <= start_down_expert, "activation timeline")
            total_up_to_down += start_down_expert - end_up_expert
            total_glu += end_glu - start_glu
        samples.append({"file": path.name, "gate_last_expert_to_up_first_expert_ns": total_gap,
                        "gate_last_node_finish_to_up_first_node_start_ns": total_after_arrival,
                        "up_first_node_start_to_up_first_expert_ns": total_before_expert,
                        "up_last_expert_to_down_first_expert_ns": total_up_to_down,
                        "glu_earliest_node_start_to_latest_finish_ns": total_glu})
    keys = list(samples[0])[1:]
    return {"T1": len(samples), "membership": dict(membership), "different_compact_slots": differing_slots,
            "per_T1_ns": samples,
            "ms_per_T1_over_37_layers": {k.removesuffix("_ns"): {
                "min": min(r[k] for r in samples)/1e6, "median": statistics.median(r[k] for r in samples)/1e6,
                "max": max(r[k] for r in samples)/1e6} for k in keys}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, default=HERE.parent)
    p.add_argument("--check", action="store_true", help="compare with the accompanying result")
    args = p.parse_args()
    root = args.archive_root
    trace = root/"2026-11-24-normal24-timeline"/"runs"/"observe"
    workers = root/"2026-11-24-prefill-arithmetic"/"runs"/"worker-trace"
    files = [trace/f"trace169-{part}.tsv" for part in ("nodes", "splits", "tensors")] + sorted(workers.glob("worker-*-*.tsv"))
    digests = {}
    for path in files:
        with path.open("rb") as stream:
            digests[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"input_SHA256": digests, "cupti_graph_structure": summarize_graph(trace),
              "worker_trace": summarize_workers(workers)}
    if args.check:
        require(result == json.loads((HERE/"SAVED_TRACE_RESULT.json").read_text()), "result differs")
        print(f"Matched {len(files)} archived inputs and both analyses.")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
