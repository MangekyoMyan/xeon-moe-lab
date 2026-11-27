"""Recompute the decision-relevant numbers from the earlier saved experiments."""
import argparse
import csv
import functools
import hashlib
import json
import math
import operator
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(singletons, saved_trace):
    routes = singletons / "ROUTING.json"
    graph = singletons / "LARGE_RESULT.json"
    run = singletons / "runs/live-large-1024/RUN.json"
    scores = singletons / "scores/live-large-1024/PY1024-P.tsv"
    route_cases = read(routes)
    counts = {}
    for case in ("JA512", "PY512"):
        counts[case] = {}
        for row in route_cases[case]:
            groups = row["groups"]
            counts[case][str(row["layer"])] = {
                "extra_original_weight_B_at_2": sum(g["native_B"] for g in groups if g["inputs"] == 2),
                "cpu_slots_le2": sum(g["inputs"] for g in groups if g["inputs"] <= 2),
                "singletons": sum(g["inputs"] == 1 for g in groups),
            }
    types = read(saved_trace)["cupti_graph_structure"]["cpu_layers_gate_up_down_types"]
    patterns = {}
    for layer in range(37):
        pattern = "/".join("MX" if x == "mxfp4" else "IQ" for x in types[str(layer)])
        assert all(x == "mxfp4" or x.startswith("iq") for x in types[str(layer)])
        patterns.setdefault(pattern, []).append(layer)
    fixtures = read(graph)["history"]
    work = {}
    for record in fixtures["P_workspaces"]:
        for layer, v in record["layers"].items():
            total = work.setdefault(layer, {"chunks": 0, "shapes": 0, "max_layer_scratch_B": 0})
            total["chunks"] += v["chunks"]
            total["shapes"] += v["graph_shapes_allocated"]
            total["max_layer_scratch_B"] = max(total["max_layer_scratch_B"], v["max_private_gpu_scratch_B"])
    with scores.open() as file:
        all_rows = list(csv.DictReader(file, delimiter="\t"))
    rows = all_rows[:16]  # final distribution has no source continuation token
    delta = [float(r["B_target_nll"]) - float(r["A_target_nll"]) for r in rows]
    nll = {"positions": len(rows), "top1_changed_of_17": sum(r["A_top"] != r["B_top"] for r in all_rows),
           "source_hits_A_B": [sum(r[x+"_top"] == r["target"] for r in rows) for x in ("A", "B")],
           "mean_delta": math.fsum(delta)/len(rows), "row0_contribution_to_mean": delta[0]/len(rows),
           "row0_fraction": delta[0]/math.fsum(delta), "excluding_row0_mean": math.fsum(delta[1:])/(len(rows)-1),
           "row6_target_probability_A_B": [math.exp(-float(rows[6][x+"_target_nll"])) for x in ("A", "B")]}
    timing = []
    for record in read(run)["conversation"]:
        chunks = record["prefix_chunks"]
        durations = [x["end"] - x["start"] for x in chunks]
        timing.append({"pair": record["pair"], "arm": record["arm"], "integer_ns": sum(durations),
                       "integer_ns_converted_to_s": sum(durations)/1e9,
                       "left_add_float_s": functools.reduce(operator.add, (x/1e9 for x in durations))})
    return {"review_sha256": "2751867997629904a495c5fe65c489556521304a043b3924b92c445574e2910d",
            "source_sha256": {p.name: digest(p) for p in (routes, saved_trace, graph, run, scores)},
            "two_hit_estimates_from_recorded_routes": counts,
            "host_layers_by_gate_up_down": patterns, "workspaces": work, "PY1024": nll,
            "prior_prefix_sum": timing, "prior_75_s": {k: fixtures["suffix"][k] for k in ("O_s", "P_s")}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prior_singleton_study", type=Path)
    p.add_argument("saved_trace_json", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    report = check(args.prior_singleton_study, args.saved_trace_json)
    output = Path(__file__).with_name("PRIOR_RECHECK.json")
    if args.check:
        assert read(output) == report
    else:
        output.write_text(json.dumps(report, indent=2) + "\n")
