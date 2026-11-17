#!/usr/bin/env python3
"""Offline analysis of immutable published route records. No engine is executed.

This uses training first5 only. One conditional alternative minimizes active
layers while retaining >=90% of the ORIGINAL proposal's aggregate inclusion
COUNT under the same native-byte budget. It does not retain the same exact hits,
optimize wall time, claim generalization, or admit a GPU allocation.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PIN = "d98a17c56fadc92cf79e21ecbe80732dedeed899"
REL = "archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace"
RUN = "pilot-capture/runs/20261117T035402Z-U1024-d22b3e04"
CASE_ORDER = ["text_cache_budget_ja", "text_cache_budget_en", "mixed_build_log_ja_en_log",
              "code_stable_grouping_python", "code_stable_grouping_cpp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    n = args.repo / REL
    sources = {}

    def load(rel):
        p = n / rel
        sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())

    cases = {c: load(f"{RUN}/{c}.json") for c in CASE_ORDER}
    inv = load("runs/20261117T031214Z-U1024-93e7fe85/RUN.json")["native_expert_inventory"]
    original_report = load("TRAIN5_PROPOSAL.json")
    costs = {x["layer"]: x["native_triplet_bytes"] for x in inv}
    weights = {x["layer"]: [p["type"] for p in x["projections"]] for x in inv}
    frequencies = Counter()
    records = []
    for case, data in cases.items():
        rr = data["T1_trace"]["records"]
        assert len(rr) == 64 * 37
        keys = set()
        for r in rr:
            key = (r["decode_call"], r["layer"])
            assert key not in keys and 0 <= r["layer"] < 37 and r["flags"] == 0
            assert len(r["ids"]) == len(set(r["ids"])) == 6
            keys.add(key)
            for e in r["ids"]:
                assert 0 <= e < 384
                frequencies[(r["layer"], e)] += 1
            records.append((case, r))
        assert keys == {(i, l) for i in range(64) for l in range(37)}

    budget = original_report["candidate"]["budget_bytes"]
    ranked = sorted(frequencies, key=lambda x: (-Fraction(frequencies[x], costs[x[0]]), *x))
    original, used = set(), 0
    for l, e in ranked:
        if used + costs[l] <= budget:
            original.add((l, e)); used += costs[l]
    given = {(v["layer"], v["expert"]) for v in original_report["candidate"]["selected"]}
    assert original == given and used == 475453440 and len(original) == 35

    def describe(selected):
        active = sorted({l for l, _ in selected})
        hist, hist_active = [0]*7, [0]*7
        by_case = {c: {"all_H": [0]*7, "active_H": [0]*7} for c in CASE_ORDER}
        by_layer = {l: {"H": [0]*7, "experts": sorted(e for ll,e in selected if ll == l),
                        "projection_types": weights[l]} for l in active}
        for case, r in records:
            l = r["layer"]
            h = sum((l, e) in selected for e in r["ids"])
            hist[h] += 1; by_case[case]["all_H"][h] += 1
            if l in by_layer:
                hist_active[h] += 1; by_case[case]["active_H"][h] += 1
                by_layer[l]["H"][h] += 1
        hits = sum(h*c for h,c in enumerate(hist))
        positives = sum(hist[1:])
        for row in by_case.values():
            row["inclusion_count"] = sum(h*c for h,c in enumerate(row["all_H"]))
            row["slot_hit_fraction"] = row["inclusion_count"] / (64*37*6)
        return {"selected": [{"layer":l,"expert":e,"count":frequencies[l,e],"native_bytes":costs[l]}
                             for l,e in sorted(selected)],
                "active_layers":active, "active_layer_count":len(active),
                "expert_count":len(selected),"native_payload_bytes":sum(costs[l] for l,e in selected),
                "gpu_graph_count_for_old_per_H_builder":sum(min(6,len(x["experts"])) for x in by_layer.values()),
                "inclusion_count":hits,"all_H":hist,"active_H":hist_active,
                "slot_hit_fraction":hits/(len(records)*6),
                "positive_episodes":positives,"positive_H1_fraction":hist[1]/positives,
                "active_miss_to_hit_episode_ratio":hist_active[0]/positives,
                "all_miss_to_hit_episode_ratio":hist[0]/positives,
                "per_case":by_case,"per_layer":by_layer}

    original_data = describe(original)
    target = (9*original_data["inclusion_count"] + 9)//10  # ceil(0.90 * original)
    unit = math.gcd(*[costs[l] for l in range(37)])
    assert unit == 92160 and budget % unit == 0
    B, L = budget//unit, 37
    neg = -10**8
    dp = np.full((L+1,B+1),neg,np.int32); dp[0,0] = 0
    choices = []
    options_by_layer = {}
    # Exact multiple-choice knapsack. Top n expert frequencies in one layer
    # dominate any other n within that layer (same per-expert byte cost).
    for layer in range(L):
        expert_order = sorted((e for l,e in frequencies if l == layer),
                              key=lambda e:(-frequencies[layer,e], e))
        cu = costs[layer]//unit
        limit = min(B//cu,len(expert_order))
        expert_order = expert_order[:limit]
        options_by_layer[layer] = expert_order
        values = np.cumsum([frequencies[layer,e] for e in expert_order],dtype=np.int32)
        nxt = dp.copy()
        choice = np.zeros(dp.shape,np.uint8)
        # strict > preserves the skip-layer option on a tie, then smaller n.
        for num, value in enumerate(values,1):
            w = num*cu
            source = dp[:layer+1,:B+1-w]
            candidate = source + value
            dest = nxt[1:layer+2,w:]
            replace = (source >= 0) & (candidate > dest)
            dest[replace] = candidate[replace]
            ch = choice[1:layer+2,w:]
            ch[replace] = num
        choices.append(choice)
        dp = nxt
    frontier = []
    chosen_l = None
    for num_layers in range(1,L+1):
        val = int(dp[num_layers].max())
        if val < 0:
            continue
        b = int(np.flatnonzero(dp[num_layers] == val)[0])
        frontier.append({"active_layer_count":num_layers,"maximum_inclusion_count":val,
                         "least_native_bytes_at_maximum":b*unit})
        if chosen_l is None and val >= target:
            chosen_l = num_layers
    assert chosen_l is not None
    chosen_b = int(np.argmax(dp[chosen_l]))
    chosen_score = int(dp[chosen_l,chosen_b])
    alternative = set()
    k,b = chosen_l,chosen_b
    for layer in reversed(range(L)):
        num = int(choices[layer][k,b])
        if num:
            alternative.update((layer,e) for e in options_by_layer[layer][:num])
            b -= num*(costs[layer]//unit); k -= 1
    assert (k,b) == (0,0)
    alt_data = describe(alternative)
    assert alt_data["inclusion_count"] == chosen_score >= target
    assert alt_data["active_layer_count"] == chosen_l
    assert alt_data["native_payload_bytes"] == chosen_b*unit <= budget
    assert all(x["maximum_inclusion_count"] < target for x in frontier if x["active_layer_count"] < chosen_l)

    # Eight deterministic actual-route cells, covering original H0/H1/H2/H3
    # and IQ2_S/IQ3_XXS/IQ3_S/MXFP4 triplet patterns. Raw contains no activations.
    requirements = [(7,0),(7,1),(7,2),(7,3),(35,3),(13,1),(28,1),(20,2)]
    cells = []
    for layer,h in requirements:
        found = next((case,r) for case,r in records if r["layer"] == layer and
                     sum((layer,e) in original for e in r["ids"]) == h)
        case,r = found
        cells.append({"cell":len(cells),"case_id":case,"decode_call":r["decode_call"],
                      "position":r["position"],"observed_generation":r["generation"],
                      "context_label_in_source_session":r["context"],"layer":layer,
                      "ordered_ids":r["ids"],"H_under_original_candidate":h,
                      "gpu_slots":[i for i,e in enumerate(r["ids"]) if (layer,e) in original],
                      "projection_types":weights[layer],
                      "source":f"{RUN}/{case}.json","source_record_ordinal":r["ordinal"],
                      "activation_status":"not present in repository; capture in dedicated baseline replay"})

    result = {"schema":"independent-gpu-workspace-bank-tradeoff-v1","source_commit":PIN,
              "source_root":REL,"input_sha256":sources,"training_case_order":CASE_ORDER,
              "records":len(records),"slots":6*len(records),"distinct_pairs":len(frequencies),
              "original_reproduced":True,"original":original_data,
              "conditional_alternative":{
                  "status":"offline diagnostic option only; not benchmarked, allocated, or admitted",
                  "selection_rule":"minimize active layers subject to native bytes <=480706560 and aggregate training inclusion count >=ceil(0.9*4176); then maximize aggregate count and minimize bytes",
                  "why_90_percent":"declared engineering retention target, not a measured time penalty or quality criterion",
                  "target_inclusion_count":target,"byte_unit":unit,"capacity_units":B,
                  "same_exact_hits_retained":False,"frontier":frontier,**alt_data},
              "representative_capture_cells":cells,
              "measurement_warning":"no GPU-ready or saved-time estimates; counts cannot establish parallel speedup, capacity admission, quality, or language causality"}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"BANK_TRADEOFF_RECOMPUTED.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (args.out/"CAPTURE_CELLS.json").write_text(json.dumps({"source_commit":PIN,"candidate_plan_sha256":original_report["candidate"]["plan_sha256"],"cells":cells},ensure_ascii=False,indent=2)+"\n")
    # Keep the exact published parser's header/types/counts/40-layer grammar.
    plan = f"XEON_HOT_FFN_PLAN_V1 40 384 5120 2304 {len(alternative)} {alt_data['native_payload_bytes']}\n"
    for layer in range(40):
        ids = sorted(e for l,e in alternative if l == layer)
        row = [str(layer),*[x.upper() for x in weights[layer]],str(len(ids)),*[str(e) for e in ids]]
        plan += " ".join(row)+"\n"
    (args.out/"CONDITIONAL_CONCENTRATED_CANDIDATE.plan").write_text(plan)
    print(json.dumps({"original":{k:original_data[k] for k in ['expert_count','active_layer_count','native_payload_bytes','inclusion_count','all_H','active_H']},
                      "alternative":{k:alt_data[k] for k in ['expert_count','active_layer_count','native_payload_bytes','inclusion_count','all_H','active_H']},
                      "target":target,"frontier":frontier[:chosen_l+1],"capture_cells":cells},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
