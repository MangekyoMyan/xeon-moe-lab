"""Reconstruct the measured layer timeline using ggml and CUPTI evidence together."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(directory, name):
    with (directory / f"trace169-{name}.tsv").open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def by_integer(items, name):
    return {int(row[name]): row for row in items}


def group(items, name):
    result = defaultdict(list)
    for row in items:
        result[int(row[name])].append(row)
    return result


def exact(indices, label):
    assert len(indices) == 1, (label, indices)
    return indices[0]


def shape(tensor, dims, typ):
    return tensor["type"] == typ and tuple(int(tensor["ne" + str(i)]) for i in range(4)) == dims


def launch(kernel, grid, block):
    return all(int(kernel["grid_" + a]) == n for a, n in zip("xyz", grid)) and all(
        int(kernel["block_" + a]) == n for a, n in zip("xyz", block))


def q8_1(kernel, x):
    return "quantize_q8_1" in kernel["name"] and launch(kernel, (x, 1, 1), (256, 1, 1))


def mmvq(kernel, typ, fusion, output_rows):
    # The CUDA12.9 activity normally emits an Itanium symbol. Accept its C++
    # spelling as well, with the same template arguments. On Volta Q6_K uses
    # two warps and Q8_0 uses four. small_k changes rows per CTA to nwarps.
    name = kernel["name"]
    mangled = re.search(r"mul_mat_vec_qIL9ggml_type(\d+)ELi1ELb([01])ELb([01])ELb0E", name)
    demangled = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), 1, (true|false), (true|false), false>", name)
    m = mangled or demangled
    if not m or int(m[1]) != typ or (m[2] in ("1", "true")) != fusion:
        return False
    warps = 2 if typ == 14 else 4
    rpb = warps if m[3] in ("1", "true") else 1
    return launch(kernel, ((output_rows+rpb-1)//rpb, 1, 1), (32, warps, 1))


def clamp(kernel):
    return ("swiglu_clamp_kernelIfE" in kernel["name"] or "swiglu_clamp_kernel<float>" in kernel["name"]) and launch(
        kernel, (9, 1, 1), (256, 1, 1))


def signature(items, i):
    k = items[i:i+7]
    return len(k) == 7 and q8_1(k[0], 20) and mmvq(k[1], 14, False, 2304) and q8_1(k[2], 20) and \
        mmvq(k[3], 8, False, 2304) and clamp(k[4]) and q8_1(k[5], 10) and mmvq(k[6], 14, True, 5120)


def analyze(run):
    receipt = json.loads((run / "trace169-summary.json").read_text())
    assert not any(receipt[k] for k in ("unknown_kernel_correlation", "foreign_thread_calls", "dropped_inside",
                                          "unknown_records", "errors"))
    calls = by_integer(rows(run, "calls"), "call")
    splits = by_integer(rows(run, "splits"), "split")
    tensors = by_integer(rows(run, "tensors"), "tensor")
    nodes = group(rows(run, "nodes"), "split")
    copied = group(rows(run, "copies"), "call")
    assert len(calls) == 8
    assert {(int(c["case_id"]), int(c["call_id"])) for c in calls.values()} == {
        (case, i) for case in (0, 1) for i in range(1, 5)}
    correlation = {}
    for r in rows(run, "correlations"):
        key = int(r["correlation"])
        assert correlation.setdefault(key, int(r["external_id"])) == int(r["external_id"])
    gpu = defaultdict(list)
    for k in rows(run, "kernels"):
        gpu[correlation[int(k["correlation"])]].append(k)
    for k in gpu.values():
        k.sort(key=lambda x: int(x["start_ns"]))
    result = []
    for call_id, call in calls.items():
        local = sorted((x for x in splits.values() if int(x["call"]) == call_id), key=lambda x: int(x["ordinal"]))
        assert len(local) == int(call["n_splits"])
        layers = []
        for i, cpu in enumerate(local[:-1]):
            if cpu["backend"] != "CPU":
                continue
            c_nodes = nodes[int(cpu["split"])]
            c_names = [tensors[int(n["tensor"])]["name"] for n in c_nodes]
            gate_pos = [p for p, name in enumerate(c_names) if re.fullmatch(r"ffn_moe_gate-\d+", name)]
            if not gate_pos:
                continue
            j = exact(gate_pos, "routed CPU gate")
            layer = int(c_names[j].split("-")[-1])
            layers.append(layer)
            label = f"case{call['case_id']}.call{call['call_id']}.L{layer}"
            assert c_names[j:j+4] == [f"{n}-{layer}" for n in (
                "ffn_moe_gate", "ffn_moe_up", "ffn_moe_swiglu_limited", "ffn_moe_down")], label
            assert tensors[int(c_nodes[j]["tensor"])]["op"] == "MUL_MAT_ID", label
            next_gpu = local[i+1]
            assert next_gpu["backend"] == "CUDA0", label
            gid = int(next_gpu["split"])
            g_nodes = nodes[gid]
            ts = [tensors[int(n["tensor"])] for n in g_nodes]
            at = exact([p for p, t in enumerate(ts) if t["name"] == f"ffn_gate-{layer}"], label)
            assert [t["name"] for t in ts[at:at+5]] == [f"{n}-{layer}" for n in (
                "ffn_gate", "ffn_up", "ffn_swiglu_limited", "ffn_shexp", "ffn_out")], label
            assert [t["op"] for t in ts[at:at+5]] == ["MUL_MAT", "MUL_MAT", "GLU", "MUL_MAT", "ADD"], label
            gg, gu, act, dd, add = g_nodes[at:at+5]
            assert gg["src1"] == gu["src1"] and tensors[int(gg["src1"])]["name"] == f"ffn_norm-{layer}", label
            assert (act["src0"], act["src1"], dd["src1"], add["src1"]) == (
                gg["tensor"], gu["tensor"], act["tensor"], dd["tensor"]), label
            assert tensors[int(add["src0"])]["name"] == f"ffn_moe_out-{layer}", label
            assert all(shape(t, ((2304 if z < 3 else 5120), 1, 1, 1), "f32") for z,t in enumerate(ts[at:at+5]))
            ws = [tensors[int(n["src0"])] for n in (gg, gu, dd)]
            assert [t["name"] for t in ws] == [f"blk.{layer}.ffn_{k}_shexp.weight" for k in ("gate","up","down")]
            assert all(shape(t, dims, typ) for t, dims, typ in zip(ws, [(5120,2304,1,1)]*2+[(2304,5120,1,1)],
                                                                   ["q6_K","q8_0","q6_K"])), label
            assert ts[0]["name"] == f"ffn_moe_weighted-{layer}", label
            assert all("ffn_shexp-" not in t["name"] for t in ts[:at]), label
            nc = copied[call_id]
            ready = []
            for s in ("src1","src2"):
                covers = [c for c in nc if c["dst"] == c_nodes[j][s] and int(c["before_split_ordinal"]) == i]
                ready.append(int(exact(covers, (label,s))["return_ns"]))
            cpu0, cpu1 = int(cpu["dispatch_ns"]), int(cpu["return_ns"])
            assert all(t <= cpu0 for t in ready) and cpu0 < cpu1, label
            matches = [p for p in range(len(gpu[gid])) if signature(gpu[gid], p)]
            # L36's final CUDA split can also include GPU-routed layers 37..39.
            # Bound the match to this layer's weighted tail preceding its HC post.
            before_hc = next((p for p, k in enumerate(gpu[gid]) if "hc_post" in k["name"]), len(gpu[gid]))
            p = exact([p for p in matches if p+6 < before_hc], (label, "kernel signature", len(matches), before_hc))
            seq = gpu[gid][p:p+7]
            assert len({(k["device"],k["context"],k["stream"]) for k in seq}) == 1, label
            intervals = [(int(k["start_ns"]),int(k["end_ns"])) for k in seq]
            for x,y in zip(intervals,intervals[1:]):
                assert x[1] <= y[0], label
            record = dict(case=int(call["case_id"]), call=int(call["call_id"]), layer=layer,
                cpu_split=int(cpu["split"]), gpu_split=gid, norm_ready_ns=ready[0], ids_ready_ns=ready[1],
                cpu_begin_ns=cpu0, cpu_end_ns=cpu1, gpu_host_begin_ns=int(next_gpu["dispatch_ns"]),
                gpu_host_return_ns=int(next_gpu["return_ns"]), shared_begin_ns=intervals[0][0],
                gate_up_clamp_end_ns=intervals[4][1], fused_down_end_ns=intervals[6][1],
                stream=int(seq[0]["stream"]), graph=int(seq[0]["graph"]),
                graph_nodes=[int(k["graph_node"]) for k in seq], correlations=[int(k["correlation"]) for k in seq],
                kernel_ns=[b-a for a,b in intervals], prefix_busy_ns=sum(b-a for a,b in intervals[:5]),
                prefix_span_ns=intervals[4][1]-intervals[0][0], down_merge_busy_ns=sum(b-a for a,b in intervals[5:]),
                prefix_overlap_cpu_ns=sum(max(0,min(b,cpu1)-max(a,cpu0)) for a,b in intervals[:5]),
                cpu_to_prefix_ns=intervals[0][0]-cpu1)
            result.append(record)
        assert layers == list(range(37)), (call, layers)
    by_call = defaultdict(list)
    for row in result: by_call[(row["case"],row["call"])].append(row)
    metrics = ("prefix_busy_ns","prefix_span_ns","down_merge_busy_ns","prefix_overlap_cpu_ns")
    totals = [{"case":case,"call":call,**{k:sum(row[k] for row in v) for k in metrics}}
              for (case,call),v in by_call.items()]
    return dict(source_sha256=sha(Path(__file__)), trace_sha256={p.name:sha(p) for p in sorted(run.glob("trace169-*"))},
                attribution="graph DAG plus unique 7-kernel typed, shaped, fused signature before first matching layer HC post",
                per_layer=result, per_call=totals, median_per_call={k:statistics.median(t[k] for t in totals) for k in metrics})


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("run",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    analysis=analyze(args.run)
    with args.output.open("x") as out: json.dump(analysis,out,indent=2); out.write("\n")
    print(json.dumps(analysis["median_per_call"],indent=2))
