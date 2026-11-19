"""Verify early CUDA submission, exclusive device kernels and measured CPU overlap."""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from analyze_trace import by_integer, clamp, exact, group, mmvq, q8_1, rows, sha, shape


def union_ns(intervals):
    total = 0
    end = 0
    for a,b in sorted(intervals):
        if b>a:
            total += max(0, b-max(a,end))
            end = max(end,b)
    return total


def analyze(run):
    summary = json.loads((run/"trace169-summary.json").read_text())
    assert summary["calls"] == 8 and summary["splits"] == 904
    assert not any(summary[k] for k in ("unknown_kernel_correlation","foreign_thread_calls",
                                          "dropped_inside","unknown_records","errors"))
    calls = by_integer(rows(run,"calls"),"call")
    splits = group(rows(run,"splits"),"call")
    nodes = group(rows(run,"nodes"),"split")
    tensors = by_integer(rows(run,"tensors"),"tensor")
    copies = group(rows(run,"copies"),"call")
    expect_calls = {(c,i) for c in range(2) for i in range(1,5)}
    assert {(int(x["case_id"]),int(x["call_id"])) for x in calls.values()} == expect_calls
    def t(v):
        return tensors[int(v)]
    cor = {}
    for row in rows(run,"correlations"):
        assert cor.setdefault(int(row["correlation"]),int(row["external_id"])) == int(row["external_id"])
    kernels = defaultdict(list)
    for k in rows(run,"kernels"):
        kernels[cor[int(k["correlation"])]].append(k)
    for seq in kernels.values():
        seq.sort(key=lambda k: int(k["start_ns"]))
    record=[]
    scratch = set()
    for call_id, call in calls.items():
        local = sorted(splits[call_id],key=lambda s:int(s["ordinal"]))
        assert len(local)==int(call["n_splits"])==113
        layers=[]
        for ci,cpu in enumerate(local):
            if cpu["backend"]!="CPU": continue
            c=nodes[int(cpu["split"])]
            routed=[x for x in c if re.fullmatch(r"ffn_moe_gate-\d+",t(x["tensor"])["name"])]
            if not routed: continue
            gate=exact(routed,cpu)
            layer=int(t(gate["tensor"])["name"].split("-")[-1]);layers.append(layer)
            tag=(call["case_id"],call["call_id"],layer)
            assert ci>0 and ci+1<len(local)
            e,tail=local[ci-1],local[ci+1]
            assert e["backend"]==tail["backend"]=="CUDA0" and int(e["n_nodes"])==3
            en=nodes[int(e["split"])];tn=nodes[int(tail["split"])]
            assert [t(n["tensor"])["name"] for n in en]==[f"{stem}-{layer}" for stem in ("ffn_gate","ffn_up","ffn_swiglu_limited")],tag
            assert [t(n["tensor"])["op"] for n in en]==["MUL_MAT","MUL_MAT","GLU"]
            g,u,glu=en
            assert glu["src0"]==g["tensor"] and glu["src1"]==u["tensor"] and g["src1"]==u["src1"]
            assert all(shape(t(n["tensor"]),(2304,1,1,1),"f32") for n in en)
            assert all(t(n["tensor"])["view_hex"]=="0" for n in en)
            private=tuple(int(t(n["tensor"])["data_hex"],16) for n in en)
            assert len(set(private))==3 and len({t(n["tensor"])["buffer_hex"] for n in en})==1
            assert all(max(a,b)-min(a,b)>=2304*4 for j,a in enumerate(private) for b in private[j+1:])
            scratch.add(private)
            assert all(t(n["tensor"])["name"] not in {t(z["tensor"])["name"] for z in en} for n in tn)
            assert t(tn[0]["tensor"])["name"]==f"ffn_moe_weighted-{layer}"
            d=exact([j for j,n in enumerate(tn) if t(n["tensor"])["name"]==f"ffn_shexp-{layer}"],tag)
            dn,add=tn[d:d+2]
            assert dn["src1"]==glu["tensor"] and add["src1"]==dn["tensor"]
            assert t(add["src0"])["name"]==f"ffn_moe_out-{layer}" and t(add["tensor"])["name"]==f"ffn_out-{layer}"
            assert t(g["src0"])["name"]==f"blk.{layer}.ffn_gate_shexp.weight" and t(g["src0"])["type"]=="q6_K"
            assert t(u["src0"])["name"]==f"blk.{layer}.ffn_up_shexp.weight" and t(u["src0"])["type"]=="q8_0"
            assert t(dn["src0"])["name"]==f"blk.{layer}.ffn_down_shexp.weight" and t(dn["src0"])["type"]=="q6_K"
            assert t(g["src1"])["name"]==f"ffn_norm-{layer}"
            ready=[int(exact([q for q in copies[call_id] if q["before_split_ordinal"]==e["ordinal"] and q["dst"]==gate[s]],
                            (tag,s))["return_ns"]) for s in ("src1","src2")]
            h0,h1 = int(cpu["dispatch_ns"]),int(cpu["return_ns"])
            assert max(ready)<=int(e["dispatch_ns"])<=int(e["return_ns"])<=h0<h1<=int(tail["dispatch_ns"]),tag
            pre=kernels[int(e["split"])]
            assert len(pre)==5 and q8_1(pre[0],20) and mmvq(pre[1],14,False,2304) and q8_1(pre[2],20) and \
                mmvq(pre[3],8,False,2304) and clamp(pre[4]),tag
            post=kernels[int(tail["split"])]
            hcp=next((j for j,k in enumerate(post) if "hc_post" in k["name"]),len(post))
            at=exact([j for j in range(hcp-1) if q8_1(post[j],10) and mmvq(post[j+1],14,True,5120)],tag)
            assert not any(clamp(k) for k in post[:hcp]),tag
            end=post[at:at+2]; seq=pre+end
            assert len({(k["device"],k["context"],k["stream"]) for k in seq})==1,tag
            intervals=[(int(k["start_ns"]),int(k["end_ns"])) for k in seq]
            assert all(a<=b for a,b in intervals) and h1<=intervals[5][0],tag
            adjacent_overlap=[max(0,a[1]-b[0]) for a,b in zip(intervals,intervals[1:])]
            per_kernel_overlap=[max(0,min(b,h1)-max(a,h0)) for a,b in intervals[:5]]
            overlap=union_ns([(max(a,h0),min(b,h1)) for a,b in intervals[:5]])
            record.append(dict(case=int(call["case_id"]),call=int(call["call_id"]),layer=layer,
                cpu_split=int(cpu["split"]),early_split=int(e["split"]),tail_split=int(tail["split"]),
                norm_ready_ns=ready[0],ids_ready_ns=ready[1],early_submit_ns=int(e["dispatch_ns"]),
                early_submit_return_ns=int(e["return_ns"]),cpu_start_ns=h0,cpu_end_ns=h1,
                early_start_ns=intervals[0][0],early_end_ns=intervals[4][1],down_start_ns=intervals[5][0],
                merge_end_ns=intervals[6][1],stream=int(seq[0]["stream"]),
                early_graph=int(pre[0]["graph"]),tail_graph=int(end[0]["graph"]),
                graph_nodes=[int(k["graph_node"]) for k in seq],correlations=[int(k["correlation"]) for k in seq],
                kernel_ns=[b-a for a,b in intervals],kernel_intervals_ns=intervals,
                apparent_adjacent_overlap_ns=adjacent_overlap,prefix_busy_ns=union_ns(intervals[:5]),
                prefix_sum_kernel_ns=sum(b-a for a,b in intervals[:5]),
                prefix_span_ns=max(b for a,b in intervals[:5])-min(a for a,b in intervals[:5]),
                down_merge_busy_ns=union_ns(intervals[5:]),prefix_overlap_cpu_ns=overlap,
                per_kernel_overlap_cpu_ns=per_kernel_overlap))
        assert layers==list(range(37)),(call,layers)
    assert len(scratch)==1 and len(record)==296
    grouped=defaultdict(list)
    for row in record: grouped[(row["case"],row["call"])].append(row)
    totals=[]
    for (c,i),v in grouped.items():
        totals.append(dict(case=c,call=i,
            **{k:sum(row[k] for row in v) for k in ("prefix_busy_ns","prefix_span_ns","prefix_overlap_cpu_ns","down_merge_busy_ns")},
            layers_with_any_overlap=sum(row["prefix_overlap_cpu_ns"]>0 for row in v),
            layers_fully_overlap=sum(row["prefix_busy_ns"]==row["prefix_overlap_cpu_ns"] for row in v)))
    positive=all(t["prefix_overlap_cpu_ns"]>0 for t in totals)
    return dict(trace_sha256={p.name:sha(p) for p in sorted(run.glob("trace169-*"))},source_sha256=sha(Path(__file__)),
        structural_guard_passed=True,actual_layer_calls=len(record),positive_overlap_every_call=positive,
        number_of_calls_with_observed_overlap=sum(t["prefix_overlap_cpu_ns"]>0 for t in totals),
        layer_calls_with_apparent_adjacent_interval_overlap=sum(any(x["apparent_adjacent_overlap_ns"]) for x in record),
        largest_apparent_adjacent_interval_overlap_ns=max(max(x["apparent_adjacent_overlap_ns"]) for x in record),
        duration_method="Union of kernel intervals, and union after clipping to the synchronous CPU interval. Individual raw endpoints, duration sums and apparent overlaps are retained.",
        all_early5_and_fused_tail_verified=True,private_payload_bytes=3*2304*4,distinct_private_address_triples=len(scratch),
        cpu_interval="Synchronous CPU provider graph dispatch, including plan construction and ggml_graph_compute.",
        baseline_reference="Original trace: 296 layer calls, zero shared-prefix overlap with the CPU dispatch.",
        overlap_scope="All measured overlaps including zero are retained; CUPTI and host-side descriptor collection affect dispatch latency. Uninstrumented timing is evaluated separately.",
        per_layer=record,per_call=totals,median_prefix_overlap_ns=statistics.median(t["prefix_overlap_cpu_ns"] for t in totals))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("run",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    result=analyze(args.run)
    with args.output.open("x") as f: json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:result[k] for k in ("structural_guard_passed","actual_layer_calls","positive_overlap_every_call","median_prefix_overlap_ns")}))
