"""Aggregate per-worker timestamps and exact expert row/chunk assignments from the time-bounded trace."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics
import sys

NAMES = re.compile(r"^ffn_moe_(gate|up|down)-(\d+)$")


def read(p):return json.loads(p.read_bytes())


def summary(values):
    sorted_values=sorted(values)
    return {"n":len(values),"mean":statistics.mean(values),"median":statistics.median(values),
            "p95":sorted_values[(95*len(values)+99)//100-1],"max":sorted_values[-1]}


def load_table(path):
    groups=collections.defaultdict(list)
    with path.open() as f:
        for v in csv.DictReader(f,delimiter="\t"):
            nums={k:int(x) for k,x in v.items() if k not in ("name","op","qtype","experts")}
            experts=[list(map(int,x.split(","))) for x in v["experts"].split(";") if x]
            row={**nums,**{k:v[k] for k in ("name","op","qtype")},"experts":experts}
            assert row["start"]<=row["finish"]<=row["released"] and row["cpu"]==row["end_cpu"]
            groups[row["graph"],row["index"]].append(row)
    for key,rows in groups.items():
        assert sorted(r["ith"] for r in rows)==list(range(rows[0]["nth"])) and len({r["tid"] for r in rows})==len(rows)
        assert len({(r["name"],r["op"],r["nth"],r["nrows"],r["nin"],r["qtype"],r["sync"]) for r in rows})==1
        rows.sort(key=lambda r:r["ith"])
    return groups


def waiting(enter, exit_times):
    last=max(enter)
    assert min(exit_times)>=last
    return {"last_minus_mean_us":(last-statistics.mean(enter))/1000,"last_minus_first_us":(last-min(enter))/1000,
            "mean_release_after_last_us":(statistics.mean(exit_times)-last)/1000,
            "last_release_after_last_us":(max(exit_times)-last)/1000}


def analyze(root):
    run=root/"runs/worker-trace"
    driver=read(run/"RUN.json")
    checked=read(run/"VERIFIED.json")
    assert checked["all_full_rows_match"] and checked["worker_files"]==16
    calls=[];bykind=collections.defaultdict(list);cpu_last=collections.Counter()
    slot_rows=collections.defaultdict(collections.Counter);slot_ids=collections.defaultdict(set)
    totals=[];worker_map={};by_worker_spent=collections.defaultdict(list)
    for ci,b in enumerate(driver["blocks"]):
        assert b["case_index"]==ci
        for step in range(5,65,8):
            groups=load_table(run/f"worker-{ci}-{step}.tsv")
            assert len({g for g,n in groups})==driver["worker_graphs"][str(ci)][str(step)]
            call_begin, call_end=b["start_ns"][step-1],b["output_ready_ns"][step-1]
            rows_by_kind=collections.defaultdict(list)
            histogram=collections.Counter();graph_ends={}
            for (g,i),rows in groups.items():
                if rows[0]["op"]=="graph_end":graph_ends[g]=rows
                assert all(call_begin<=r["start"]<=r["released"]<=call_end for r in rows)
            coverage=set();call_rows=[]
            for key,rows in sorted(groups.items()):
                first=rows[0]
                m=NAMES.match(first["name"])
                if not m:continue
                kind,layer=m[1],int(m[2])
                assert first["nth"]==24 and first["nin"]==1 and first["op"]=="MUL_MAT_ID"
                assert {r["cpu"] for r in rows}==set(range(24)) and all(len(r["experts"])==6 for r in rows)
                assert (kind,layer) not in coverage
                coverage.add((kind,layer))
                for r in rows:
                    worker_map.setdefault(str(r["ith"]),set()).add((r["tid"],r["cpu"]))
                nrows=first["nrows"]
                assert nrows==(5120 if kind=="down" else 2304)
                same=tuple(x[0] for x in first["experts"])
                assert len(set(same))==6 and all(tuple(x[0] for x in r["experts"])==same for r in rows)
                for j,e in enumerate(same):
                    parts=[r["experts"][j] for r in rows]
                    assert all(len(x)==8 and x[1]==1 and x[5]==0 and x[4]>=0 and x[3]>=0 and x[7]>=x[6] for x in parts)
                    assert len({x[2] for x in parts})==1 and sum(x[3] for x in parts)==nrows
                    slot_rows[kind].update(x[3] for x in parts);slot_ids[kind].add(e)
                prep=waiting([v["prep_in"] for v in rows],[v["prep_out"] for v in rows])
                sync_rows=rows if first["sync"] else graph_ends[key[0]]
                assert [r["ith"] for r in sync_rows]==list(range(24))
                end=waiting([v["finish"] for v in sync_rows],[v["released"] for v in sync_rows])
                work=[(v["finish"]-v["prep_out"])/1000 for v in rows]
                arrivals=[v["finish"] for v in sync_rows]
                late=sync_rows[arrivals.index(max(arrivals))]["cpu"]
                cpu_last[late]+=1;histogram[late]+=1
                soc=[statistics.mean(work[i] for i,r in enumerate(rows) if r["cpu"]//12==s) for s in (0,1)]
                for i,r in enumerate(rows):by_worker_spent[r["cpu"]].append(work[i]/statistics.mean(work))
                item={"case":ci,"step":step,"graph":key[0],"layer":layer,"projection":kind,"weight_type":first["qtype"],
                      "active_ids":same,"nrows":nrows,"prep":prep,"end":end,"last_cpu":late,"socket_work_us":soc,
                      "work_mean_us":statistics.mean(work),"work_max_us":max(work),"work_min_us":min(work),
                      "max_minus_mean_work_us":max(work)-statistics.mean(work),
                      "node_span_us":(max(r["finish"] for r in rows)-min(r["start"] for r in rows))/1000}
                bykind[kind].append(item);call_rows.append(item)
            assert coverage=={(k,l) for k in ("gate","up","down") for l in range(37)}
            graph_wait=collections.defaultdict(float);release=collections.defaultdict(float)
            for (g,_),rows in groups.items():
                q=rows[0]
                if q["sync"] and q["nth"]>1:
                    target=q["op"] if q["op"]!="graph_end" else "graph_end"
                    v=waiting([r["finish"] for r in rows],[r["released"] for r in rows])
                    graph_wait[target]+=v["last_minus_mean_us"]/1000
                    release[target]+=v["mean_release_after_last_us"]/1000
            t={"case":ci,"step":step,"t1_ms":(call_end-call_begin)/1e6,
               "routed_node_span_sum_ms":sum(v["node_span_us"] for v in call_rows)/1000,
               "routed_inner_imbalance_ms":sum(v["prep"]["last_minus_mean_us"] for v in call_rows)/1000,
               "routed_end_imbalance_ms":sum(v["end"]["last_minus_mean_us"] for v in call_rows)/1000,
               "routed_inner_release_ms":sum(v["prep"]["mean_release_after_last_us"] for v in call_rows)/1000,
               "routed_end_release_ms":sum(v["end"]["mean_release_after_last_us"] for v in call_rows)/1000,
               "last_cpu_hist":dict(histogram),"all_explicit_node_barriers_ms":dict(graph_wait),
               "all_explicit_release_after_last_ms":dict(release)}
            totals.append(t);calls.extend(call_rows)
    kinds={}
    for kind,rs in bykind.items():
        kinds[kind]={"nodes":len(rs),"end_imbalance_us":summary([v["end"]["last_minus_mean_us"] for v in rs]),
                     "prep_imbalance_us":summary([v["prep"]["last_minus_mean_us"] for v in rs]),
                     "max_minus_mean_work_us":summary([v["max_minus_mean_work_us"] for v in rs]),
                     "socket_mean_work_us":[statistics.mean(v["socket_work_us"][s] for v in rs) for s in (0,1)],
                     "rows_per_worker_per_expert":{str(k):n for k,n in sorted(slot_rows[kind].items())},
                     "distinct_expert_ids":len(slot_ids[kind])}
    fields=[k for k in totals[0] if k.endswith("_ms") and isinstance(totals[0][k],float)]
    return {"sampled_T1":len(totals),"fully_covered_routed_projections":len(calls),"by_projection":kinds,
        "per_token":totals,"per_token_summary_ms":{k:summary([v[k] for v in totals]) for k in fields},
        "cpu_last_counts":{str(k):v for k,v in sorted(cpu_last.items())},
        "worker_locations":{k:sorted([list(x) for x in v]) for k,v in worker_map.items()},
        "worker_work_div_group_mean":{str(k):statistics.mean(v) for k,v in sorted(by_worker_spent.items())},
        "nodes":calls,"interpretation":"last minus team mean arrival is normalized worker waiting, not a wall-time speedup bound; instrumented only. Explicit inner and terminal synchronization separated."}


if __name__=="__main__":
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    r=analyze(root);out=root/"WORKER_RESULT.json"
    if "--check" in sys.argv:assert json.loads(json.dumps(r))==read(out)
    else:out.write_text(json.dumps(r,indent=1)+"\n")
    print("traced",r["sampled_T1"],"T1",r["fully_covered_routed_projections"],"routed projections")
    print({k:round(v["median"],3) for k,v in r["per_token_summary_ms"].items()})
    print({k:(v["rows_per_worker_per_expert"],round(v["end_imbalance_us"]["median"],2),v["socket_mean_work_us"]) for k,v in r["by_projection"].items()})
