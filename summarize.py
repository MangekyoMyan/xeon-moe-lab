"""Aggregate complete, matched synthetic workflows. Use measurement-root paths or the review package."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def agg(root):
    workflow=root/"workflow-evidence"
    rows=lines(workflow/"workflows.jsonl")
    groups=defaultdict(list)
    for row in rows:groups[row["session"]].append(row)
    assert len(groups)==10,(len(groups),list(groups))
    seq=["A","edit127","A","edit128","A","edit129","A","edit255","A","edit256","A","edit257"]
    pooled=defaultdict(list)
    detail=[]
    for series,rr in groups.items():
        key=rr[0]["mode"]
        if series.endswith("unprobed"):
            assert [r["op"]["name"] for r in rr]==["A","edit256"]
            unprobed=[dict(name=r["op"]["name"],first_ms=r["op"]["first_ms"],total_ms=r["op"]["total_ms"]) for r in rr]
            continue
        assert [r["op"]["name"] for r in rr]==seq
        measurements=[]
        for idx,r in enumerate(rr):
            x=r["op"];m=x["receipt"];ret=x["resume"]
            ev=m["checkpoint"].get("events",[])
            saves=[e for e in ev if e["status"]=="saved"]
            assert m["turn_committed"]
            if x["name"].startswith("edit"):
                p=int(x["name"][4:]);assert x["common"]==p==ret["common_prefix"]
                assert ret["old_replayed"]==p-ret["from_position"]
                assert ret["new_input"]==len(x["ids"])-p
            point=dict(session=series,mode=key,index=idx,name=x["name"],first_ms=x["first_ms"],total_ms=x["total_ms"],
                       max_gap_ms=x["max_delta_gap_ms"],resume_ms=ret["total_ms"] if ret else None,
                       from_position=ret["from_position"] if ret else None,old_replayed=ret.get("old_replayed") if ret else None,
                       prompt_n=m["timings"]["prompt_n"],predicted_n=m["timings"]["predicted_n"],
                       answer=m["content"],positions=[e["position"] for e in saves],save_ms=sum(e["pause_ms"] for e in saves),
                       written=sum(e["bytes"] for e in saves),interval=m["checkpoint"].get("interval_tokens"),
                       phase=[e["phase"] for e in saves])
            measurements.append(point)
            pooled[(idx,key)].append(point)
        detail.extend(measurements)
    table=[]
    for idx,name in enumerate(seq):
        out={"name":name,"index":idx}
        for mode in ("off","fixed","auto"):
            vals=pooled[idx,mode];assert len(vals)==3
            out[mode]={k:median(t[k] for t in vals) if all(t[k] is not None for t in vals) else None for k in
                       ("first_ms","total_ms","resume_ms","old_replayed","prompt_n","predicted_n","save_ms","written","max_gap_ms")}
            out[mode].update(origins=[t["from_position"] for t in vals],intervals=[t["interval"] for t in vals])
        table.append(out)
    ids={}
    for x in rows:
        r=x["op"];sha=hashlib.sha256(json.dumps(r["ids"]).encode()).hexdigest()
        if r["name"] in ids:assert sha==ids[r["name"]]
        ids[r["name"]]=sha
    whole={};edits={}
    for mode in ("off","fixed","auto"):
        by=defaultdict(list)
        for point in detail:
            if point["mode"]==mode:by[point["session"]].append(point)
        whole[mode]=dict(median_session_wall_sum_ms=median(sum(p["total_ms"] for p in row) for row in by.values()),
                         median_cumulative_bytes=median(sum(p["written"] for p in row) for row in by.values()),
                         median_saves=median(sum(len(p["positions"]) for p in row) for row in by.values()))
        e=[p for row in by.values() for p in row if p["name"]!="A"]
        edits[mode]={k:median(p[k] for p in e) for k in ("resume_ms","first_ms","total_ms","old_replayed")}
    pressure=lines(workflow/"pressure.jsonl")
    assert len(pressure)==2
    for v in pressure:
        assert v["disk_actual"]<=18*1024**2 and v["rejection"]=="budget"
    return dict(schema=1,sessions=len(groups),operations=len(rows),primary_repeats=3,workload=seq,operations_by_index=table,
                per_mode=whole,pooled_edits=edits,unprobed_off=unprobed,
                all_answer_bodies_equal=all(len({p["answer"] for p in detail if p["index"]==i})==1 for i in range(len(seq))),
                answers_equal_positions=sum(len({p["answer"] for p in detail if p["index"]==i})==1 for i in range(len(seq))),
                pressure=[dict(policy=p["policy"],bytes=p["disk_actual"],retained=p["preview"]["currently_available"],
                               rejected_budget=p["rejection"]) for p in pressure],input_hashes=ids)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--write",action="store_true")
    a=parser.parse_args();data=agg(a.root)
    if a.write:(a.root/"SUMMARY.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    else:print(json.dumps(data["pooled_edits"],indent=2))

