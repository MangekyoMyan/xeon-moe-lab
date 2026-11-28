"""Replay the monotonic clocks and partition nested calls without double counting."""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def lines(p):return [json.loads(t) for t in p.read_text().splitlines()]
def ms(n):return n/1e6
def describe(v):
    m=st.median(v)
    return {"n":len(v),"median":m,"min":min(v),"max":max(v),"mad":st.median(abs(x-m) for x in v),"mean":st.mean(v)}


def partition(row):
    clocks=row["clocks"];events=row["trace"]["events"];children=defaultdict(list)
    assert events and events[0]["label"]=="manager_request" and events[0]["parent"] is None
    for i,e in enumerate(events):
        assert e["begin"]<=e["end"]
        if i:
            parent=e["parent"];assert 0<=parent<i
            assert events[parent]["begin"]<=e["begin"]<=e["end"]<=events[parent]["end"]
            children[parent].append(i)
    bins=Counter();calls=Counter()
    for i,e in enumerate(events):
        sub=[events[j] for j in children[i]]
        assert all(a["end"]<=b["begin"] for a,b in zip(sub,sub[1:]))
        exclusive=e["end"]-e["begin"]-sum(c["end"]-c["begin"] for c in sub)
        assert exclusive>=0
        path=[];which=i
        while which is not None:
            path.append(events[which]["label"]);which=events[which]["parent"]
        k=e["label"]
        if k=="sql_commit":k="sql_commit_outgoing" if "publish_outgoing" in path else "sql_commit_active"
        if k in ("existing_blob_sha","existing_token_check"):
            k=("outgoing_" if "publish_outgoing" in path else "target_")+k
        calls[k]+=1
        bins[k]+=exclusive
    bins["ticket_roundtrip"]=clocks["ticket_end"]-clocks["start"]
    bins["client_before_manager"]=events[0]["begin"]-clocks["ticket_end"]
    bins["manager_to_client"]=clocks["switch_end"]-events[0]["end"]
    assert min(bins.values())>=0 and sum(bins.values())==clocks["switch_end"]-clocks["start"]
    return dict(bins),dict(calls)


def analyze(root,stage):
    data=lines(root/"evidence"/(stage+".jsonl"))
    design=read(root/"EXTEND_PLAN.json")["order"] if stage=="extend" else read(root/"DESIGN.json")["order"][stage]
    keys=[(r["size"],r["duplicate"]) for r in design];out={}
    assert len(data)==len(design)
    for r,z in zip(data,design):assert all(r[k]==v for k,v in z.items())
    fixtures={key:read(root/"cases"/(key+".json")) for key in {k for k,_ in keys}}
    for (size,dup) in sorted(set(keys)):
        tag=size+("-resident" if dup else "-fresh");case=fixtures[size];p,t,d=(case["points"][k] for k in ("P","T","D"))
        selected=[r for r in data if r["size"]==size and r["duplicate"]==dup]
        groups=defaultdict(list);pairs=defaultdict(dict);traces=defaultdict(list)
        for r in selected:
            clocks=r["clocks"];v=r["variant"];x=r["outgoing"]
            assert r["replayed"]==0 and r["exact_head"] and r["images_evicted"]==0
            assert (x["sha"],x["bytes"],x["processed"])==(d["sha"],d["bytes"],len(d["tokens"]))
            assert r["target_id"]==t["id"] and r["target_sha"]==t["sha"]
            assert {k.removesuffix(".slot") for k in r["initial_inventory"]}=={q["sha"] for q in ([p,t,d] if dup else [p,t])}
            assert sum(r["final_inventory"].values())<=r["budget"] and set(r["initial_inventory"])<=set(r["final_inventory"])
            n=r["next"];tk=n["timings"]
            assert tk["cache_n"]==len(t["tokens"]) and tk["prompt_n"]==len(case["next_ids"])-len(t["tokens"])
            assert r["source_cache_n"]==len(p["tokens"]) and r["source_new"]==len(d["tokens"])-len(p["tokens"])
            assert n["content"]==case["answer"] and clocks["start"]<=clocks["switch_end"]<=clocks["next_begin"]<=clocks["first"]<=clocks["done"]
            if v=="native":
                assert n["sampled"]==case["answer_ids"]
            else:
                assert n["pending_sampled_ids"]==case["answer_ids"][-1:]
            nums={k:ms(clocks[c]-clocks["start"]) for k,c in (("switch_ms","switch_end"),("first_ms","first"),("done_ms","done"))}
            groups[v].append(nums);pairs[r["pair"]][v]=nums
            if "trace" in r:
                assert r["trace"]["synchronous"]==2 and r["trace"]["journal_mode"]=="wal"
                bins,calls=partition(r);traces[v].append((bins,calls))
        sums={v:{field:describe([n[field] for n in ns]) for field in ns[0]} for v,ns in groups.items()}
        deltas={}
        for a,b in itertools.combinations(sorted(groups),2):
            ps=[(g[a],g[b]) for g in pairs.values()];id=a+" minus "+b
            deltas[id]={}
            for field in ("switch_ms","first_ms","done_ms"):
                values=[l[field]-r[field] for l,r in ps]
                deltas[id][field]={**describe(values),"positive":sum(x>0 for x in values),"values":values}
        parts={}
        for v,z in traces.items():
            names=set().union(*(set(bins) for bins,calls in z))
            parts[v]={k:{"median_ms":st.median(ms(bins.get(k,0)) for bins,calls in z),
                         "counts":[calls.get(k,0) for bins,calls in z],
                         "total_ns":sum(bins.get(k,0) for bins,calls in z)} for k in sorted(names)}
        out[tag]={"timings":sums,"paired":deltas,"exclusive_parts":parts,"dirty_tokens":len(d["tokens"]),"dirty_bytes":d["bytes"],
                  "target_tokens":len(t["tokens"]),"target_bytes":t["bytes"],"next_new":len(case["next_ids"])-len(t["tokens"]),"answer":case["answer"]}
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["discover","confirm","timed","extend"])
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--check",action="store_true")
    a=p.parse_args();out=analyze(a.root,a.stage);path=a.root/(a.stage.upper()+".json")
    if a.check:assert read(path)==out
    else:path.write_text(json.dumps(out,indent=2)+"\n")
    for k,r in out.items():
        print(k,{v:round(m["switch_ms"]["median"],2) for v,m in r["timings"].items()})
