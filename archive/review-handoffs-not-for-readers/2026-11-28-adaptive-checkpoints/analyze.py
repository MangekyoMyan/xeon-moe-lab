"""Replay published per-request receipts and the cost formula without an engine or model."""
import argparse
import array
from bisect import bisect_right
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def token_sha(ids):
    v=array.array("i",ids)
    if sys.byteorder!="little":v.byteswap()
    return hashlib.sha256(v.tobytes()).hexdigest()
def comparable(window,n):return [x for x in window if n<=2*x[0] and x[0]<=2*n]
def median3(s):return st.median(x[1] for x in s) if len(s)>=3 else None
def same(x,y):
    return x is y if x is None or y is None else math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-8)
def equivalent(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return same(a,b)
    return a==b
def verify_session(root,run,jobs,roots):
    mode=run["mode"];interval=1024
    initial=run["initial"]
    win=initial.get("observation_windows",{"prefill":{},"saves":[],"restores":{},"restore_route":None})
    cuts=initial["checkpoint_policy"]["prefill_cutpoints"]
    pp=defaultdict(lambda:deque(maxlen=8));ss=deque(win["saves"],maxlen=8);rs=defaultdict(lambda:deque(maxlen=8))
    for k,w in win["prefill"].items():pp[int(k)].extend(w)
    for k,w in win["restores"].items():rs[k].extend(w)
    route=win["restore_route"]
    snaps={z["id"]:{"tokens":tuple(z["tokens"]),"parent":None,"sha":z["sha"]} for z in roots.values()}
    resident={z["sha"] for z in roots.values()}
    branches={}
    for name,v in jobs.items():
        a=roots[v["root"]];branches[name]={"ids":tuple(a["tokens"]),"base":a["id"]}
    active="linear";ticket=initial["ticket"];predictions=[];restored=[]
    for row in run["rows"]:
        tag=row["tag"];op=row["op"];v=row["result"];diag=v["_measurements"];branch=branches[active]
        note=v.get("restore_observation")
        if note:
            kind="image" if note["bytes"] else "empty";route=note["route"]
            rs[route+"/"+kind].append((note["position"],note["ms"],note["bytes"]))
            restored.append({"tag":tag,"route":route,"id":v.get("restored_prefix",v.get("from_snapshot")),"ms":note["ms"]})
        for t in diag["prefills"]:
            if mode!="off" and t["prompt_n"] and t["prompt_ms"] and t["prompt_ms"]>0:
                pp[bisect_right(cuts,t["prompt_n"])].append((t["prompt_n"],t["prompt_ms"]))
        bump=1
        if op=="turn":
            name,step=tag.split(".");task=jobs[name][step];assert active==name
            ids=tuple(task["ids"]);samples=diag["sampled_ids"]
            assert token_sha(ids)==diag["input_sha256"]
            assert len(samples)==v["sampled_count"] and samples[-1:]==v["pending_sampled_ids"]
            processed=ids+tuple(samples[:-1]);n=len(processed);tm=v["timings"]
            assert n==v["processed"] and len(ids)==tm["cache_n"]+tm["prompt_n"]
            assert v["turn_committed"] and v["ticket"]["generation"]==ticket["generation"]+1
            cp=v["checkpoint"];assert cp["mode"]==mode and cp["interval_tokens"]==interval
            if v["previous_boundary_matches"]:
                assert ids[:len(branch["ids"])]==branch["ids"]
            elif v["restored_prefix"]:
                old=snaps[v["restored_prefix"]]["tokens"]
                assert ids[:len(old)]==old and tm["cache_n"]==len(old)
            branch["ids"]=processed
            if mode=="off":
                assert cp["status"]=="disabled" and not diag["saves"]
                if v["previous_boundary_matches"]:
                    assert diag["counts"].get("image_sha",0)==diag["counts"].get("file_fsync",0)==0
            else:
                chain=[];sid=branch["base"]
                while sid:
                    z=snaps[sid]
                    if z["sha"] in resident and processed[:len(z["tokens"])]==z["tokens"]:chain.append(sid)
                    sid=z["parent"]
                anchor=max(chain,key=lambda i:len(snaps[i]["tokens"]),default=None)
                start=len(snaps[anchor]["tokens"]) if anchor else 0;d=n-start
                assert (cp["anchor_id"],cp["anchor_processed"],cp["distance_tokens"])==(anchor,start,d),(tag,mode,anchor,cp)
                if mode=="auto" and d>0:
                    p=comparable(pp[bisect_right(cuts,d)],d);s=comparable(ss,n);images=rs.get(str(route)+"/image",())
                    a=comparable(images,start) if start else rs.get(str(route)+"/empty",())
                    b=comparable(images,n)
                    assert cp["samples"]=={"prefill":len(p),"save":len(s),"restore_anchor":len(a),"restore_candidate":len(b)}
                    assert cp["restore_route"]==route
                    ra=median3(a);rb=median3(b);sv=median3(s)
                    ptime=d*sum(i[1] for i in p)/sum(i[0] for i in p) if len(p)>=3 else None
                    if not start and ra is None:ra=0
                    est=cp["estimate_ms"]
                    for key,value in (("restore_anchor",ra),("restore_candidate",rb),("save",sv),("replay",ptime)):
                        assert same(value,est[key]),(tag,key,value,est)
                    if all(x is not None for x in (ra,rb,sv,ptime)):
                        gain=ptime+ra-rb
                        assert same(gain,est["gain"]) and cp["basis"]=="cost_model" and cp["model_accept"]==(gain>1.5*sv)
                        assert (cp["status"]=="saved")==cp["model_accept"]
                        compare={"rep":run["rep"],"tag":tag,"estimate":est,"counts":cp["samples"],"route":route,"anchor_id":anchor,
                                 "new_id":cp["snapshot_id"],"candidate":n,"distance":d,"accepted":cp["model_accept"],
                                 "observed_optional_S_ms":cp["checkpoint_elapsed_ms"] if cp["status"]=="saved" else None,
                                 "observed_prompt_P_ms":tm["prompt_ms"] if start==tm["cache_n"] and d==tm["prompt_n"] and len(ids)==n else None}
                        predictions.append(compare)
                    else:assert cp["basis"]=="cold_fallback" and (cp["status"]=="saved")== (d>=interval)
                elif mode=="fixed":assert (cp["status"]=="saved")== (d>=interval)
                assert cp["status"] in ("skipped","saved")
            if cp["status"]=="saved":
                bump+=1;assert cp["snapshot_committed"] and cp["native_save_called"] and cp["image_bytes"]>0
                assert cp["snapshot_id"]==diag["saves"][0]["snapshot_id"]
                ss.append((n,cp["checkpoint_elapsed_ms"],cp["image_bytes"]))
            if v["previous_boundary_matches"]:
                assert diag["counts"].get("directory_fsync",0)==(cp["status"]=="saved") and diag["counts"].get("file_fsync",0)==(cp["status"]=="saved")
                assert diag["counts"]["sql_commit"]==bump and diag["counts"]["sql_autocommit"]==bump
                if cp["status"]!="saved":assert diag["counts"].get("image_sha",0)==0
            else:
                extra_touch=int(bool(v["restored_prefix"] and snaps[v["restored_prefix"]]["sha"]))
                assert diag["counts"]["sql_commit"]==bump and diag["counts"]["sql_autocommit"]==bump+extra_touch
        for save in diag["saves"]:
            if save["committed"]:
                sid=save["snapshot_id"];assert sid not in snaps
                snaps[sid]={"tokens":branch["ids"],"sha":save["image_sha256"],"parent":branch["base"]};branch["base"]=sid
                resident.add(save["image_sha256"]);resident.difference_update(save["evicted"])
            else:
                assert save["stage"]=="reused" and save["snapshot_id"]==branch["base"]
        if op=="checkout":
            name=tag.split(".")[0]
            assert v["exact_head"] and v["replayed"]==0 and v["from_snapshot"]==branches[name]["base"]
            active=name
        assert v["ticket"]["revision"]==ticket["revision"]+bump and v["ticket"]["session"]==ticket["session"]
        ticket=v["ticket"]
    assert ticket==run["final"]["ticket"] and run["initial"]["budget_bytes"]==run["final"]["budget_bytes"]==1024**3
    final=run["final"].get("observation_windows")
    if final:
        def compact(mapping):
            return {str(k):[tuple(s) for s in val] for k,val in mapping.items() if val}
        assert compact(pp)==compact(final["prefill"]) and compact(rs)==compact(final["restores"])
        assert [tuple(x) for x in ss]==[tuple(x) for x in final["saves"]]
        assert route==final["restore_route"]
    for pred in predictions:
        for label,key in (("A","anchor_id"),("B","new_id")):
            pred["real_R_"+label]=[{"tag":x["tag"],"ms":x["ms"]} for x in restored if x["route"]==pred["route"] and pred[key] and x["id"]==pred[key]]
    return predictions


def phase_summary(run,phase):
    chunk=run["rows"][phase["from_row"]:phase["to_row"]];before,after=phase["before"],phase["after"]
    c=Counter();decisions=Counter();turns=[];optional_ms=0.0
    for row in chunk:
        v=row["result"];m=v["_measurements"];c.update(m["counts"])
        for s in m["saves"]:
            key="optional" if s["optional"] else "mandatory"
            c[key+"_writes"]+=int(s["native_called"]);c[key+"_bytes"]+=s["written_bytes"] or 0
            c["noop"]+=int(s["stage"]=="reused");c["evictions"]+=len(s["evicted"])
        if row["op"]=="turn":
            cp=v["checkpoint"];tm=v["timings"];decisions[cp["basis"]+":"+cp["status"]]+=1
            c["prompt_n"]+=tm["prompt_n"];c["cache_n"]+=tm["cache_n"]
            optional_ms+=cp["checkpoint_elapsed_ms"]
            turns.append({"tag":row["tag"],"first_ms":(row["first"]-row["start"])/1e6 if row["first"] else None,
                "done_ms":(row["end"]-row["start"])/1e6,"prompt_ms":tm["prompt_ms"],"prompt_n":tm["prompt_n"],"cache_n":tm["cache_n"],
                "answer":v["content"],"anchor":v["restored_prefix"],"common":v["common_prefix"],"processed":v["processed"],
                "decision_ms":cp["decision_ms"],"save_ms":cp["checkpoint_elapsed_ms"],"basis":cp["basis"],"status":cp["status"],"reason":cp["reason"]})
        else:c["replayed"]+=v.get("replayed",0)
    return {"wall_s":(phase["end"]-phase["start"])/1e9,"sum_requests_s":sum((r["end"]-r["start"]) for r in chunk)/1e9,
            "counts":dict(c),"decisions":dict(decisions),"optional_ms":optional_ms,"turns":turns,
            "snapshot_count_delta":after["snapshots"]-before["snapshots"],"resident_delta":after["image_bytes"]-before["image_bytes"],
            "metadata_bytes_delta":{k:v-before["metadata_files"][k] for k,v in after["metadata_files"].items()}}


def analyze(root):
    design=read(root/"DESIGN.json");jobs=read(root/"fixtures/workload.json");roots={n:read(root/"fixtures"/(n+"-anchor.json")) for n in ("short","long")}
    runs=lines(root/"evidence/sessions.jsonl")
    assert design["sequence"]==[[r["rep"],r["mode"]] for r in runs]
    assert hashlib.sha256((root/"fixtures/workload.json").read_bytes()).hexdigest()==design["freeze_sha256"]
    summaries=[];pred=[]
    for r in runs:
        pred+=verify_session(root,r,jobs,roots)
        ph={s["name"]:phase_summary(r,s) for s in r["phases"]}
        all_counts=Counter()
        for v in ph.values():all_counts.update(v["counts"])
        summaries.append({"rep":r["rep"],"mode":r["mode"],"phases":ph,"total_s":(r["last_ns"]-r["first_ns"])/1e9,
             "main_s":sum(ph[k]["wall_s"] for k in ("linear","late","early")),"setup_s":sum(ph[k]["wall_s"] for k in ("enter-late","enter-early")),
             "counts":dict(all_counts),"final":r["final"],"initial":r["initial"]})
    paired={}
    for cat in ("linear","late","early","preparation","main_s","total_s"):
        values={}
        for mode in ("off","fixed","auto"):
            values[mode]=[r[cat] if cat.endswith("_s") else r["phases"][cat]["wall_s"] for r in summaries if r["mode"]==mode]
        deltas={m:[a-b for a,b in zip(values["off"],values[m],strict=True)] for m in ("fixed","auto")}
        paired[cat]={"raw_seconds":values,"medians":{k:st.median(v) for k,v in values.items()},
                     "paired_seconds_saved":deltas,"paired_medians":{k:st.median(v) for k,v in deltas.items()}}
    return {"runs":summaries,"paired":paired,"actual_cost_comparisons":pred}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--check",action="store_true");a=p.parse_args()
    v=analyze(a.root);out=a.root/"RESULT.json"
    if a.check:assert equivalent(read(out),v)
    else:out.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
    for k,c in v["paired"].items():print(k,{m:round(v,3) for m,v in c["medians"].items()})
    print("cost-model evaluations",len(v["actual_cost_comparisons"]))
