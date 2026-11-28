"""Replay all observed capacity admissions against immutable synthetic token prefixes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/"source" if (R/"source").is_dir() else R/"checkout/current/experiments/state-branches"))
from native import pack,unpack
from planner import eviction,evaluate,Profile
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def replay():
    seen_edits=0;admissions=0;unique=0;all_results=[]
    exported=R/"workflow-evidence/catalog-prefixes.json"
    portable=json.loads(exported.read_bytes()) if exported.exists() else {}
    from_local=not exported.exists()
    for s in read(R/"workflow-evidence/plans.jsonl"):
        label=s["session"]
        if from_local:
            db=sqlite3.connect("file:"+str(R/"cases-final"/label/"branches.sqlite3")+"?mode=ro&immutable=1",uri=True)
            points={k:(tok,sha) for k,tok,sha in db.execute("SELECT id,tokens,sha FROM anchors")}
            db.close()
            portable[label]={k:[unpack(tok),sha] for k,(tok,sha) in points.items()}
        else:
            points={k:(pack(tok),sha) for k,(tok,sha) in portable[label].items()}
        current={}
        operations=[r["op"] for r in read(R/"workflow-evidence/workflows.jsonl") if r["session"]==label]
        for op in operations:
            data=pack(op["ids"]);history=len(data)//4+11
            for e in op["receipt"]["checkpoint"].get("events",[]):
                if not e.get("bytes"):continue
                rows={sha:dict(bytes=n,protected=False,positions=sorted({len(d)//4 for d in sources if data.startswith(d)}))
                      for sha,(n,sources) in current.items()}
                victim=eviction(rows,e["bytes"],48*1024**2,(0,history),incoming=e["position"])
                if e["status"]=="saved":
                    assert victim==e["evicted"],(label,op["name"],e["position"],victim,e["evicted"])
                    for sha in victim:current.pop(sha)
                    ids,sha=points[e["snapshot_id"]]
                    assert sha==e["sha"] and ids[:len(data)]==data[:len(ids)]
                    if len(ids)>len(data):data=ids
                    current.setdefault(sha,(e["bytes"],[]))[1].append(ids)
                    unique+=1
                else:
                    assert victim is None,(label,e,victim)
                admissions+=1
        v=s["plan"];cfg=v["config"];prof=Profile(v["profile"]["runtime"],v["profile"]["route_cuts"],s["profile"])
        new=evaluate(v["requested_range"],v["currently_available"],prof,cfg["target_ms"],v["return_interval"],
                     capture_allowed=cfg["mode"]!="off" and v["capacity"]["new_save_possible"])
        for k in ("target_met","estimated_max_resume","estimated_min_equivalent","max_replay_tokens"):
            assert v["current"][k]==new[k],(label,k)
        seen_edits+=new["evaluated_positions"]
        all_results.append({"session":label,"evaluated_positions":new["evaluated_positions"],"time_target":new["target_met"]})
    if from_local:exported.write_text(json.dumps(portable,separators=(",",":"))+"\n")
    return dict(capacity_admissions=admissions,successful=unique,session_status_positions=seen_edits,sessions=all_results)
if __name__=="__main__":
    data=replay();(R/"POLICY_REPLAY.json").write_text(json.dumps(data,indent=2)+"\n");print(data)

