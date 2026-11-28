"""Feed the delivered policy the exact observations in their original chronological order."""
import json
from pathlib import Path
import sys
from analyze import lines, same

def check(root):
    cli=root/"cli" if (root/"cli").is_dir() else root/"checkout/current/experiments/state-branches"
    sys.path.insert(0,str(cli))
    from checkpoint_policy import CheckpointPolicy
    sessions=lines(root/"evidence/sessions.jsonl")
    if (root/"evidence/extra.jsonl").is_file():sessions+=lines(root/"evidence/extra.jsonl")
    calls=0
    for run in sessions:
        mode=run["mode"];p=CheckpointPolicy(mode,1024,runtime=run["initial"]["checkpoint_policy"]["runtime"],thresholds=(128,128))
        for route,samples in run["initial"].get("observation_windows",{}).get("restores",{}).items():
            for position,ms,size in samples:p.observe_restore(route.split("/")[0],position,size,ms)
        for row in run["rows"]:
            v=row["result"];obs=v.get("restore_observation")
            if obs:p.observe_restore(obs["route"],obs["position"],obs["bytes"],obs["ms"])
            if mode!="off":
                for t in v["_measurements"]["prefills"]:p.observe_prefill(t)
            if row["op"]!="turn":continue
            rec=v["checkpoint"];calls+=1
            if mode=="off":
                assert rec["status"]=="disabled"
                continue
            expected=p.decide(rec["anchor_id"],rec["anchor_processed"],rec["candidate_processed"])
            assert (expected["status"]=="attempt")== (rec["status"]=="saved")
            for k in ("basis","reason","samples","restore_route"):
                assert expected[k]==rec[k]
            for k,value in expected["estimate_ms"].items():assert same(value,rec["estimate_ms"][k])
            if rec["status"]=="saved":p.observe_save(rec["candidate_processed"],rec["image_bytes"],rec["checkpoint_elapsed_ms"])
    return calls

if __name__=="__main__":print("replayed shipped decisions",check(Path(__file__).resolve().parent))
