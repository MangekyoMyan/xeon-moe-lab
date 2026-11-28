"""Verify real processed IDs, positions, candidate/stock placements and all vocabulary scores."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys
from analyze import lines,read

R=Path(__file__).resolve().parent
def verify(root):
    spec=read(root/"evidence/numeric-fixtures.json");idx=read(root/"VALIDATION_RUNS.json")
    private,stock=[root/"runs"/idx[k] for k in ("candidate","stock")]
    a,b=[read(d/"READY.json")["identity"] for d in (private,stock)]
    assert a["server_impl"]!=b["server_impl"] and {k:v for k,v in a.items() if k!="server_impl"}=={k:v for k,v in b.items() if k!="server_impl"}
    tr,ts=[lines(d/"trace/calls.jsonl") for d in (private,stock)]
    actual={x["name"]:x["response"] for x in read(root/"evidence/candidate-numeric.json")}
    manual={x["name"]:x["response"] for x in read(root/"evidence/stock-numeric.json")}
    cases={};total=exact=0
    for c in spec["cases"]:
        pos=len(spec["points"][c["anchor"]]["tokens"]);name=c["name"];cs=[x for x in ts if x["phase"]==name]
        assert cs and cs[0]["pos"]==list(range(pos,len(c["ids"]))) and cs[0]["input_ids"]==c["ids"][pos:]
        for mode in ("off","fixed"):
            key=mode+"-"+name;ms=[x for x in tr if x["phase"]==key];assert len(ms)==len(cs)
            out=actual[key];assert out["content"]==manual[name]["content"] and out["pending_sampled_ids"]==manual[name]["tokens"][-1:]
            assert out["normalized_input_count"]==len(c["ids"]) and out["timings"]["cache_n"]==pos
            comparison=[]
            for x,y in zip(ms,cs):
                assert x["rc"]==y["rc"]==0 and x["outputs"]==y["outputs"]==1
                assert x["pos"]==y["pos"] and x["input_ids"]==y["input_ids"] and x["ubatches"]==y["ubatches"]
                u=(private/"trace"/x["logits_file"]).read_bytes();v=(stock/"trace"/y["logits_file"]).read_bytes();assert len(u)==len(v)==129280*4
                eq=u==v;total+=1;exact+=eq
                def floats(raw):
                    f=array.array("f");f.frombytes(raw)
                    if sys.byteorder!="little":f.byteswap()
                    assert all(math.isfinite(i) for i in f)
                    return f
                aa=floats(u);bb=aa if eq else floats(v)
                comparison.append({"position":x["pos"][-1],"equal":eq,"max_abs":0.0 if eq else max(abs(i-j) for i,j in zip(aa,bb)),
                      "candidate_sha":hashlib.sha256(u).hexdigest(),"stock_sha":hashlib.sha256(v).hexdigest()})
            cases[key]={"cached":pos,"new":len(c["ids"])-pos,"answer":out["content"],"checkpoint":out["checkpoint"]["status"],"rows":comparison}
    bad=read(root/"evidence/uncertain.json");ent=read(root/"evidence/entry.json")
    assert bad["response"]["turn_committed"] and bad["response"]["checkpoint"]["status"]=="unknown" and bad["mode"]=="uncertain"
    assert ent["initial"]["mode"]=="ready" and ent["initial"]["ticket"]["generation"]==bad["after"]["generation"] and ent["recovered_equals_committed_tokens_and_pending"]
    return {"cases":cases,"rows":total,"equal":exact,"scores":total*129280,"recovered_tokens":ent["initial_branch"]["last_replay_tokens"],
            "restored_answer":ent["turn"]["content"],"launcher_auto_interval":ent["initial"]["checkpoint_policy"]["interval_tokens"]}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=R);p.add_argument("--check",action="store_true");a=p.parse_args()
    r=verify(a.root);out=a.root/"NUMERIC.json"
    if a.check:assert read(out)==r
    else:out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print("byte-exact",r["equal"],"/",r["rows"],"scores",r["scores"],"replayed",r["recovered_tokens"])
