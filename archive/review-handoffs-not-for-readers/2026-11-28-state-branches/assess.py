"""Recompute branch, recovery and latency summaries from the synthetic receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as stats
import struct
from verify_rows import read,records,verify


def summarize(root):
    e=root/"evidence"
    runs=root/"runs"
    a,b,quota,c,entry=[read(e/f) for f in ("basic.json","deep.json","budget.json","final-cost.json","entry.json")]
    primary={x["step"]:x for x in records(runs/"basic-1/observations.jsonl")}
    deep={x["step"]:x for x in records(runs/"deep-1/observations.jsonl")}
    assert a["A_correct"]==(a["A_answer"].strip()=="731") and b["B_correct"]==(b["B_answer"].strip()=="284")
    assert a["P"]["id"]==primary["fork-A"]["response"]["branch"]["head"]==primary["fork-B"]["response"]["branch"]["head"]
    assert primary["before-fork-B"]["response"]["image_bytes"]==primary["after-fork-B"]["response"]["image_bytes"]
    assert a["B_appended"]>128 and a["A"]["processed"]==a["B"]["processed"]
    assert primary["back-to-A"]["response"]["exact_head"] and primary["back-to-A"]["response"]["replayed"]==0
    assert sorted(x.get("error","ok") for x in a["race"])==["ok","stale"]
    assert deep["reject-native-1"]["response"]["details"]["rollback"]["exact_head"]
    assert deep["reject-native-2"]["response"]["code"]=="rollback_failed" and deep["retained-head-2"]["response"]["mode"]=="invalid"
    assert deep["interrupted-reopen"]["response"]["mode"]=="uncertain" and deep["fence-ambiguous-backend"]["response"]["code"]=="native_outcome_unknown"
    assert deep["interrupted-reopen"]["response"]["ticket"]==b["crash_retained_ticket"]
    assert entry["exit"]==0 and entry["checkout_replayed"]==0 and not entry["actual_user_directory_created"]
    refs={"parent":a["P"],"point-A":a["A"],"point-B":a["B"]}
    refs.update({k:v["snapshot"] for k,v in read(e/"anchors.json").items()})
    ids={}
    for name,m in refs.items():
        raw=(e/"references"/(name+".slot")).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==m["image_sha256"] and len(raw)==m["image_bytes"]
        magic,version,nwords,flag,codec,n=struct.unpack("<IIIiII",raw[:24])
        assert (magic,version,flag,codec)==(0x67677371,4,-1,1) and nwords==n+4 and n==m["processed"]
        ids[name]=list(struct.unpack_from("<"+str(n)+"i",raw,24))
    assert ids["point-A"]!=ids["point-B"] and len(ids["point-A"])==len(ids["point-B"])
    for name in ("odd","even"):
        assert ids[name]==read(e/"anchors.json")[name]["ids"]
    assert b["different_codec"]=="incompatible"
    fallback={}
    for name,reason in (("missing-A","missing"),("damaged-B","damaged")):
        v=b[name]["receipt"]
        assert v["skipped"][0]["reason"]==reason and v["from_position"]+v["replayed"]==v["processed"]
        fallback[name]={k:v[k] for k in ("from_position","processed","replayed","exact_head")}
    assert quota["after"]<=quota["cap"] and quota["pinned_bytes"]<=quota["after"] and quota["recovery"]["replayed"]>0
    numerical=verify(root)
    ready={v:read(runs/(v+"-1")/"READY.json") for v in ("numeric","stock")}
    lhs=ready["numeric"]["identity"];rhs=ready["stock"]["identity"]
    assert lhs["server_impl"]!=rhs["server_impl"]
    assert {k:v for k,v in lhs.items() if k!="server_impl"}=={k:v for k,v in rhs.items() if k!="server_impl"}
    assert lhs["offload_thresholds"]==[128,128] and lhs["threads"]==[24,24]
    assert len(c["ordinary"])==8 and len(c["dirty_checkout"])==4
    def aggregate(rows,fields):
        out={}
        for path in ("native","managed"):
            group=[v for v in rows if v["path"]==path]
            out[path]={k:{"median":stats.median(v[k] for v in group),"all":[v[k] for v in group]} for k in fields}
        return out
    for v in c["ordinary"]:
        assert v["cache"]==255 and v["new"]==33 and v.get("image_bytes_added",0)==0
    dirty=c["dirty_checkout"]
    assert len({(v["cached"],v["new"],v["saved_tokens"],v["saved_bytes"],v["content"]) for v in dirty})==1
    assert all(v["resident_at_measurement"]<=c["image_budget"] for v in dirty)
    return {"functional":{"P_len":len(ids["parent"]),"P_bytes":a["P"]["image_bytes"],"siblings_len":a["A_and_B_same_length"],
               "sibling_bytes_each":a["A"]["image_bytes"],"B_appended":a["B_appended"],"switch_replayed":0,
               "A_answer":a["A_answer"],"B_after_restart":b["B_answer"],"concurrent_success":1,"concurrent_stale":1,
               "fallback":fallback,"budget":quota,"confirmed_failure_rollback":"restored original head",
               "failed_rollback":"invalid slot, recorded head retained","interrupted_commit":"fenced until new backend"},
            "numeric":{"cases":len(numerical["cases"]),"rows":numerical["rows"],"exact_rows":numerical["exact_rows"],
                       "scores":numerical["compared_scores"],"max_abs":numerical["max_abs"],
                       "natural_aligned":all(v["candidate_text"]==v["stock_text"] for v in numerical["cases"].values())},
            "ordinary":aggregate(c["ordinary"],("first_ms","total_ms")),
            "switch":aggregate(dirty,("switch_ms","return_to_first_ms")),"switch_inputs":{k:dirty[0][k] for k in ("saved_tokens","saved_bytes","cached","new","content")},
            "deep_cli_process_switch":b["dirty_switch_and_first"],"unicode":c["unicode"],"terminal_entry":entry}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--check",action="store_true")
    args=p.parse_args();value=summarize(args.root);target=args.root/"SUMMARY.json"
    if args.check: assert read(target)==value
    else: target.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    print("verified branch, recovery, full-vocabulary and paired timing evidence")
