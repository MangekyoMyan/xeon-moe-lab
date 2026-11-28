"""Verify the edited tool result, recovery fence and matched native score files."""
import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from analyze import read,lines

R=Path(__file__).resolve().parent

def check(root):
    e=root/"evidence";runs=root/"runs"
    tune=read(root/"VALIDATION_RUNS.json")
    toolr,cand,stock=[runs/tune[k] for k in ("tool","restart","stock")]
    tool=read(e/"tool-edit.json")
    a,b=tool["A_request"],tool["B_request"]
    diffs=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y];assert len(a)==len(b) and len(diffs)==1
    idx=diffs[0];assert a[idx]["role"]==b[idx]["role"]=="tool"
    assert {k:v for k,v in a[idx].items() if k!="content"}=={k:v for k,v in b[idx].items() if k!="content"}
    assert json.loads(a[idx]["content"])["registered_number"]==731 and json.loads(b[idx]["content"])["registered_number"]==284
    assert tool["external_tool_executions"]==0
    p=tool["parent"];n=len(p["tokens"]);assert tool["A_ids"][:n]==tool["B_ids"][:n]==p["tokens"]
    assert tool["B_reply"]["restored_prefix"]==p["id"] and tool["B_reply"]["timings"]["cache_n"]==n
    assert tool["A"]["sha"]!=tool["B"]["sha"] and len({p["id"],tool["A"]["id"],tool["B"]["id"]})==3
    after=read(e/"restart-answers.json");cobs={r["label"]:r for r in lines(cand/"logical.jsonl")}
    errors={r["label"]:r["error"] for r in lines(toolr/"logical.jsonl") if "error" in r}
    assert errors["native-reject-1"]["details"]["rollback"]["exact_head"] and errors["native-reject-2"]["code"]=="rollback_failed"
    assert errors["invalid-fence"]["code"]=="invalid_slot" and errors["uncertain-fence"]["code"]=="native_outcome_unknown"
    assert errors["crash-in-checkout"]["code"]=="manager_disconnected"
    cancelled=read(e/"cancelled-placement.json")
    assert errors["cancel-after-active-and-last-use-writes"]["details"]["rollback"]["exact_head"]
    assert cancelled["target_used_before"]==cancelled["target_used_after"]
    x=read(cand/"READY.json")["identity"];y=read(stock/"READY.json")["identity"]
    assert x["server_impl"]!=y["server_impl"] and {k:v for k,v in x.items() if k!="server_impl"}=={k:v for k,v in y.items() if k!="server_impl"}
    assert x["offload_thresholds"]==[128,128] and x["threads"]==[24,24]
    val=read(e/"validation-cases.json");manual={r["name"]:r for r in read(e/"stock-answers.json")}
    comparisons={}
    ct,st=[lines(d/"trace/calls.jsonl") for d in (cand,stock)]
    for t in val["prompts"]:
        name="fixed-"+t["name"];start=len(val["points"][t["anchor"]]["tokens"])
        groups=[[c for c in tr if c["phase"]==name] for tr in (ct,st)]
        assert len(groups[0])==len(groups[1])>=1
        assert groups[0][0]["input_ids"]==t["ids"][start:] and groups[0][0]["pos"]==list(range(start,len(t["ids"])))
        scores=[]
        for ca,cb in zip(*groups):
            assert ca["rc"]==cb["rc"]==0 and ca["outputs"]==cb["outputs"]==1
            assert ca["input_ids"]==cb["input_ids"] and ca["pos"]==cb["pos"] and ca["ubatches"]==cb["ubatches"]
            left=(cand/"trace"/ca["logits_file"]).read_bytes();right=(stock/"trace"/cb["logits_file"]).read_bytes()
            assert len(left)==len(right)==4*129280
            def floats(data):
                v=array.array("f");v.frombytes(data)
                if sys.byteorder!="little":v.byteswap()
                assert all(math.isfinite(x) for x in v)
                return v
            x=floats(left);identical=left==right;y=x if identical else floats(right)
            ia=max(range(len(x)),key=x.__getitem__);ib=ia if identical else max(range(len(y)),key=y.__getitem__)
            scores.append({"position":ca["pos"][-1],"equal":identical,"candidate_top1":ia,"native_top1":ib,
                           "candidate_sha256":hashlib.sha256(left).hexdigest(),"native_sha256":hashlib.sha256(right).hexdigest(),
                           "max_abs":0.0 if identical else max(abs(a-b) for a,b in zip(x,y))})
        text=cobs["read-"+t["name"]]["result"]["content"]
        comparisons[t["name"]]={"cached":start,"new":len(t["ids"])-start,"text":text,"expected":t["expected"],
                                 "stock_text":manual[t["name"]]["content"],"scores":scores}
    all_scores=[z for c in comparisons.values() for z in c["scores"]]
    past=read(root/"prior/NUMERIC.json")["cases"]
    prior_rows=[(z,s) for key,t in past.items() for z,s in zip(comparisons[key.removeprefix("score-")]["scores"],t["distributions"],strict=True)]
    assert len(tool["A_ids"])==len(tool["B_ids"])
    token_differences=[i for i,(left,right) in enumerate(zip(tool["A_ids"],tool["B_ids"])) if left!=right]
    assert token_differences and tool["B_reply"]["common_prefix"]==token_differences[0]
    return {"parent_tokens":n,"edited_full_tokens":len(tool["B_ids"]),"edited_new":tool["B_reply"]["timings"]["prompt_n"],
            "edited_token_positions_zero_based":token_differences,"textual_common_token_prefix":token_differences[0],
            "restored_prefix":p["id"],"tool_outputs":{"A":tool["A_reply"]["content"],"B":tool["B_reply"]["content"],"after_restart":after},
            "recovery_status":{"native_failure":"rollback","double_failure":"invalid","interrupted":"uncertain until new backend"},
            "cases":comparisons,"rows":len(all_scores),"byte_exact":sum(z["equal"] for z in all_scores),
            "compared_scores":len(all_scores)*129280,"previous_phase1_stock_rows":len(prior_rows),
            "same_previous_stock_hash":sum(z["native_sha256"]==s["stock_sha256"] for z,s in prior_rows)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=R);p.add_argument("--check",action="store_true")
    a=p.parse_args();v=check(a.root);dst=a.root/"VALIDATION.json"
    if a.check:assert v==read(dst)
    else:dst.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
    print("matching native rows",v["byte_exact"],"/",v["rows"],"tool cached/new",v["parent_tokens"],v["edited_new"])
