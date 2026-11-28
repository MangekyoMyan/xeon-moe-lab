"""Recompute native-vs-branch numerical checks from the synthetic trace and FP32 rows."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def values(data):
    x=array.array("f")
    x.frombytes(data)
    if sys.byteorder!="little": x.byteswap()
    assert len(x)==129280 and all(math.isfinite(v) for v in x)
    return x


def compare_rows(a,b):
    aa,bb=a.read_bytes(),b.read_bytes()
    x=values(aa);exact=aa==bb;y=x if exact else values(bb)
    ia=max(range(len(x)),key=x.__getitem__);ib=ia if exact else max(range(len(y)),key=y.__getitem__)
    return {"candidate_sha256":sha(aa),"stock_sha256":sha(bb),"scores":len(x),"exact":exact,
            "top1_candidate":ia,"top1_stock":ib,"max_abs":0 if exact else max(abs(k-j) for k,j in zip(x,y))}


def verify(root):
    evidence=root/"evidence"
    src=[root/"runs"/(s+"-1")/"trace" for s in ("numeric","stock")]
    calls=[records(d/"calls.jsonl") for d in src]
    cases=read(evidence/"numeric-inputs.json");anchors=read(evidence/"anchors.json")
    stock={x["name"]:x for x in read(evidence/"manual-stock.json")}
    cli={v["step"]:v for v in records(root/"runs/numeric-1/observations.jsonl")}
    full=(1<<37)-1
    out={}
    for case in cases:
        name="score-"+case["name"]
        pairs=[[c for c in rows if c["phase"]==name] for rows in calls]
        assert len(pairs[0])==len(pairs[1])>=1
        n=len(anchors[case["anchor"]]["ids"]);new=case["new"];total=n+new
        assert len(case["ids"])==total
        a,b=pairs[0][0],pairs[1][0]
        assert a["pos"]==list(range(n,total)) and a["input_ids"]==case["ids"][n:]
        assert len(a["ubatches"])==1 and a["ubatches"][0]["tokens"]==new
        dev="CPU" if new<128 else "CUDA0"
        assert a["ubatches"][0][dev]==[full]*3
        q=[]
        for i,(a,b) in enumerate(zip(*pairs)):
            assert a["rc"]==b["rc"]==0 and a["outputs"]==b["outputs"]==1
            assert a["pos"]==b["pos"] and a["input_ids"]==b["input_ids"] and a["ubatches"]==b["ubatches"]
            if i: assert a["pos"]==[total+i-1]
            q.append({"last_position":a["pos"][-1],**compare_rows(src[0]/a["logits_file"],src[1]/b["logits_file"])})
        actual=cli[name]["response"]
        assert actual["timings"]["cache_n"]==stock[name]["timings"]["cache_n"]==n
        assert actual["timings"]["prompt_n"]==stock[name]["timings"]["prompt_n"]==new
        out[name]={"parent_tokens":n,"new":new,"placement_host_experts":dev,"positions_and_inputs_equal":True,"distributions":q,
                   "candidate_text":actual["content"],"stock_text":stock[name]["content"],"correct":actual["content"].strip()=="619"}
    all_rows=[z for row in out.values() for z in row["distributions"]]
    return {"cases":out,"vocab":129280,"rows":len(all_rows),"exact_rows":sum(r["exact"] for r in all_rows),
            "compared_scores":sum(r["scores"] for r in all_rows),"max_abs":max(r["max_abs"] for r in all_rows)}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--check",action="store_true")
    a=p.parse_args();v=verify(a.root);dest=a.root/"NUMERIC.json"
    if a.check: assert read(dest)==v
    else: dest.write_text(json.dumps(v,indent=2)+"\n")
    print(f'{v["exact_rows"]}/{v["rows"]} full rows byte-exact, {v["compared_scores"]} scores, max abs {v["max_abs"]}')
