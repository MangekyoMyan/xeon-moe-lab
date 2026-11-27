"""Separate cold prompt, reused context, and generated answer clocks."""
import argparse
import json
from pathlib import Path
import statistics as st

def load(p):return json.loads(Path(p).read_bytes())
def secs(j):return (j["end"]-j["start"])/1e9
def pair_summary(rows):
    times={a:[v["s"] for v in rows if v["arm"]==a] for a in ("O","P")}
    assert len(times["O"])==len(times["P"])==2
    result={a:st.mean(times[a]) for a in times}
    dif=[next(r["s"] for r in rows if r["pair"]==j and r["arm"]=="P")-next(r["s"] for r in rows if r["pair"]==j and r["arm"]=="O") for j in range(2)]
    return {"O_s":result["O"],"P_s":result["P"],"P_minus_O_s":result["P"]-result["O"],"percent":100*(result["P"]/result["O"]-1),
            "paired_delta_s":dif,"rows":rows}
def summarize(root,label):
    j=load(root/"runs"/label/"RUN.json");cfg=load(root/"large/CASES.json");assert j["passed"] and len(j["conversation"])==4
    cases={};all_primary=[]
    for c in cfg["cases"]:
        n=c["id"];rows=[r for r in j["timings"] if r["case"]==n];assert len(rows)==4
        cases[n]=pair_summary([{"pair":r["pair"],"arm":r["arm"],"s":secs(r)} for r in rows])
        cases[n]["singletons_per_layer"]=[z["cpu_pairs"] for r in rows if r["arm"]=="P" for z in r["layers"]][:2]
        cases[n]["GP_matches_O_17rows"]=next(r["identical_to_O"] for r in j["quality"] if r["case"]==n and r["arm"]=="GP")
        all_primary.extend(rows)
    h=cfg["conversation"];hist=[];new=[];answers=[];shapes=[]
    for r in j["conversation"]:
        chunks=r["prefix_chunks"];suffix=r["suffix"];assert sum(x["n"] for x in chunks)==len(h["common_prefix_tokens"])
        assert suffix["at"]==len(h["common_prefix_tokens"]) and suffix["n"]==len(h["suffix_tokens"]) and suffix["layers"]==[]
        for x in chunks:assert len(x["layers"])==(2 if x["n"]==1024 and r["arm"]=="P" else 0)
        if r["arm"]=="P":
            by_layer={}
            for layer in (4,24):
                records=[v for chunk in chunks for v in chunk["layers"] if v["layer"]==layer]
                nonempty=[v for v in records if not v.get("native_fallback",False)]
                by_layer[str(layer)]={"chunks":len(records),"singleton_counts":[v["cpu_pairs"] for v in records],
                    "graph_shapes_allocated":sum(bool(v["new_shape"]) for v in nonempty),
                    "max_private_gpu_scratch_B":max((v["scratch_GPU_B"] for v in nonempty),default=0)}
            shapes.append({"pair":r["pair"],"layers":by_layer})
        hist.append({"pair":r["pair"],"arm":r["arm"],"s":sum(secs(x) for x in chunks)})
        new.append({"pair":r["pair"],"arm":r["arm"],"s":secs(suffix)})
        txt=r["text"].strip();answers.append({"pair":r["pair"],"arm":r["arm"],"text":txt,"exact":txt==h["target"],"contains":h["target"] in txt,"greedy_ids":r["greedy"]})
        all_primary.extend(chunks)
    return {"run":label,"cold":cases,"history":{"common":len(h["common_prefix_tokens"]),"new":len(h["suffix_tokens"]),
            "chunks":[x["n"] for x in j["conversation"][0]["prefix_chunks"]],"prefix":pair_summary(hist),"suffix":pair_summary(new),
            "answers":answers,"target":h["target"],"P_workspaces":shapes},"min_recorded_free_GPU_B":min(x["free_B"] for x in all_primary)}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");p.add_argument("--check",action="store_true");a=p.parse_args()
    root=Path(__file__).resolve().parent;d=summarize(root,a.label);file=root/"LARGE_RESULT.json"
    if a.check:assert d==load(file)
    else:file.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
    print({k:(v["O_s"],v["P_s"]) for k,v in d["cold"].items()},[(r["arm"],r["text"]) for r in d["history"]["answers"]])
