"""Verify the compact archived record without an engine, model, network, or GPU."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from analyze import clocks,confusion,records,table

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def main():
    manifest=read(ROOT/"MANIFEST_SHA256.json")["files"]
    for v in manifest:
        p=ROOT/v["path"]
        assert not Path(v["path"]).is_absolute() and ".." not in p.relative_to(ROOT).parts
        assert p.stat().st_size==v["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==v["sha256"]
    assert clocks()==read(ROOT/"TIME_RESULT.json")
    passed=read(ROOT/"QUALITY_RESULT.json");d=read(ROOT/"DESIGN.json");ref={}
    for row in table(ROOT/"reference-router-sets.tsv"):
        k=(row["case"],int(row["layer"]),int(row["token"]))
        ref[k]=(list(map(int,row["ids64"].split(","))),list(map(int,row["ids128"].split(","))))
    assert len(ref)==7680
    for case in ("JA96","PY96"):
        data=list(struct.iter_unpack("<iifi",(ROOT/f"runs/candidate/q_D_{case}.gaps").read_bytes()))
        r=read(ROOT/"ROUTER_RESULT.json");labs=[];flags=[];per10=[]
        for i,(six,seven,gap,near) in enumerate(data):
            layer,pos=divmod(i,96);x,y=ref[case,layer,pos]
            assert x[-1]==six and seven not in x and near==(gap<=d["threshold"])
            flags.append(bool(near));labs.append(set(x)!=set(y));per10.append(gap<=r["cut_by_layer"]["0.1"][layer])
        assert confusion(flags,labs)==r["cases"][case]["absolute_L0_cut_across_all_layers"]
        assert confusion(per10,labs)==r["cases"][case]["per_layer_JA_calibrated"]["0.1"]
    error=collections.defaultdict(float);N=0
    for v in table(ROOT/"dot_reference.tsv"):
        error[v["case"],int(v["token"])]+=(float(v["GPU"])-float(v["F64"]))**2;N+=1
    loc=read(ROOT/"LOCAL_RESULT.json");assert N==loc["reference_dots"]==73728
    for name,case in loc["cases"].items():
        rows=case["per_token"];ss=[error[name,i] for i in range(96)];m=[v["near"] for v in rows]
        assert all(abs(row["SSE"]-x)<1.e-12 for row,x in zip(rows,ss))
        for shift,key in ((0,"share_of_baseline_squared_error"),(37,"same_budget_rotation37")):
            assert abs(sum(x for i,x in enumerate(ss) if m[(i+shift)%96])/sum(ss)-case[key])<1e-12
    for tag,v in passed["compared"].items():
        values=collections.defaultdict(list)
        for p in sorted((ROOT/"risks"/tag).glob("*.tsv")):
            values["fixed96" if p.stem in ("JA96","PY96") else "held_chats"].extend(table(p))
        for name,rs in values.items():
            check=v[name];assert len(rs)==check["rows"] and sum(r["A_top"]!=r["B_top"] for r in rs)==check["top_changes"]
            assert abs(statistics.mean(float(r["TV"]) for r in rs)-check["mean_TV"])<1e-12
            if tag=="F-A":
                for cut,stats in passed["output_gap_vs_actual_all_residual"][name].items():
                    assert confusion([float(r["A_margin"])<=float(cut) for r in rs],[r["A_top"]!=r["B_top"] for r in rs])==stats
    gpu=read(ROOT/"ROUTER_MICRO.json");rows=table(ROOT/"router_micro/samples.tsv")
    for case,summary in gpu["stats"].items():
        for field,x in summary.items():
            arms=[{int(v["pair"]):float(v[field]) for v in rows if v["case"]==case and int(v["decision"])==k} for k in (0,1)]
            assert len(arms[0])==len(arms[1])==24
            assert abs(statistics.median(arms[1][i]-arms[0][i] for i in arms[0])-x["paired_delta_median"])<1e-10
    cpu=read(ROOT/"SAMPLER_RESULT.json");tt=table(ROOT/"output_cost.tsv")
    for k in ("native","top2","cached1"):
        ns=[int(v["ns"])/1000 for v in tt if v["method"]==k];assert len(ns)==cpu["calls_per_method"]==1830 and statistics.median(ns)==cpu["median_us"][k]
    final=read(ROOT/"FINAL_STATE.json");assert final["api"]=="ok" and final["closed_window"] and final["ordinary_prefill_thresholds"]==[64,64]
    assert sum(len(records(ROOT/"runs"/q)) for q in ("F0","candidate","F1"))==70
    print(f"Verified {len(manifest)} files, 70 model prefix measurements, 73728 FP64 dot references, 7680 router positions and both decision microbenchmarks.")

if __name__=="__main__":main()
