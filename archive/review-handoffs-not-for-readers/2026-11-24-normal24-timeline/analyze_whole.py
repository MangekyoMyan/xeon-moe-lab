"""Recompute the matched disabled/serial full-model comparison and finite logits."""
import array
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_bytes())


def score(path_a,path_b,targets):
    vocab=129280;rows=[]
    assert len(targets)>0 and path_a.stat().st_size==path_b.stat().st_size==len(targets)*vocab*4
    with path_a.open("rb") as a,path_b.open("rb") as b:
        for i,target in enumerate(targets):
            raw=a.read(4*vocab);other=b.read(4*vocab)
            x=array.array("f");x.frombytes(raw)
            if sys.byteorder!="little":x.byteswap()
            assert len(x)==vocab and all(map(math.isfinite,x))
            same=raw==other
            if same:y=x
            else:
                y=array.array("f");y.frombytes(other)
                if sys.byteorder!="little":y.byteswap()
                assert all(map(math.isfinite,y))
            maxx=max(x);maxy=maxx if same else max(y)
            topx=x.index(maxx);topy=topx if same else y.index(maxy)
            e=0 if same else math.fsum((p-q)**2 for p,q in zip(x,y))
            den=math.fsum(p*p for p in x)
            lossx=math.log(math.fsum(math.exp(p-maxx) for p in x))+(maxx-x[target])
            lossy=lossx if same else math.log(math.fsum(math.exp(p-maxy) for p in y))+(maxy-y[target])
            rows.append({"row":i,"exact":same,"F_argmax":topx,"S_argmax":topy,"target_id":target,
                "reference_energy":den,"error_energy":e,"max_abs":0 if same else max(abs(p-q) for p,q in zip(x,y)),
                "F_target_nll":lossx,"S_target_nll":lossy})
    return {"shape":[len(rows),vocab],"exact_rows":sum(v["exact"] for v in rows),"argmax_changes":sum(v["F_argmax"]!=v["S_argmax"] for v in rows),
            "nmse":math.fsum(v["error_energy"] for v in rows)/math.fsum(v["reference_energy"] for v in rows),
            "max_abs":max(v["max_abs"] for v in rows),"mean_target_nll_delta_S_minus_F":statistics.mean(v["S_target_nll"]-v["F_target_nll"] for v in rows),
            "per_row":rows}


def calc(times):
    return {"calls":len(times),"T1_s":len(times)*1e9/sum(times),"mean_ms":statistics.mean(times)/1e6,
            "p95_ms":sorted(times)[math.ceil(len(times)*.95)-1]/1e6}


def analyze(root):
    cases=read(root/"driver/CASES.json")["cases"];out={}
    for ci,case in enumerate(cases):
        name=case["case_id"];filename=name+".f32";groups={};per={};hists={}
        for arm in ("F1","S1","S2","F2"):
            d=root/"runs"/("whole-"+arm);j=read(d/"RUN.json");v=read(d/"VERIFIED.json")
            assert j["passed"] and v["pid"]==j["linux_entry"]["pid"] and j["context"]["threads"]==j["context"]["threads_batch"]==24
            b=next(q for q in j["blocks"] if q["case_id"]==name)
            assert b["prefix_ids"]==case["prefix_ids"] and b["input_tokens"]==case["used_T1_ids"] and b["status"]==[0]*64
            assert b["file"]==filename and b["rows"]==65 and len(b["start_ns"])==len(b["output_ready_ns"])==64
            times=[z-a for a,z in zip(b["start_ns"],b["output_ready_ns"])];assert min(times)>0
            groups[arm]=times;per[arm]=calc(times)
            hists[arm]=[z-a for a,z in zip(b["bank_before_decode"]["H"][7],b["bank_after_decode"]["H"][7])]
            assert sum(hists[arm])==(64 if arm.startswith("S") else 0)
            assert all((b["bank_after_decode"]["calls"][k]-b["bank_before_decode"]["calls"][k])==(64 if arm.startswith("S") and k==7 else 0) for k in range(40))
        base=root/"runs/whole-F1"/filename
        assert base.read_bytes()==(root/"runs/control"/filename).read_bytes()==(root/"runs/whole-F2"/filename).read_bytes()
        cand=root/"runs/whole-S1"/filename
        assert cand.read_bytes()==(root/"runs/whole-S2"/filename).read_bytes()
        f=calc(groups["F1"]+groups["F2"]);s=calc(groups["S1"]+groups["S2"])
        paired=[per["S"+n]["T1_s"]/per["F"+n]["T1_s"] for n in ("1","2")]
        out[name]={"F":f,"S":s,"S_over_F_T1_s":s["T1_s"]/f["T1_s"],"order_pair_T1_s_ratio":paired,
            "both_pairs_and_p95_improve":min(paired)>1 and s["p95_ms"]<f["p95_ms"],"per_process":per,"H_each_run":hists,
            "quality":score(base,cand,case["next_targets_prefix_and64T1"])}
    return {"cases":out,"timed_calls":512,"tested_candidate":"Only layer 7 IDs 19,154,331 GPU-resident/serial, other layers unchanged. Same diagnostic DSO in F and S. F/S/S/F, one model per process.",
            "passes_timing_screen_both_workloads":all(v["both_pairs_and_p95_improve"] for v in out.values())}


if __name__=="__main__":
    root=Path(sys.argv[1] if len(sys.argv)>1 else ".");v=analyze(root)
    if "--check" in sys.argv:assert v==read(root/"WHOLE_MODEL.json")
    else:(root/"WHOLE_MODEL.json").write_text(json.dumps(v,indent=2)+"\n")
    for k,row in v["cases"].items():print(k,row["F"],row["S"],row["order_pair_T1_s_ratio"],row["quality"]["argmax_changes"],row["quality"]["mean_target_nll_delta_S_minus_F"])
