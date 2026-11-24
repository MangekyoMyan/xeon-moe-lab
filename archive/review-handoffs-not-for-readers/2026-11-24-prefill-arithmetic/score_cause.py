"""Measure what changes downstream when only layer zero is altered."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

def read(p):return json.loads(p.read_bytes())


def compare(root):
    fixtures=read(root/"precision/FIXTURES.json")
    run=root/"runs/causal";actual=read(run/"RUN.json")
    assert read(run/"VERIFIED.json")["control_equals_normal_128_all_rows"]
    dest=root/"risk/causal";dest.mkdir(exist_ok=True)
    by_mode={k:[] for k in ("control","gpu-gates","residual","gpu-layer-output")}
    for item in fixtures["cases"]:
        name=item["id"]
        if name not in ("JA96","PY96"):continue
        inputs=dest/(name+".ids");inputs.write_text("\n".join(map(str,item["targets"]))+"\n")
        for mode,acc in by_mode.items():
            table=dest/(name+"-"+mode+".tsv")
            ref=root/"runs/precision-64-64"/(name+".f32")
            new=run/(name+"-"+mode+".f32")
            subprocess.run([str(root/"precision/score_logits"),str(ref),str(new),str(inputs),str(table)],check=True)
            for raw in csv.DictReader(table.open(),delimiter="\t"):
                acc.append({"case":name,"row":int(raw["row"]),"A":int(raw["A_top"]),"B":int(raw["B_top"]),"TV":float(raw["TV"]),
                        "nll_change":float(raw["B_target_nll"])-float(raw["A_target_nll"])})
    summary={}
    normal=by_mode["control"]
    for mode,rows in by_mode.items():
        assert len(rows)==len(normal)==130
        matched=[];new=[];changes=[]
        for old,v in zip(normal,rows):
            assert (old["case"],old["row"],old["A"])==(v["case"],v["row"],v["A"])
            orig=old["A"]!=old["B"];after=v["A"]!=v["B"]
            if orig and not after:matched.append([v["case"],v["row"]])
            if not orig and after:new.append([v["case"],v["row"]])
            if after:changes.append([v["case"],v["row"],v["A"],v["B"]])
        samples=sum(v["F64_samples"] for case in actual["counterfactual"].values() for v in case[mode]["visits"])
        sq=sum(v["F64_squared_error"] for case in actual["counterfactual"].values() for v in case[mode]["visits"])
        summary[mode]={"rows":len(rows),"mean_TV_to_64":statistics.mean(v["TV"] for v in rows),
             "mean_target_NLL_delta_to_64":statistics.mean(v["nll_change"] for v in rows),
             "total_top_changes_to_64":len(changes),"original_switches_recovered":matched,"additional_switches":new,
             "changes_to_64":changes,"corrected_F64_sample_RMSE":math.sqrt(sq/samples) if samples else None,
             "corrected_F64_sample_count":samples,
             "prefix_elapsed_s":[case[mode]["prefill_ns"]/1e9 for case in actual["counterfactual"].values()],
             "extra_diagnostic_visits_s":[sum(v["elapsed_ns"] for v in case[mode]["visits"])/1e9 for case in actual["counterfactual"].values()]}
    return summary


if __name__=="__main__":
    root=Path(__file__).resolve().parent;j=compare(root)
    (root/"CAUSAL_RESULT.json").write_text(json.dumps(j,indent=2)+"\n")
    for k,v in j.items():
        print(k,v["total_top_changes_to_64"],"recover",v["original_switches_recovered"],"new",v["additional_switches"],
              "TV",round(v["mean_TV_to_64"],5),"local RMSE",v["corrected_F64_sample_RMSE"])
