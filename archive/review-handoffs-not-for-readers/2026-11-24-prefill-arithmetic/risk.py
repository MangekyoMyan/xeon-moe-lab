"""Recompute paired logit and probability differences, keeping workload and shared histories explicit."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def read(p):return json.loads(p.read_bytes())
def stats(values):
    v=sorted(values)
    return {"mean":statistics.mean(v),"median":statistics.median(v),"p95":v[math.ceil(.95*len(v))-1],"max":v[-1]}


def score(root,pairs):
    design=read(root/"precision/FIXTURES.json")
    direct={v["id"]:v["targets"] for v in design["cases"]}
    normal=root/"runs/precision-64-64"
    greedy=read(normal/"GREEDY.json")
    assert len(greedy)==8
    direct.update({"hold-"+n:e["reference_ids"] for n,e in greedy.items()})
    prompts={v["id"]:len(v["ids"]) for v in design["chats"]}
    outputs={}
    def group(name):
        if name.endswith("96") and not name.startswith("hold-"):return "original96"
        if not name.startswith("hold-"):return "controls160"
        return "affected_chats" if 64<=prompts[name[5:]]<128 else "control_chats"
    labeldir=root/"risk";labeldir.mkdir(exist_ok=True)
    for tag,a,b in pairs:
        sub=labeldir/tag;sub.mkdir(exist_ok=True)
        aa=root/"runs"/("precision-"+a);bb=root/"runs"/("precision-"+b)
        assert read(aa/"VERIFIED.json") and read(bb/"VERIFIED.json")
        ga,gb=read(aa/"GREEDY.json"),read(bb/"GREEDY.json")
        grouped={g:[] for g in ("original96","controls160","affected_chats","control_chats")}
        pp={}
        for name,ids in direct.items():
            gold=sub/(name+".ids");gold.write_text("\n".join(map(str,ids))+"\n")
            raw=sub/(name+".tsv")
            subprocess.run([str(root/"precision/score_logits"),str(aa/(name+".f32")),str(bb/(name+".f32")),str(gold),str(raw)],check=True)
            vals=[]
            for j,x in enumerate(csv.DictReader(raw.open(),delimiter="\t")):
                ints={"row","target","exact","A_top","B_top","A_second","B_second","Btop_rank_in_A","Atop_rank_in_B","top5_intersection"}
                v={k:int(w) if k in ints else float(w) for k,w in x.items()};v["case"]=name;assert v["row"]==j and v["target"]==ids[j]
                if name.startswith("hold-"):
                    assert (v["A_top"],v["B_top"])==(ga[name[5:]]["observed_top_ids"][j],gb[name[5:]]["observed_top_ids"][j])
                if tag=="both" and name in ("JA96","PY96"):
                    old=read(root.parent/"normal24-timeline-20261124/PREFIX_QUALITY.json")["cases"][name]["quality"]["per_row"][j]
                    assert (v["A_top"],v["B_top"])==(old["F_argmax"],old["S_argmax"]) and abs(v["A_target_nll"]-old["F_target_nll"])<2e-8 and abs(v["B_target_nll"]-old["S_target_nll"])<2e-8
                vals.append(v)
            assert len(vals)==len(ids)
            grouped[group(name)].extend(vals)
            pp[name]={"rows":len(vals),"identical_rows":sum(v["exact"] for v in vals),"argmax_changes":sum(v["A_top"]!=v["B_top"] for v in vals),
                      "TV":stats([v["TV"] for v in vals]),"mean_target_nll_delta":statistics.mean(v["B_target_nll"]-v["A_target_nll"] for v in vals)}
        gsummary={}
        for g,vals in grouped.items():
            changed=[v for v in vals if v["A_top"]!=v["B_top"]]
            gsummary[g]={"rows":len(vals),"full_rows_byte_equal":sum(v["exact"] for v in vals),"changes":len(changed),
                     "TV":stats([v["TV"] for v in vals]),"JS_nats":stats([v["JS_nats"] for v in vals]),
                     "average_target_nll_delta":statistics.mean(v["B_target_nll"]-v["A_target_nll"] for v in vals),
                     "mean_top5_intersection":statistics.mean(v["top5_intersection"] for v in vals),
                     "changed_positions":changed}
        outputs[tag]={"A":a,"B":b,"grouped":gsummary,"by_case":pp}
        print(tag,[(k,v["rows"],v["full_rows_byte_equal"],v["changes"],round(v["TV"]["mean"],5)) for k,v in gsummary.items()],flush=True)
    return outputs


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    pairs=[("both","64-64","128-128"),("MX_only","64-64","64-128"),("general_only","64-64","128-64"),
           ("MX_after_general","128-64","128-128"),("general_after_MX","64-128","128-128")]
    if len(sys.argv)>1:pairs=[x for x in pairs if x[0] in sys.argv[1:]]
    result=score(root,pairs)
    for k,v in result.items():(root/"risk"/(k+".json")).write_text(json.dumps(v,indent=2)+"\n")
