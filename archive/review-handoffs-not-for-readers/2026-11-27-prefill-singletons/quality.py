"""Use the prior full-vocabulary scorer. Only the first 16 rows have a scored reference target."""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_bytes())
def analyze(label):
    audit=HERE/"runs/full-model-check"
    audit_data=load(audit/"RUN.json")
    root=HERE/"runs"/label;record=load(root/"RUN.json")
    integrated=label.startswith("live-two-")
    large=label.startswith("live-large-")
    fixtures={c["id"]:c for c in load(HERE/"large/CASES.json")["cases"]} if large else {}
    dest=HERE/"scores"/label;dest.mkdir(exist_ok=True,parents=True)
    scorer=HERE.parent/"worker-balance-20261124/precision/score_logits"
    summary={}
    for name in (("JA1024","PY1024") if large else ("JA512","PY512")):
        targets=fixtures[name]["forced"] if large else audit_data["forced_inputs"][name]
        assert len(targets)==16
        targetfile=dest/(name+"-targets.txt");targetfile.write_text("\n".join(map(str,[*targets,0]))+"\n")
        ref="O" if integrated or large else "plain"
        for arm in (("GP","P","S") if large else ("GP","P4","P24","P","S") if integrated else ("P4","P24")):
            out=dest/(name+"-"+arm+".tsv")
            subprocess.run([str(scorer),str(root/(name+"-"+ref+".f32")),str(root/(name+"-"+arm+".f32")),str(targetfile),str(out)],check=True)
            with out.open() as f:rows=list(csv.DictReader(f,delimiter="\t"))
            assert len(rows)==17
            first=rows[:16];tv=[float(x["TV"]) for x in rows]
            summary[name+"-"+arm]={"positions":len(rows),"top1_changes":[int(x["row"]) for x in rows if x["A_top"]!=x["B_top"]],
                "all_logits_byte_equal_rows":sum(int(x["exact"]) for x in rows),"mean_TV":statistics.mean(tv),"max_TV":max(tv),
                "targets_scored":16,"native_target_hits":sum(x["A_top"]==x["target"] for x in first),
                "candidate_target_hits":sum(x["B_top"]==x["target"] for x in first),
                "mean_target_NLL_change":statistics.mean(float(x["B_target_nll"])-float(x["A_target_nll"]) for x in first)}
        if integrated:
            free=[r for r in record["free_greedy"] if r["case"]==name]
            summary[name+"-free"]={r["arm"]:r["greedy"] for r in free}
        elif not large:summary[name+"-free"]={r["phase"]:r["greedy_ids"] for r in record["audit"] if r["case"]==name and r["phase"]!="boundary"}
    (dest/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");a=p.parse_args()
    analyze(a.label)
