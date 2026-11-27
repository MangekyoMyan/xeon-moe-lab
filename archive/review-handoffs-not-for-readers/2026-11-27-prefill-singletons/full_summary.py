"""Replay the complete model's paired prefill comparisons."""
import json
from pathlib import Path
import statistics as st

def load(p):return json.loads(Path(p).read_bytes())
def score(root,run):
    r=load(root/"runs"/run/"RUN.json");assert r["passed"]
    result={"driver_run":run,"cases":{}}
    for case in ("JA512","PY512"):
        qual=[j for j in r["quality"] if j["case"]==case]
        assert len(qual)==6 and all(j["same_as_reference_when_compared"] for j in qual)
        raw=[j for j in r["timings"] if j["case"]==case]
        assert len(raw)==8
        pairs=[];arms={}
        for a in ("O","P"):
            rows=[j for j in raw if j["arm"]==a]
            durations=[(j["end_ns"]-j["start_ns"])/1e9 for j in rows]
            arms[a]={"mean_s":st.mean(durations),"median_s":st.median(durations),"range_s":[min(durations),max(durations)],"all_s":durations}
        for pair in range(4):
            rows=[j for j in raw if j["pair"]==pair];assert [x["order"] for x in rows]==[0,1]
            d={j["arm"]:(j["end_ns"]-j["start_ns"])/1e9 for j in rows}
            assert set(d)=={"O","P"}
            pairs.append({"pair":pair,"order":"".join(j["arm"] for j in rows),"P_minus_O_s":d["P"]-d["O"]})
        free=[j for j in r["free_greedy"] if j["case"]==case]
        assert len(free)==2
        result["cases"][case]={"arms":arms,"pairs":pairs,"mean_delta_s":st.mean(p["P_minus_O_s"] for p in pairs),
            "time_change_percent":100*(arms["P"]["mean_s"]/arms["O"]["mean_s"]-1),"pairs_shorter":sum(p["P_minus_O_s"]<0 for p in pairs),
            "by_order_delta_s":{o:st.mean(p["P_minus_O_s"] for p in pairs if p["order"]==o) for o in ("OP","PO")},
            "greedy_ids_identical":free[0]["greedy"]==free[1]["greedy"],
            "greedy_text":{u["arm"]:u["text"] for u in free},
            "GPU_scratch_bytes_per_layer":{str(x["layer"]):x["scratch_GPU_B"] for j in raw if j["arm"]=="P" for x in j["receipt"]["layers"]},
            "min_GPU_free_B_during_completed_phase":min(x["free_gpu_B"] for j in raw if j["arm"]=="P" for x in j["receipt"]["layers"])}
    return result
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("run");p.add_argument("--check",action="store_true");a=p.parse_args()
    here=Path(__file__).resolve().parent;d=score(here,a.run);path=here/"WHOLE_MODEL.json"
    if a.check:assert d==load(path)
    else:path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n")
    for case,v in d["cases"].items():print(case,v["arms"]["O"]["mean_s"],v["arms"]["P"]["mean_s"],v["pairs_shorter"],v["time_change_percent"])
