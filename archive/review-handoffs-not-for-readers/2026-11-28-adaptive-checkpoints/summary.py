"""Small tables; retain every paired session, decision and timing in RESULT and JSONL."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import statistics as st
from analyze import read

def summarize(result):
    out={}
    for mode in ("off","fixed","auto"):
        runs=[r for r in result["runs"] if r["mode"]==mode]
        def med(f):return st.median(f(r) for r in runs)
        summary={"sessions":len(runs),"total_s":med(lambda r:r["total_s"]),"main_s":med(lambda r:r["main_s"]),
                 "setup_s":med(lambda r:r["setup_s"]),"preparation_s":med(lambda r:r["phases"]["preparation"]["wall_s"]),
                 "image_bytes":med(lambda r:r["final"]["image_bytes"]),"initial_image_bytes":runs[0]["initial"]["image_bytes"],
                 "snapshots":med(lambda r:r["final"]["snapshots"]),"initial_snapshots":runs[0]["initial"]["snapshots"],
                 "metadata_before":{k:med(lambda r:r["initial"]["metadata_files"][k]) for k in ("","-wal","-shm")},
                 "metadata_after":{k:med(lambda r:r["final"]["metadata_files"][k]) for k in ("","-wal","-shm")},
                 "median_stats_ms":med(lambda r:r["final"]["checkpoint_policy"]["stats_ms"]),
                 "median_decision_total_ms":med(lambda r:r["final"]["checkpoint_policy"]["decision_ms"])}
        count_keys=("optional_writes","mandatory_writes","optional_bytes","mandatory_bytes","noop","evictions","image_sha","file_fsync","directory_fsync","sql_commit","sql_autocommit","prompt_n","replayed")
        summary["median_session_counts"]={k:med(lambda r:r["counts"].get(k,0)) for k in count_keys}
        summary["tasks"]={}
        for name in ("linear","late","early"):
            last=[r["phases"][name]["turns"][-1] for r in runs]
            states=[r["phases"][name] for r in runs]
            summary["tasks"][name]={
                "session_wall_s":[p["wall_s"] for p in states],"optional_ms":[p["optional_ms"] for p in states],
                "last_first_ms":[t["first_ms"] for t in last],"last_done_ms":[t["done_ms"] for t in last],
                "last_prompt_ms":[t["prompt_ms"] for t in last],"last_cached":[t["cache_n"] for t in last],"last_new":[t["prompt_n"] for t in last],
                "answers":[t["answer"] for t in last], "start_first_ms":[r["phases"][name]["turns"][0]["first_ms"] for r in runs],
                "start_done_ms":[r["phases"][name]["turns"][0]["done_ms"] for r in runs],
                "optional_n":[p["counts"].get("optional_writes",0) for p in states],"image_write_B":[p["counts"].get("optional_bytes",0)+p["counts"].get("mandatory_bytes",0) for p in states],
                "meta_growth":[sum(p["metadata_bytes_delta"].values()) for p in states]}
        b=Counter();c=Counter()
        for r in runs:
            b.update(r["final"]["checkpoint_policy"]["decision_bases"]);c.update(r["final"]["checkpoint_policy"]["outcomes"])
        summary["decision_bases_all"]=dict(b);summary["outcomes_all"]=dict(c)
        out[mode]=summary
    pred={}
    for p in result["actual_cost_comparisons"]:
        pred.setdefault(p["tag"],[]).append(p)
    estimates={}
    for name,samples in pred.items():
        estimates[name]={k:st.median(x["estimate"][k] for x in samples) for k in ("replay","save","restore_anchor","restore_candidate","gain")}
        for target in ("observed_optional_S_ms","observed_prompt_P_ms"):
            xs=[x[target] for x in samples if x[target] is not None];estimates[name][target]=st.median(xs) if xs else None
        estimates[name]["evaluations"]=len(samples);estimates[name]["selected"]=sum(x["accepted"] for x in samples)
        estimates[name]["routes"]=sorted({x["route"] for x in samples})
        for side in ("A","B"):
            matched=[st.median(v["ms"] for v in x["real_R_"+side]) for x in samples if x["real_R_"+side]]
            estimates[name]["matching_actual_R_"+side+"_decisions"]=len(matched)
            estimates[name]["observed_R_"+side+"_ms"]=st.median(matched) if matched else None
    return {"modes":out,"estimates_by_tag":estimates,"paired":result["paired"]}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--check",action="store_true")
    a=p.parse_args();v=summarize(read(a.root/"RESULT.json"));out=a.root/"SUMMARY.json"
    if a.check:assert read(out)==v
    else:out.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
