"""A single bounded extension only when replay/save decisions and elapsed order disagree."""
import argparse
import json
import statistics as st
from analyze import ROOT as R, read, lines as rows, verify_session

def write(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")

def select():
    result=read(R/"RESULT.json");selected=[];reasons={}
    for name in ("linear","late","early"):
        records=result["runs"];why=[]
        for mode in ("fixed","auto"):
            deltas=result["paired"][name]["paired_seconds_saved"][mode]
            s=[r["phases"][name] for r in records if r["mode"]==mode]
            off=[r["phases"][name] for r in records if r["mode"]=="off"]
            changed=any(x["counts"].get("optional_writes",0)>0 or x["counts"]["prompt_n"]!=y["counts"]["prompt_n"] for x,y in zip(s,off))
            if changed and min(deltas)<0<max(deltas):why.append(mode)
        if why:selected.append(name)
        reasons[name]=why
    return {"selected":selected,"triggering_modes":reasons,"through":selected[-1] if selected else None,
            "order":[[rep,m] for rep,seq in [(3,("auto","off","fixed")),(4,("fixed","off","auto"))] for m in seq] if selected else []}

def verify(root):
    plan=read(root/"SUPPLEMENT.json");res=read(root/"RESULT.json");out={}
    extra=rows(root/"evidence/extra.jsonl") if plan["order"] else []
    assert [[r["rep"],r["mode"]] for r in extra]==plan["order"]
    jobs=read(root/"fixtures/workload.json");anchors={k:read(root/"fixtures"/(k+"-anchor.json")) for k in ("short","long")}
    for r in extra:verify_session(root,r,jobs,anchors)
    for name in plan["selected"]:
        wall={m:list(v) for m,v in res["paired"][name]["raw_seconds"].items()}
        for r in extra:
            ph=next(p for p in r["phases"] if p["name"]==name)
            wall[r["mode"]].append((ph["end"]-ph["start"])/1e9)
        delta={m:[o-x for o,x in zip(wall["off"],wall[m],strict=True)] for m in ("fixed","auto")}
        out[name]={"times_s":wall,"paired_s":delta,"paired_medians":{m:st.median(v) for m,v in delta.items()}}
    return out

def measure_extra(run,binding,design):
    from trial import session,append,Native
    out=R/"evidence/extra.jsonl";done=rows(out) if out.exists() else []
    assert [[q["rep"],q["mode"]] for q in done]==design["order"][:len(done)]
    n=Native(binding)
    for rep,m in design["order"][len(done):]:append(out,session(n,binding,run,rep,m,through=design["through"]))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["rule","plan","run","check"]);p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    if a.action=="rule":
        dest=R/"SUPPLEMENT_RULE.json";assert not dest.exists()
        write(dest,{"rule":"If any on/off phase wall-time pairs among the three original repeats cross zero AND the policy actually wrote an optional image or changed prompt_n in that phase, add exactly two matched repeats. Do not extend for noise between paths with identical actions. Run only up to the last selected workload to preserve actual training/history; summarize extensions only for selected workloads. Keep original trio and full raw repeats. No threshold changes.","timing_of_rule":"Specified with the original general variability clause, detailed here after the first matched group and before the full three groups."})
    elif a.action=="plan":
        assert (R/"SUPPLEMENT_RULE.json").exists();write(R/"SUPPLEMENT.json",select());print(select()["selected"])
    elif a.action=="check":
        assert verify(R)==read(R/"EXTRA.json")
    else:
        from trial import STORE,engine,window,session,append,Native
        design=read(R/"SUPPLEMENT.json")
        if design["order"]:
            run=R/"runs"/f"supplement-{a.attempt}";assert not run.exists()
            with window(R,"ADAPTIVE-SUPPLEMENT"),engine(R,STORE,run) as binding:
                measure_extra(run,binding,design)
            write(run/"DONE.json",{"complete":True})
        write(R/"EXTRA.json",verify(R))
