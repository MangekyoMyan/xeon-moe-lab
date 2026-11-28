"""Confirm a single chosen optimization, with a prespecified additional-pair rule."""
import argparse
from collections import defaultdict
from pathlib import Path
import signal
from analyze import analyze,describe
from measure import CELLS,run_stage
from study_lib import R,STORE,engine,window,read,write,rows


def continuation(summary):
    cells=[];why={}
    for size,duplicate in CELLS:
        key=size+("-resident" if duplicate else "-fresh")
        z=summary[key]["paired"]["base minus new"]["switch_ms"]
        ambiguous=(3<=z["positive"]<=7 or abs(z["median"])<=z["mad"])
        why[key]={"additional":ambiguous,"paired_median_ms":z["median"],"paired_mad_ms":z["mad"],"positive_of_10":z["positive"]}
        if ambiguous:cells.append((size,duplicate))
    order=[]
    for k in range(10,20):
        for j in range(len(cells)):
            cell=(k+j)%len(cells);size,dup=cells[cell]
            variants=("base","new") if (k+cell)%2==0 else ("new","base")
            for at,v in enumerate(variants):
                order.append({"pair":k,"stage":"extend","size":size,"duplicate":dup,"order":at,"variant":v})
    return {"reason":why,"order":order}


def paired_total(root):
    data=rows(root/"evidence/confirm.jsonl")
    if (root/"evidence/extend.jsonl").exists():data+=rows(root/"evidence/extend.jsonl")
    out={}
    for size,dup in CELLS:
        pairs=defaultdict(dict)
        for v in data:
            if v["variant"] not in ("base","new") or (v["size"],v["duplicate"])!=(size,dup):continue
            pairs[v["pair"]][v["variant"]]=v
        z={}
        for clock in ("switch_end","first","done"):
            series={k:[(pair[k]["clocks"][clock]-pair[k]["clocks"]["start"])/1e6 for _,pair in sorted(pairs.items())] for k in ("base","new")}
            delta=[a-b for a,b in zip(series["base"],series["new"])]
            z[clock]={"base":describe(series["base"]),"new":describe(series["new"]),
                      "base_minus_new":{**describe(delta),"shorter":sum(v>0 for v in delta),"raw":delta}}
        out[size+("-resident" if dup else "-fresh")]=z
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["plan","run","summarize","check"])
    p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    if a.action=="plan":
        write(R/"CONFIRM_PLAN.json",{"primary_pairs_per_cell":10,"order_source":"DESIGN.json",
          "repeat_rule":"If 3 to 7 of 10 candidate switch times are faster inclusive, or abs(paired median) <= paired MAD, add 10 plain base/candidate pairs for that cell. Record primary and combined outcomes; do not retry until positive.",
          "timed_pairs_per_cell":3,"clock":"start of ticket lookup to checkout receipt; next-body time also stored"})
        return
    if a.action in ("summarize","check"):
        x=paired_total(R)
        if a.action=="check":assert x==read(R/"PAIRED.json")
        else:write(R/"PAIRED.json",x)
        return
    assert (R/"DECISION.json").is_file() and read(R/"CONFIRM_PLAN.json")["primary_pairs_per_cell"]==10
    run=R/"runs"/("confirmall-"+str(a.attempt));assert not run.exists()
    def stop(signum,frame):raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    with window(R,"BRANCH-COST-CONFIRM"),engine(R,STORE,run) as binding:
        run_stage("confirm",run,binding)
        primary=analyze(R,"confirm");write(R/"CONFIRM.json",primary)
        extra=continuation(primary);write(R/"EXTEND_PLAN.json",extra)
        if extra["order"]:
            print("Additional paired comparisons:",{k:v["positive_of_10"] for k,v in extra["reason"].items() if v["additional"]},flush=True)
            run_stage("extend",run,binding,extra["order"]);write(R/"EXTEND.json",analyze(R,"extend"))
        run_stage("timed",run,binding)
        write(R/"TIMED.json",analyze(R,"timed"));write(R/"PAIRED.json",paired_total(R))
    write(run/"DONE.json",{"stages":["confirm","conditional extend","timed"],"complete":True})

if __name__=="__main__":main()
