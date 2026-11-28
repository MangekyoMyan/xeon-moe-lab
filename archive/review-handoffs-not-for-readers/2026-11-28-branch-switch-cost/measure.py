"""Counterbalanced real native / manager trials on one exclusively owned Unix-socket backend."""
import argparse
import contextlib
import itertools
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from study_lib import (R,LAB,FEATURE,STORE,BUDGET,Native,Error,append,asset,blob,digest,engine,follow,
                       image_tokens,pack,read,reset_trial,rows,rpc,window,write)

ns=time.monotonic_ns
CELLS=[("short",False),("long",True),("long",False),("short",True)]
STAGES={"discover":(6,("native","base","base-trace")),
        "confirm":(10,("native","base","new")),"timed":(3,("base-trace","new-trace"))}


@contextlib.contextmanager
def server(version,traced,binding,run,label,store=STORE,special=None):
    code=R/"base-cli" if version=="base" else FEATURE
    cmd=[sys.executable,"-B",str(special or R/"timed_worker.py"),"--store",str(store),"serve","--binding",str(binding)]
    owner=LAB.stat()
    env={**os.environ,"BRANCH_LAB_SOURCE":str(code),"BRANCH_LAB_TIMED":"1" if traced else "0"}
    with (run/"workers.log").open("a") as log:
        proc=subprocess.Popen(cmd,user=owner.st_uid,group=owner.st_gid,env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=log,start_new_session=True)
        try:
            limit=time.monotonic()+180
            while time.monotonic()<limit and proc.poll() is None:
                try:
                    if rpc(store,{"op":"status"})["mode"]=="ready":break
                except (FileNotFoundError,ConnectionRefusedError):
                    pass
                time.sleep(.015)
            else:raise RuntimeError(("manager failed",version,label,proc.poll()))
            yield
        finally:
            if proc.poll() is None:proc.terminate()
            proc.wait(timeout=130)


def action(store,op,**kw):
    return rpc(store,{"op":op,"ticket":rpc(store,{"op":"ticket"})["ticket"],**kw})


def inventory(native):
    return {p.name:p.stat().st_size for p in sorted(native.images.glob("*.slot"))}


def manual(native,case,label):
    p,t,d=(case["points"][k] for k in ("P","T","D"))
    native.restore(p["sha"],pack(p["tokens"]),p["bytes"])
    dirty_ids=native.render(follow(p["messages"],case["dirty_request"]))
    initial=native.generate(dirty_ids,1,lambda _:None)
    assert dirty_ids==d["tokens"] and initial["sampled"]==d["pending"] and initial["timings"]["cache_n"]==len(p["tokens"])
    inv=inventory(native);cur=native.images/("manual-"+label+".slot")
    assert not cur.exists()
    clocks={"start":ns()}
    try:
        clocks["save_begin"]=ns()
        save=native.call("POST","/slots/0?action=save",{"filename":cur.name})
        clocks["save_end"]=ns()
        clocks["restore_begin"]=ns()
        restore=native.call("POST","/slots/0?action=restore",{"filename":blob(t)})
        clocks["switch_end"]=ns()
        clocks["next_begin"]=ns()
        ids=native.render(follow(t["messages"],case["next_request"]))
        assert ids==case["next_ids"]
        first=[]
        nxt=native.generate(ids,8,lambda _:first.append(ns()))
        clocks["first"]=first[0];clocks["done"]=ns()
        assert save["n_saved"]==len(dirty_ids) and save["n_written"]==d["bytes"]
        assert restore["n_restored"]==len(t["tokens"]) and restore["n_read"]==t["bytes"]
        assert digest(cur)==d["sha"] and image_tokens(cur,native.limit)==pack(dirty_ids)
        after=inventory(native)
        full_sha=digest(native.images/blob(t))
        # Settle this path's scratch writes only after the reported first-text and completion boundaries.
        with cur.open("rb") as f:os.fsync(f.fileno())
    finally:
        cur.unlink(missing_ok=True)
    return {"clocks":clocks,"initial_inventory":inv,"final_inventory":after,"images_evicted":0,
            "outgoing":{"id":None,"sha":d["sha"],"bytes":save["n_written"],"processed":save["n_saved"]},
            "target_id":t["id"],"target_sha":full_sha,"replayed":0,"exact_head":True,"receipt":"native full image, returned n_restored/n_read",
            "native_save":save,"native_restore":restore,"next":nxt,"source_cache_n":initial["timings"]["cache_n"],
            "source_new":initial["timings"]["prompt_n"]}


def managed(native,case,label,version,traced,binding,run):
    p,t,d=(case["points"][k] for k in ("P","T","D"))
    with server(version,traced,binding,run,label):
        dirty=action(STORE,"turn",text=case["dirty_request"],max_tokens=1)
        assert dirty["processed"]==len(d["tokens"]) and dirty["pending_sampled_ids"]==d["pending"] and dirty["timings"]["cache_n"]==len(p["tokens"])
        initial=rpc(STORE,{"op":"status"});inv=inventory(native)
        assert initial["budget_bytes"]==BUDGET and initial["image_bytes"]==sum(inv.values())
        clocks={"start":ns()}
        ticket=rpc(STORE,{"op":"ticket"})["ticket"]
        clocks["ticket_end"]=ns()
        req={"op":"checkout","name":"target","ticket":ticket}
        if traced:req["_measure"]=label
        clocks["checkout_send"]=ns()
        out=rpc(STORE,req)
        clocks["switch_end"]=ns()
        clocks["next_begin"]=ns()
        ticket=rpc(STORE,{"op":"ticket"})["ticket"]
        clocks["next_ticket_end"]=ns()
        first=[]
        nxt=rpc(STORE,{"op":"turn","ticket":ticket,"text":case["next_request"],"max_tokens":8},lambda _:first.append(ns()))
        clocks["first"]=first[0];clocks["done"]=ns()
        assert out["from_snapshot"]==t["id"] and out["exact_head"] and out["replayed"]==0
        outgoing=rpc(STORE,{"op":"inspect","kind":"snapshot","name":out["outgoing_snapshot"]})
        assert outgoing["image_status"]=="available" and outgoing["image_sha256"]==d["sha"] and outgoing["processed"]==len(d["tokens"])
        after=inventory(native);end=rpc(STORE,{"op":"status"})
        assert end["mode"]=="ready" and end["image_bytes"]==sum(after.values()) and set(inv)<=set(after)
        assert end["budget_bytes"]==BUDGET and outgoing["image_bytes"]==d["bytes"]
        result={"clocks":clocks,"initial_inventory":inv,"final_inventory":after,"images_evicted":len(set(inv)-set(after)),
                "outgoing":{"id":outgoing["id"],"sha":outgoing["image_sha256"],"bytes":outgoing["image_bytes"],"processed":outgoing["processed"]},
                "target_id":out["from_snapshot"],"target_sha":digest(native.images/blob(t)),"replayed":out["replayed"],"exact_head":out["exact_head"],
                "receipt":"checkout plus validated immutable departure","next":nxt,"source_cache_n":dirty["timings"]["cache_n"],
                "source_new":dirty["timings"]["prompt_n"],"managed_budget_resident":[end["budget_bytes"],end["image_bytes"]]}
        if traced:
            result["trace"]=rpc(STORE,{"op":"read-checkout-trace","label":label})
            assert result["trace"]["synchronous"]==2 and result["trace"]["journal_mode"]=="wal"
    return result


def design():
    result={}
    for stage,(n,variants) in STAGES.items():
        sequence=[]
        if len(variants)==3:
            a,b,c=variants
            orders=[(a,b,c),(c,b,a),(b,c,a),(a,c,b),(c,a,b),(b,a,c)]
        else:
            orders=list(itertools.permutations(variants))
        for j in range(n):
            for k in range(len(CELLS)):
                cell=(k+j)%len(CELLS);size,dup=CELLS[cell]
                index=((j+cell)%len(orders)) if len(variants)==2 else ((j+2*cell)%6)
                for order,v in enumerate(orders[index]):
                    sequence.append({"stage":stage,"pair":j,"size":size,"duplicate":dup,"order":order,"variant":v})
        result[stage]=sequence
    return result


def run_stage(stage,run,binding,planned=None):
    native=Native(binding)
    cases={s:read(R/"cases"/(s+".json")) for s in ("short","long")}
    dest=R/"evidence"/(stage+".jsonl")
    saved=rows(dest) if dest.exists() else []
    if planned is None:
        planned=read(R/"DESIGN.json")["order"][stage]
    assert all({k:v[k] for k in z}==z for z,v in zip(planned,saved)) and len(saved)<=len(planned)
    for i,z in enumerate(planned):
        if i<len(saved):continue
        case=cases[z["size"]];version=z["variant"];dup=z["duplicate"]
        reset_trial(native,case,dup)
        label=f"{stage}-{i}"
        if version=="native":
            row=manual(native,case,label)
        else:
            row=managed(native,case,label,version.split("-")[0],version.endswith("-trace"),binding,run)
        nxt=row["next"]; tm=nxt["timings"]
        assert nxt["content"]==case["answer"] and tm["prompt_n"]==len(case["next_ids"])-len(case["points"]["T"]["tokens"])
        assert tm["cache_n"]==len(case["points"]["T"]["tokens"])
        assert (blob(case["points"]["D"]) in row["initial_inventory"])==dup
        assert row["target_sha"]==case["points"]["T"]["sha"] and sum(row["final_inventory"].values())<=BUDGET
        row["id"]=label;row["budget"]=BUDGET;row["run"]=run.name
        append(dest,{**z,**row})
        if i%12==11:print(stage,i+1,"/",len(planned),flush=True)
    assert len(rows(dest))==len(planned)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["design","discover","confirm","timed","validation","after"])
    p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    if a.stage=="design":
        write(R/"DESIGN.json",{"order":design(),"clock":"integer same-host monotonic_ns, first nonempty content, include ticket on managed path",
              "fixed_mark_baseline":"28b056e8d7b0926dcd0bad25015447549cfd5cbe","native":"same fixed private HTTP adapter, regular CPU/CUDA/llama/base on both timing paths",
              "isolation":"One protected native process per run. Named synthetic catalog is closed/reset outside timing. Same immutable origin/target images and logged dirty and next messages. Both paths warm page cache; no drop-caches. No image eviction.",
              "sqlite":"WAL, synchronous=FULL, independently checked in each traced worker","budget":BUDGET,
              "diagnosis":"6 interleaved native, base plain and base-with-spans triples per length and image presence, then select one change. Confirm with 10 balanced native/base/candidate triples and 3 more paired span observations for each length and presence. Single-run native output gets a file+directory settlement outside its timed interval.",
              "metrics":"Primary paired baseline-minus-candidate milliseconds to checkout and to next body; report every replicate and median absolute deviation. Do not equate SQLite COMMIT with one physical disk flush."})
        return
    assert read(R/"PREPARED.json")["complete"] and read(R/"DESIGN.json")["order"]==design()
    if a.stage in ("confirm","timed","after"):assert (R/"DECISION.json").is_file()
    def finish(signum,frame):raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,finish)
    stages=["confirm","timed"] if a.stage=="after" else [a.stage]
    with window(R,"BRANCH-COST-"+a.stage):
        for stage in stages:
            run=R/"runs"/f"{stage}-{a.attempt}"
            assert not run.exists()
            with engine(R,STORE,run) as binding:
                if stage=="validation":raise RuntimeError("use the independent numerical validation controller")
                run_stage(stage,run,binding)
            write(run/"DONE.json",{"complete":True,"stage":stage})


if __name__=="__main__":
    main()
