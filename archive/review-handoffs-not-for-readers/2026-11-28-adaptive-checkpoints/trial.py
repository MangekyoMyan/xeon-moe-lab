"""Controlled OFA comparison using the real patched CLI on the established private slot."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time
import traceback
R=Path(__file__).resolve().parent
LAB=R.parent
F=R/"checkout/current/experiments/state-branches"
sys.path.insert(0,str(F))
from branchctl import rpc
from storage import Store,uid
from native import Native,encode,pack,digest,image_tokens,sync_dir
from x9ram_pilot import engine,manager,window,private
sys.path.insert(0,str(R))
STORE=R/"trial-store"
BUDGET=1024**3
MODES=("off","fixed","auto")
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def append(p,v):
    with Path(p).open("a") as f:f.write(encode(v)+"\n")
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def anchors():return {name:read(R/"fixtures"/(name+"-anchor.json")) for name in ("short","long")}
def design():
    jobs=read(R/"fixtures/workload.json")
    order=[(rep,MODES[(rep+i)%3]) for rep in range(3) for i in range(3)]
    return {"sequence":[list(v) for v in order],"interval":1024,"budget_bytes":1024**3,
       "phases":{"linear":["linear.B","linear.C"],"preparation":[f"checkout/turn/mark practice{i}" for i in (1,2,3)],
                 "late":["checkout late","late.B","late.C","late.edit"],"early":["checkout early","early.B","early.C","early.edit"]},
       "scope":"One independent, reinitialized manager/catalog per (rep,mode); three distinct initial branches per session. First linear workload is untrained and never resumed; shared preparation counted once before both edited workloads. Learned RAM observations carry through that session; each new replicate/arm starts empty. Same original full native slot images are reused without rebuilding long prefixes.",
       "inputs":"Fixed before measurements with deployed template and tokenizer. The B text requests OK, following messages contain the fixed assistant text OK. Late edit changes the following user receipt number. Early edit changes B's user message before B's saved boundary.",
       "costs":"Server monotonic span includes request/ticket and streaming to final durable receipt, and any checkpoint. First-text timestamp is receipt of first nonempty delta. A phase wall runs from first client request to last receipt. Catalog inventories sampled outside phase clocks. Session wall and sum of operation clocks are both reported. Instrumentation counts sync/commits/image hashes, validated native prompt_ms and save_active outcomes in RAM. No extra model evaluations.",
       "freeze_sha256":digest(R/"fixtures/workload.json"),"limits":{"gpu_w":150,"fan_percent":70,"offload":[128,128],"threads":[24,24],"kv":"f16","context":131072,"ubatch":1024},
       "conditional":"Three paired sessions. If the on-versus-off sign is mixed and could change the stated workload conclusion, add paired checks for that workload, report separately; no N or coefficient tuning."}

def make_template(n):
    path=R/"template";path.mkdir()
    s=Store(path,n,BUDGET);roots=anchors();jobs=read(R/"fixtures/workload.json")
    with s.tx():
        for label,z in roots.items():
            assert image_tokens(LAB/"state-branches-switch-20261128/assets"/(z["sha"]+".slot"),n.limit)==pack(z["tokens"])
            s.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)",(z["id"],None,n.compat,pack(z["tokens"]),pack(z["pending"]),encode(z["messages"]),z["sha"],z["bytes"],time.time()))
            s.db.execute("INSERT INTO blobs VALUES(?,?,1,?)",(z["sha"],z["bytes"],time.time()))
            s.db.execute("INSERT INTO marks VALUES(?,?,1)",("root-"+label,z["id"]))
        for name,job in jobs.items():
            z=roots[job["root"]];v=(z["id"],z["id"],z["id"],pack(z["tokens"]),pack(z["pending"]),encode(z["messages"]))
            if name=="linear":
                s.db.execute("UPDATE branches SET name='linear',origin=?,base=?,head=?,tokens=?,pending=?,messages=? WHERE name='main'",v)
            else:
                s.db.execute("INSERT INTO branches VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),name,*v,0,0,time.time(),0))
        s.state("ready")
    s.db.close()

def reset(n):
    assert STORE==R/"trial-store" and not (STORE/"manager.sock").exists()
    roots=anchors();wanted={z["sha"]+".slot" for z in roots.values()}
    for p in n.images.iterdir():
        if p.name not in wanted:
            assert p.name.endswith(".slot");p.unlink()
    for z in roots.values():
        fn=z["sha"]+".slot";src=LAB/"state-branches-switch-20261128/assets"/fn;dest=n.images/fn
        if not dest.exists():os.link(src,dest)
        assert digest(dest)==z["sha"]
    with (STORE/"manager.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=STORE/"branches.sqlite3"
        if db.exists():
            con=sqlite3.connect(db)
            try:assert con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]==0
            finally:con.close()
            wal=STORE/"branches.sqlite3-wal"
            assert not wal.exists() or wal.stat().st_size==0
            wal.unlink(missing_ok=True);(STORE/"branches.sqlite3-shm").unlink(missing_ok=True)
        temp=STORE/".fresh";shutil.copyfile(R/"template/branches.sqlite3",temp);private(temp,False)
        with temp.open("rb") as f:os.fsync(f.fileno())
        temp.replace(db);sync_dir(STORE);sync_dir(n.images)

def session(n,binding,run,rep,mode,through="early"):
    reset(n);jobs=read(R/"fixtures/workload.json")
    log=[];phases=[];first_session=None;last_session=None
    def inventory():return rpc(STORE,{"op":"measure-inventory"})
    def call(tag,op,**args):
        nonlocal first_session,last_session
        start=time.monotonic_ns();ticket=rpc(STORE,{"op":"ticket"})["ticket"];seen=[]
        result=rpc(STORE,{"op":op,"ticket":ticket,"_measured":True,**args},lambda _:seen.append(time.monotonic_ns()))
        end=time.monotonic_ns()
        if first_session is None:first_session=start
        last_session=end
        log.append({"tag":tag,"op":op,"start":start,"end":end,"first":seen[0] if seen else None,"result":result})
        return result
    def turn(name,step):
        task=jobs[name][step]
        v=call(name+"."+step,"turn",messages=task["messages"],max_tokens=task["max_tokens"])
        assert v["_measurements"]["input_sha256"]==hashlib.sha256(pack(task["ids"])).hexdigest()
        assert v["turn_committed"] and v["checkpoint"]["status"] not in ("unknown","failed","saved_cleanup_warning"),v["checkpoint"]
        assert v["processed"]==len(task["ids"])+v["sampled_count"]-1
        assert v["timings"]["prompt_n"]+v["timings"]["cache_n"]==len(task["ids"])
        return v
    def group(name,ops):
        before=inventory();i=len(log)
        try:ops()
        except BaseException:
            write(run/f"partial-{rep}-{mode}.json",{"phase":name,"completed_rows":log,"partial_phases":phases})
            raise
        after=inventory()
        phases.append({"name":name,"before":before,"after":after,"start":log[i]["start"],"end":log[-1]["end"],
                       "from_row":i,"to_row":len(log)})
        print(rep,mode,name,len(log)-i,"completed",flush=True)
    def practice():
        for i in (1,2,3):
            name=f"practice{i}"
            call(name+".checkout","checkout",name=name)
            turn(name,"B");call(name+".mark","mark",name=name)
    with manager(STORE,binding,run/f"{rep}-{mode}.log",R/"audit_worker.py",checkpoint_mode=mode,checkpoint_interval=1024):
        initial=inventory();assert initial["checkpoint_policy"]["mode"]==mode
        group("linear",lambda:[turn("linear",k) for k in ("B","C")])
        if through!="linear":
            group("preparation",practice)
            for name in ("late","early"):
                group("enter-"+name,lambda:call(name+".checkout","checkout",name=name))
                group(name,lambda:[turn(name,k) for k in ("B","C","edit")])
                if name==through:break
        final=inventory()
        # Retain only two preselected synthetic images for the subsequent controlled numeric test.
        if rep==0 and mode=="auto":
            out=R/"reference";out.mkdir(exist_ok=True)
            for name in ("linear","late"):
                j=next(row["result"] for row in log if row["tag"]==name+".B")
                sid=j["checkpoint"]["snapshot_id"]
                if sid and j["checkpoint"]["status"]=="saved":
                    s=rpc(STORE,{"op":"inspect","kind":"snapshot","name":sid})
                    src=n.images/(s["image_sha256"]+".slot");shutil.copyfile(src,out/(name+".slot"))
                    assert digest(out/(name+".slot"))==s["image_sha256"]
                    write(out/(name+".json"),{"image":s,"tokens":jobs[name]["B"]["ids"],"pending":j["pending_sampled_ids"],"messages":jobs[name]["B"]["messages"]+[{"role":"assistant","content":j["content"]}]})
    return {"rep":rep,"mode":mode,"rows":log,"phases":phases,"first_ns":first_session,"last_ns":last_session,"initial":initial,"final":final}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["design","preflight","run"]);p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    if a.action=="design":
        assert not (R/"DESIGN.json").exists();write(R/"DESIGN.json",design());sys.exit()
    if a.action=="preflight":
        from types import SimpleNamespace
        assert (R/"template/branches.sqlite3").exists()
        reset(SimpleNamespace(images=STORE/"images"))
        conn=sqlite3.connect(f"file:{STORE/'branches.sqlite3'}?mode=ro&immutable=1",uri=True)
        try:assert conn.execute("SELECT COUNT(*) FROM branches").fetchone()[0]==6
        finally:conn.close()
        print("Actual scratch catalog reset and ownership preflight passed; no model call")
        sys.exit()
    assert design()==read(R/"DESIGN.json") and os.geteuid()==0
    run=R/"runs"/f"comparison-{a.attempt}";assert not run.exists()
    out=R/"evidence/sessions.jsonl";old=rows(out) if out.exists() else []
    planned=design()["sequence"];assert [[v["rep"],v["mode"]] for v in old]==planned[:len(old)]
    def stop(sig,frame):raise KeyboardInterrupt(sig)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    with window(R,"ADAPTIVE-COMPARISON"),engine(R,STORE,run) as binding:
        native=Native(binding)
        try:
            if not (R/"template/branches.sqlite3").exists():make_template(native)
            for rep,mode in planned[len(old):]:
                v=session(native,binding,run,rep,mode);append(out,v)
                print(rep,mode,round((v["last_ns"]-v["first_ns"])/1e9,2),v["final"]["checkpoint_policy"]["decision_bases"],flush=True)
        except BaseException:
            (run/"FAILURE.txt").write_text(traceback.format_exc());traceback.print_exc();raise
    write(run/"DONE.json",{"complete":True})
