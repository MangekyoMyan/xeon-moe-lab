"""Restore genuine optional checkpoints; verify numerical rows and a committed uncertain turn."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from trial import R,LAB,F,BUDGET,anchors,read,write,append
from native import Native,Error,encode,pack,unpack,digest,image_tokens
from branchctl import rpc
from storage import Store,uid
from x9ram_pilot import private,engine,manager,window

VS=R/"entry/user-store"
def act(store,op,**kw):
    return rpc(store,{"op":op,"ticket":rpc(store,{"op":"ticket"})["ticket"],**kw})
def phase(run,name):
    p=run/"trace/phase.new";p.write_text(name+"\n");p.replace(run/"trace/phase")
def catalog(store,branch=None):
    con=sqlite3.connect("file:"+str(store/"branches.sqlite3")+"?mode=ro",uri=True);con.row_factory=sqlite3.Row
    try:
        b=con.execute("SELECT * FROM branches WHERE id=(SELECT active FROM control)").fetchone() if branch is None else con.execute("SELECT * FROM branches WHERE name=?",(branch,)).fetchone()
        return {"id":b["id"],"name":b["name"],"head":b["head"],"base":b["base"],"processed":len(b["tokens"])//4,"tokens":unpack(b["tokens"]),
                "pending":unpack(b["pending"]),"messages":json.loads(b["messages"]),"generation":b["gen"]}
    finally:con.close()
def install(n,point,src):
    dest=n.images/(point["sha"]+".slot")
    assert digest(src)==point["sha"] and image_tokens(src,n.limit)==pack(point["tokens"])
    if not dest.exists():
        shutil.copyfile(src,dest);dest.chmod(0o444)


def prepare(n):
    roots=anchors();jobs=read(R/"fixtures/workload.json")
    p=roots["short"];points={"original":p};install(n,p,LAB/"state-branches-switch-20261128/assets"/(p["sha"]+".slot"))
    cases=[]
    for name in ("linear","late"):
        meta=R/"reference"/(name+".json")
        if not meta.exists():continue
        j=read(meta);s=j["image"]
        z={"id":s["id"],"sha":s["image_sha256"],"bytes":s["image_bytes"],"tokens":j["tokens"],"pending":j["pending"],"messages":j["messages"]}
        install(n,z,R/"reference"/(name+".slot"));points[name]=z
        for kind in ("real","129"):
            msg=[dict(m) for m in jobs[name]["C"]["messages"]]
            if kind=="129":
                def render(k):
                    return n.render(msg[:-1]+[{"role":"user","content":msg[-1]["content"]+" x"*k}])
                lo,hi=0,160
                while lo<hi:
                    mid=(lo+hi)//2
                    if len(render(mid))<len(z["tokens"])+129:lo=mid+1
                    else:hi=mid
                msg[-1]["content"]+=" x"*lo
            ids=n.render(msg);assert ids[:len(z["tokens"])]==z["tokens"]
            if kind=="129":assert len(ids)-len(z["tokens"])==129
            cases.append({"name":name+"-"+kind,"anchor":name,"messages":msg,"ids":ids,"limit":4})
    assert cases,"no optional validation image"
    s=Store(VS,n,1024**3)
    with s.tx():
        for name,z in points.items():
            s.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)",(z["id"],None,n.compat,pack(z["tokens"]),pack(z["pending"]),encode(z["messages"]),z["sha"],z["bytes"],0))
            s.db.execute("INSERT OR IGNORE INTO blobs VALUES(?,?,1,0)",(z["sha"],z["bytes"]))
            s.db.execute("INSERT INTO marks VALUES(?,?,1)",("shared-"+name,z["id"]))
        s.db.execute("UPDATE branches SET head=?,base=?,tokens=?,pending=?,messages=? WHERE name='main'",
                     (p["id"],p["id"],pack(p["tokens"]),pack(p["pending"]),encode(p["messages"])))
    s.db.close();private(VS/"branches.sqlite3",False)
    spec={"points":points,"cases":cases};write(R/"evidence/numeric-fixtures.json",spec)
    return spec


def candidate(run,binding):
    n=Native(binding);spec=prepare(n);answers=[]
    for mode in ("off","fixed"):
        with manager(VS,binding,run/(mode+".log"),checkpoint_mode=mode,checkpoint_interval=8):
            for case in spec["cases"]:
                name=mode+"-"+case["name"];base=spec["points"][case["anchor"]]
                act(VS,"fork",name=name,**{"from":"shared-"+case["anchor"]})
                chosen=act(VS,"checkout",name=name);assert chosen["exact_head"] and chosen["replayed"]==0
                phase(run,name)
                try:r=act(VS,"turn",messages=case["messages"],max_tokens=case["limit"])
                finally:phase(run,"idle")
                assert r["timings"]["cache_n"]==len(base["tokens"])
                assert r["checkpoint"]["status"]==("saved" if mode=="fixed" else "disabled")
                answers.append({"name":name,"response":r})
    write(R/"evidence/candidate-numeric.json",answers)
    loss_case(run,binding)


def loss_case(run,binding):
    n=Native(binding)
    with manager(VS,binding,run/"fault.log",R/"lost_worker.py",checkpoint_mode="fixed",checkpoint_interval=8):
        act(VS,"checkout",name="main");before=catalog(VS)
        (run/"lose-next-receipt").write_text("synthetic\n")
        start=rpc(VS,{"op":"ticket"})["ticket"];visible=[]
        v=rpc(VS,{"op":"turn","ticket":start,"text":"この枝には検証番号731を登録します。今は OK とだけ答えてください。","max_tokens":8},visible.append)
        state=rpc(VS,{"op":"status"});b=catalog(VS)
        assert v["turn_committed"] and v["checkpoint"]["status"]=="unknown" and v["ticket"]==state["ticket"] and state["mode"]=="uncertain"
        assert v["ticket"]["revision"]==start["revision"]+1 and len(b["messages"])==len(before["messages"])+2 and b["head"] is None and b["base"]==before["base"]
        assert b["pending"]==v["pending_sampled_ids"] and "".join(visible)==v["content"]
        try:act(VS,"recover")
        except Error as err:assert err.code=="native_outcome_unknown"
        else:raise AssertionError("ambiguous backend accepted")
        orphan={p.name:p.stat().st_size for p in n.images.glob(".writing-*.slot")}
        assert orphan
        write(R/"evidence/uncertain.json",{"before":before,"after":b,"start":start,"response":v,"mode":state["mode"],
              "unconfirmed_files_seen_by_test_only":orphan,"recover":"native_outcome_unknown"})


def stock(run,binding):
    n=Native(binding);spec=read(R/"evidence/numeric-fixtures.json");results=[]
    for case in spec["cases"]:
        name=case["anchor"];z=spec["points"][name];install(n,z,R/"reference"/(name+".slot"))
        n.restore(z["sha"],pack(z["tokens"]),z["bytes"])
        assert n.render(case["messages"])==case["ids"]
        phase(run,case["name"])
        try:
            v=n.call("POST","/completion",{"id_slot":0,"prompt":case["ids"],"n_predict":case["limit"],"temperature":0,"seed":19,
               "cache_prompt":True,"return_tokens":True,"response_fields":["content","tokens","tokens_cached","timings"]})
        finally:phase(run,"idle")
        results.append({"name":case["name"],"response":v})
    write(R/"evidence/stock-numeric.json",results)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stages",nargs="+",choices=["candidate","stock"]);p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    with window(R,"ADAPTIVE-NUMERIC"):
        for stage in a.stages:
            run=R/"runs"/(stage+"-"+str(a.attempt));assert not run.exists()
            with engine(R,VS if stage=="candidate" else R/"stock-store",run,observed=True,stock=stage=="stock") as binding:
                (candidate if stage=="candidate" else stock)(run,binding)
            write(run/"DONE.json",{"complete":True})
            p=R/"VALIDATION_RUNS.json";index=read(p) if p.exists() else {};index[stage]=run.name;write(p,index)
