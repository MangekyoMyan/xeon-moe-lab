"""Fresh-process return after wrapping the real 1280-cell SWA allocation, and an interrupted turn."""
from pathlib import Path
import json
import os
import shutil
import struct
import sys
import time
import native_trial as t
from native import Native, Uncertain, pack, unpack
from manager import Manager
from storage import Store
from x9ram_pilot import engine, window, private
from workflow import act, fixture, request, message

R=t.R
E=R/"ring-evidence"
POINTS=[1279,1280,1281,1535,1536,1537]
STOPS=POINTS+[1538,1539]
def source(binding,run,store):
    n=Native(binding)
    jobs=fixture(n);raw=[]
    for rep in range(3):
        dbroot=E/("off-catalog-"+str(rep));dbroot.mkdir()
        catalog=Store(dbroot,n,48*1024**2)
        ordinary=Manager(catalog);act(ordinary,"recover")
        for name in ("A","edit127","A","edit256"):
            raw.append(dict(rep=rep,op=request(ordinary,jobs[name],name,off_probe=False)))
        catalog.db.close()
    t.record(E/"practical-off.json",raw)
    # The 18-MiB trial could not hold the last image at all. At 56 MiB two
    # late images can coexist and the two coverage objectives can actually compete.
    pressure=[]
    for policy in ("all_history","recent"):
        root=E/("pressure-"+policy);root.mkdir()
        catalog=Store(root,n,56*1024**2);catalog.clean_files()
        m=Manager(catalog,"fixed",128,coverage_policy=policy,recent_window_tokens=130)
        act(m,"recover")
        act(m,"turn",messages=message(" f",65),max_tokens=1)
        named=act(m,"mark",name="pin",pin=True)
        msg=message(k=630)
        msg=message(k=630+645-len(n.render(msg)))
        assert len(n.render(msg))==645
        v=request(m,msg,"capacity56")
        status=m.handle({"op":"checkpoint"})
        actual=sum(f.stat().st_size for f in n.images.glob("*.slot"))
        assert actual==catalog.listing()["image_bytes"]<=56*1024**2
        pressure.append(dict(policy=policy,pin=named,turn=v,preview=status,bytes=actual))
        catalog.db.close()
    t.record(E/"pressure56.json",pressure)
    root=E/"clean-catalog";root.mkdir()
    catalog=Store(root,n,56*1024**2);catalog.clean_files();catalog.db.close()
    ids=n.render([dict(role="user",content=" a"*1600+"\nState a number.")])[:1540]
    assert len(ids)==1540
    t.E=E
    ordinary=ids[:418]
    for tag,cap in (("ordinary-create",[385]),("ordinary-control",[])):
        n.erase();t.traced(n,run,tag,ordinary,3,[385],cap,diagnostic_logits=False)
    for name,caps in (("ring-create",POINTS),("ring-control",[])):
        n.erase()
        t.traced(n,run,name,ids,2,STOPS,caps)
    item=next(e for e in json.loads((E/"ring-create.json").read_text())["events"] if e["position"]==1536)
    os.link(n.images/(item["sha"]+".slot"),E/"ring-1536.slot")
    return ids
def restore(binding,run,store):
    n=Native(binding);t.E=E
    normal=json.loads((E/"ordinary-create.json").read_text())
    cap=normal["events"][0]
    n.erase();n.restore(cap["sha"],pack(normal["prompt"][:385]),cap["bytes"])
    result=t.traced(n,run,"ordinary-return",normal["prompt"],3,[],[],diagnostic_logits=False)
    assert result["reply"]["timings"]["cache_n"]==385 and result["reply"]["timings"]["prompt_n"]==33
    job=json.loads((E/"ring-create.json").read_text())
    ids=job["prompt"];points={e["position"]:e for e in job["events"] if e["bytes"]}
    meta=[]
    for c in POINTS:
        e=points[c]
        path=n.images/(e["sha"]+".slot")
        # Read the two explicit one-stream XKV1 geometries. Confirm a real ring wrap.
        raw=path.read_bytes()
        pattern=b"XKV1\x01\0\0\0";i=0;geometry=[]
        while (offset:=raw.find(pattern,i))!=-1:
            _,_,cells,capacity,head,ranges=struct.unpack_from("<6I",raw,offset)
            geometry.append({"occupied":cells,"capacity":capacity,"head":head,"ranges":ranges})
            i=offset+8
        n.erase();n.restore(e["sha"],pack(ids[:c]),e["bytes"])
        end=min(p for p in STOPS if p>c)
        z=t.traced(n,run,"ring-return-"+str(c),ids[:end],1,[end],[])
        assert z["reply"]["timings"]["cache_n"]==c and z["reply"]["timings"]["prompt_n"]==end-c
        meta.append(dict(c=c,end=end,sha=e["sha"],geometry=geometry,epoch=n.epoch))
    assert any(any(g["capacity"]==1280 and g["head"]<r["c"] for g in r["geometry"]) for r in meta if r["c"]>1280)
    t.record(E/"geometry.json",meta)
    # Model+tokenizer are genuine; the exact-ID editing harness fixes n=p so a chat
    # template cannot add an accidental suffix. Serve's real render is used in CLI smoke.
    s=Store(store,n,512*1024**2)
    m=Manager(s,"fixed",256)
    with s.tx():
        for c,e in points.items():
            sid="a"+format(c,"031x")
            s.db.execute("INSERT INTO anchors VALUES(?,?,?,?,?,?,?)",(sid,n.compat,pack(ids[:c]),e["sha"],e["bytes"],time.time(),
                    __import__("hashlib").sha256(pack(ids[:c])).hexdigest()))
            s.db.execute("INSERT INTO anchor_refs VALUES(?,?)",(s.branch()["id"],sid))
            s.db.execute("INSERT INTO blobs VALUES(?,?,1,?)",(e["sha"],e["bytes"],time.time()))
        s.db.execute("UPDATE branches SET tokens=?,pending=?,messages='[]',gen=1",(pack(ids[:1537]),pack([99])))
    act(m,"recover")
    n.render=lambda _:ids[:1536]
    t.phase(run,"empty-suffix")
    try:v=act(m,"turn",text="exact token-ID fixture",max_tokens=1)
    finally:t.phase(run,"idle")
    rr=v["resume"]
    assert (rr["common_prefix"],rr["from_position"],rr["old_replayed"],rr["new_input"])==(1536,1535,1,0)
    t.record(E/"empty-suffix.json",v)
    old=s.branch()
    n.render=lambda _:ids[:1540]+[1527]*268
    def cancel(_):
        raise Uncertain("cancelled","explicit diagnostic cancellation after first streamed text")
    try:m.handle(dict(op="turn",text="cancel fixture",max_tokens=8,ticket=s.stamp()),cancel)
    except Uncertain:pass
    else:raise AssertionError("cancel committed")
    assert s.branch()["tokens"]==old["tokens"] and s.branch()["pending"]==old["pending"] and s.branch()["messages"]==old["messages"] and s.branch()["gen"]==old["gen"]
    assert s.control()["mode"]=="uncertain"
    t.record(E/"cancel.json",dict(committed_ids=unpack(old["tokens"]),pending_ids=unpack(old["pending"]),generation=old["gen"],
               fence=s.control()["mode"],anchor_count=s.db.execute("SELECT count(*) FROM anchors").fetchone()[0]))
    s.db.close()
def main():
    import subprocess
    store=R/"ring-store";owner=R.stat()
    with window(R,"PREFIX-RING"):
        for part in ("source","restore"):
            run=R/"runs"/("ring-"+part+"-"+str(time.time_ns()))
            # source+destination observed, third only confirms crash recovery.
            with engine(R,store,run,observed=part!="recovered") as b:
                if part!="recovered":private(run/"trace/phase",False)
                subprocess.run([sys.executable,"-B",__file__,part,str(b),str(run),str(store)],
                               check=True,user=owner.st_uid,group=owner.st_gid,extra_groups=[])
    from cli_recovery import run as entry
    entry()
if __name__=="__main__":
    if len(sys.argv)==5:
        {"source":source,"restore":restore}[sys.argv[1]](*map(Path,sys.argv[2:]))
    else:main()

