"""Practical held-out long edit, no timing-only split in off, three paired restores of the same seed."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
R=Path(__file__).resolve().parent
E=R/"long-evidence"
sys.path.insert(0,str(R/"checkout/current/experiments/state-branches"))
from native import Native, pack
from manager import Manager
from storage import Store
from x9ram_pilot import engine,window
from workflow import act
def put(name,obj):
    (E/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def fixtures(n):
    def msg(count,pad,edit=-1,code="284"):
        records=[f"品目{i:03d}　倉庫{'東西南北'[i%4]}　空箱コード{code if i==edit else '731'}" for i in range(count)]
        return [dict(role="user",content="検品用の架空台帳です。\n"+"\n".join(records)+"\n余白"+" x"*pad+"\n確認したら OK とだけ答えてください。")]
    lo,hi=1,500
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(n.render(msg(mid,0)))<=3500:lo=mid
        else:hi=mid-1
    count=lo
    # Start at least one row below the limit so the right-hand BPE at the
    # padding/newline is stable. Check the actual templated IDs each time.
    count=max(count-1,1)
    pad=3500-len(n.render(msg(count,0)))
    for _ in range(5):
        source=msg(count,pad);ids=n.render(source)
        if len(ids)==3500:break
        pad+=3500-len(ids)
    assert len(ids)==3500
    def pair(idx,code):
        m=msg(count,pad,idx,code);other=n.render(m)
        p=next(j for j,(a,b) in enumerate(zip(ids,other)) if a!=b)
        return m,other,p
    for code in ("284","732","730","123"):
        lo,hi=0,count-1
        while lo<hi:
            mid=(lo+hi)//2
            if pair(mid,code)[2]<3328:lo=mid+1
            else:hi=mid
        edit,other,p=pair(lo,code)
        if 3328<=p<3360 and len(other)==3500:
            return dict(source=source,edit=edit,ids=ids,edited_ids=other,p=p,records=count,
                        changed_record=lo,replacement=code)
    raise AssertionError("no equal-length single-record edit in the predeclared range")
def perform(m,msg):
    started=time.monotonic_ns();deltas=[]
    v=m.handle(dict(op="turn",ticket=m.store.stamp(),messages=msg,max_tokens=4),lambda _:deltas.append(time.monotonic_ns()))
    end=time.monotonic_ns()
    assert deltas and v["turn_committed"]
    return dict(first_ms=(deltas[0]-started)/1e6,total_ms=(end-started)/1e6,receipt=v)
def worker(binding):
    n=Native(binding);fixture=fixtures(n);put("fixture.json",fixture)
    for tag in ("off-seed","fixed-seed"):
        p=E/tag;p.mkdir()
        s=Store(p,n,512*1024**2);s.clean_files()
        m=Manager(s,"off" if tag=="off-seed" else "fixed",1024)
        act(m,"recover")
        run=perform(m,fixture["source"])
        if tag=="fixed-seed":
            events=[e for e in run["receipt"]["checkpoint"]["events"] if e["status"]=="saved"]
            assert [e["position"] for e in events]==[1024,2048,3072]
            mark=act(m,"mark",name="seed",pin=True)
            put("anchor.json",dict(mark=mark,positions=events,budget=512*1024**2))
        put(tag+".json",run)
        s.db.close()
    base=sqlite3.connect(E/"fixed-seed/branches.sqlite3")
    results=[]
    for rep,order in enumerate((("off","fixed"),("fixed","off"),("off","fixed"))):
        for mode in order:
            dbroot=E/f"{rep}-{mode}";dbroot.mkdir()
            c=sqlite3.connect(dbroot/"branches.sqlite3");base.backup(c);c.close()
            n.erase()
            s=Store(dbroot,n,512*1024**2);s.clean_files()
            m=Manager(s,mode,1024)
            placement=act(m,"recover");assert placement["exact_head"] and placement["replayed"]==0
            run=perform(m,fixture["edit"])
            r=run["receipt"];assert r["common_prefix"]==fixture["p"]
            if mode=="off":assert r["timings"]["prompt_n"]==3500
            else:
                assert r["resume"]["from_position"]==3072 and r["timings"]["prompt_n"]==428
                assert r["resume"]["old_replayed"]==fixture["p"]-3072
            results.append(dict(rep=rep,mode=mode,seed_restore=placement,**run))
            put("paired.json",results)
            s.db.close()
    base.close()
def main():
    import os
    run=R/"runs"/("long-"+str(time.time_ns()))
    with window(R,"PREFIX-LONG"):
        with engine(R,R/"long-store",run) as b:
            owner=R.stat()
            subprocess.run([sys.executable,"-B",__file__,"worker",str(b)],user=owner.st_uid,group=owner.st_gid,extra_groups=[],check=True)
if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="worker":worker(Path(sys.argv[2]))
    else:main()

