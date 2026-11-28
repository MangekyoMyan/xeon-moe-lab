"""Rotated, model-loaded edit workflows. Synthetic inputs only; restore the initial service on exit."""
import argparse
import json
from pathlib import Path
import sys
import time
R=Path(__file__).resolve().parent
E=R/"workflow-evidence"
F=R/"checkout/current/experiments/state-branches"
sys.path.insert(0,str(F))
from manager import Manager
from native import Native, Error, pack, unpack
from planner import Profile
from storage import Store
from prefix_io import boundaries
from x9ram_pilot import window, engine, manager, private
from branchctl import rpc

def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def append(p,v):
    with p.open("a") as out:out.write(json.dumps(v,ensure_ascii=False)+"\n")
def act(m,op,**kw):
    return m.handle(dict(op=op,ticket=m.store.stamp(),**kw))
def db_for(n,label,mode,**options):
    p=R/"cases-final"/label
    p.mkdir(parents=True)
    s=Store(p,n,48*1024**2)
    s.clean_files()
    m=Manager(s,mode,options.pop("interval",128),**options)
    act(m,"recover")
    return m
def message(words=" x",k=368,edit=None):
    parts=[words]*k
    if edit is not None:parts[edit]=" y"
    return [{"role":"user","content":"".join(parts)+"\nWrite integers from one to a hundred, in words."}]
def fixture(n):
    # Put the next absolute multiple in generation. Edits replace one raw token with one other.
    prompt=message(k=360);adjust=378-len(n.render(prompt))
    prompt=message(k=360+adjust);ids=n.render(prompt);assert len(ids)==378
    def edited(p):
        first=n.render(message(k=360+adjust,edit=0))
        prefix=next(i for i,(a,b) in enumerate(zip(ids,first)) if a!=b)
        msg=message(k=360+adjust,edit=p-prefix)
        v=n.render(msg)
        assert len(v)==len(ids) and [i for i,(a,b) in enumerate(zip(ids,v)) if a!=b]==[p]
        return msg
    return dict(A=prompt,**{"edit"+str(p):edited(p) for p in (127,128,129,255,256,257)})
def calibration(n,tag):
    m=db_for(n,tag,"fixed")
    for rep in range(2):
        n.erase()
        def receive(e,ch,stream):
            end=e["position"];now=e["arrived_ns"];ch.sendall(b"s")
            dt=(now-state[1])/1e6
            raw.append(dict(at=state[0],length=end-state[0],ms=dt))
            m.cp.model.add("prefill",state[0],end-state[0],dt)
            state[:]=[end,time.monotonic_ns()]
        raw=[];ids=n.render(message(" z",760))
        state=[0,time.monotonic_ns()]
        with boundaries(n,[2,8,24,56,120,248,504,len(ids)],[],receive):
            assert n.replay(pack(ids),False)==len(ids)
        append(E/"calibration.jsonl",dict(rep=rep,segments=raw))
        n.erase()
        # Normal durable generation and real route validation+restore.
        act(m,"turn",messages=message(" q",380),max_tokens=2)
        branch=m.store.branch()
        append(E/"calibration.jsonl",dict(rep=rep,matching=[(r["id"],len(r["tokens"])//4) for r in m.cp.anchors.matching(branch,branch["tokens"])]))
        for c in (0,128,256):
            _,route=m.cp.select(branch,branch["tokens"],c)
            append(E/"calibration.jsonl",dict(rep=rep,route=route))
        n.erase()
        # Place the durable branch before the next public operation.
        act(m,"recover")
    training=m.cp.model.rows[:]
    m.store.db.close()
    assert sum(r[0]=="restore" for r in training)>=2 and sum(r[0]=="save" for r in training)>=2
    dump(E/"training.json",training)
    return training

def request(m,msg,name,off_probe=False):
    n=m.native;ids=n.render(msg);old=unpack(m.store.branch()["tokens"])
    common=next((i for i,(a,b) in enumerate(zip(ids,old)) if a!=b),min(len(ids),len(old)))
    # Only the off instrumented edit series inserts a single clock boundary.
    # Uninstrumented off follows as a separate pair. No capture or file I/O is added to off.
    probe=[];route={};real_erase=n.erase
    def erase():
        t=time.monotonic_ns();real_erase()
        route.update(begin=t,end=time.monotonic_ns())
    def clock(e,ch,stream):
        ch.sendall(b"s");probe.append(time.monotonic_ns())
    n.erase=erase
    times=[]
    started=time.monotonic_ns()
    try:
        with boundaries(n,[common] if off_probe and common else [],[],clock):
            v=m.handle(dict(op="turn",ticket=m.store.stamp(),messages=msg,max_tokens=12),lambda t:times.append(time.monotonic_ns()))
    finally:n.erase=real_erase
    finished=time.monotonic_ns()
    resume=v.get("resume")
    if resume is None and probe and route:
        resume=dict(common_prefix=common,from_position=0,old_replayed=common,new_input=len(ids)-common,
                    total_ms=(probe[0]-route["begin"])/1e6,route_ms=(route["end"]-route["begin"])/1e6,
                    basis="off; clock starts at erase; small pre-erase ancestor selection excluded")
    return dict(name=name,common=common,ids=ids,first_ms=(times[0]-started)/1e6 if times else None,
                total_ms=(finished-started)/1e6,max_delta_gap_ms=max(((b-a)/1e6 for a,b in zip(times,times[1:])),default=None),
                resume=resume,receipt=v)
def run_session(n,label,mode,training,fixture,sequence,off_probe=True):
    m=db_for(n,label,mode,resume_target_ms=3500)
    m.cp.model=Profile(n.compat,n.identity["offload_thresholds"],training)
    result=[]
    try:
        for name in sequence:
            z=request(m,fixture[name],name,off_probe=mode=="off" and off_probe and name.startswith("edit"))
            result.append(z)
            append(E/"workflows.jsonl",dict(session=label,mode=mode,op=z))
        p=m.handle({"op":"checkpoint","view":"status"})
        append(E/"plans.jsonl",dict(session=label,plan=p,profile=m.cp.model.rows))
    finally:m.store.db.close()
    return result

def pinned(n,label,policy,training,fixture):
    m=db_for(n,label,"fixed",coverage_policy=policy,recent_window_tokens=130)
    m.cp.model=Profile(n.compat,n.identity["offload_thresholds"],training)
    r=act(m,"turn",messages=message(" f",65),max_tokens=1)
    mark=act(m,"mark",name="pinned",pin=True)
    act(m,"budget",bytes=18*1024**2)
    v=request(m,fixture["A"],"capacity")
    before=act(m,"status")
    try:act(m,"budget",bytes=1)
    except Error as err:failure=err.code
    else:raise AssertionError("one-byte replacement cannot contain the pin")
    after=act(m,"status")
    assert before["budget_bytes"]==after["budget_bytes"]==18*1024**2
    actual=sum(p.stat().st_size for p in n.images.glob("*.slot"))
    assert actual==after["image_bytes"]<=18*1024**2
    assert all(e.get("peak_bytes",0)<=18*1024**2 for e in v["receipt"]["checkpoint"]["events"])
    append(E/"pressure.jsonl",dict(policy=policy,mark=mark,receipt=v,preview=m.handle({"op":"checkpoint"}),
           disk_actual=actual,rejection=failure))
    m.store.db.close()

def worker(b,run):
    n=Native(b)
    import os,hashlib
    dump(E/"implementation.json",dict(uid=os.geteuid(),runtime=n.identity,
         hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in F.glob("*.py")}))
    jobs=fixture(n);dump(E/"workflow-fixtures.json",jobs)
    training=calibration(n,"train-"+run.name)
    # Restore the complete original stream before each edit (A repeats), identical input order in all arms.
    sequence=["A","edit127","A","edit128","A","edit129","A","edit255","A","edit256","A","edit257"]
    for rep in range(3):
        modes=("off","fixed","auto")
        for i in range(3):
            mode=modes[(rep+i)%3]
            label=f"{run.name}-{rep}-{mode}"
            run_session(n,label,mode,training,jobs,sequence)
    run_session(n,run.name+"-off-unprobed","off",training,jobs,["A","edit256"],off_probe=False)
    for policy in ("all_history","recent"):pinned(n,run.name+"-"+policy,policy,training,jobs)

def measure():
    run=R/"runs"/("workflow-"+str(time.time_ns()))
    with window(R,"PREFIX-WORKFLOW"):
        with engine(R,R/"workflow-store",run) as b:
            import subprocess
            owner=R.stat()
            subprocess.run([sys.executable,"-B",__file__,"--worker",str(b),str(run)],check=True,
                           user=owner.st_uid,group=owner.st_gid,extra_groups=[])
        # A real Unix manager/CLI goes through its usual permission, rpc and startup checks.
        store=R/"smoke-store"
        run=R/"runs"/("cli-"+str(time.time_ns()))
        with engine(R,store,run) as binding, manager(store,binding,run/"manager.log",checkpoint_mode="auto",
                 checkpoint_interval=32,coverage_policy="recent",recent_window_tokens=64,checkpoint_budget_mib=32):
            import subprocess
            def cli(*args):
                v=subprocess.check_output([sys.executable,"-B",str(F/"branchctl.py"),"--store",str(store),*args],text=True)
                return v
            before=cli("checkpoint","preview","--history-tokens","128")
            x=json.loads(cli("--json","turn","--text"," x"*75,"--max-tokens","3"))
            after=json.loads(cli("--json","checkpoint","status"))
            assert x["turn_committed"] and after["currently_available"]
            dump(E/"cli.json",dict(before=before,response=x,after=after))
if __name__=="__main__":
    if len(sys.argv)==4 and sys.argv[1]=="--worker":worker(Path(sys.argv[2]),Path(sys.argv[3]))
    else:measure()

