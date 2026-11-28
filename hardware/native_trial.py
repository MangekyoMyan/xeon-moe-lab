"""Numerical controls for intra-request full DSV4 images. Uses only a synthetic private store."""
import json
import os
from pathlib import Path
import sys
import time
R = Path(__file__).resolve().parent
F = R/"checkout/current/experiments/state-branches"
sys.path.insert(0,str(F))
from native import Native, digest, image_tokens, unpack, pack
from prefix_io import boundaries, read_image
from x9ram_pilot import private, window, engine
E = R/"evidence"
POINTS=[255,256,257,511,512,513,768]
STOPS=sorted(set(POINTS+[769,770,771]))
def record(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
def phase(run,s):
    (run/"trace/phase").write_text(s+"\n")
def traced(n,run,name,ids,count,stops,points,diagnostic_logits=True):
    events=[]
    def receive(e,conn,stream):
        if e["bytes"]:
            filename=n.images/("prefix-%s.slot"%e["position"])
            conn.sendall(b"w")
            read_image(stream,filename,e["bytes"])
            raw=image_tokens(filename,n.limit)
            assert len(raw)==4*e["position"] and raw[:4*len(ids)]==pack(ids[:e["position"]])
            e.update(sha=digest(filename),ids=unpack(raw))
            os.replace(filename,n.images/(e["sha"]+".slot"))
            conn.sendall(b"k")
        else:
            conn.sendall(b"s")
        events.append(e)
    phase(run,name)
    try:
        with boundaries(n,stops,points,receive,diagnostic_logits=diagnostic_logits):
            v=n.call("POST","/completion",dict(prompt=ids,id_slot=0,n_predict=count,return_tokens=True,
                temperature=0,seed=19,ignore_eos=True,cache_prompt=True,
                response_fields=["tokens","timings","tokens_cached"]))
    finally:
        phase(run,"idle")
    job=dict(name=name,prompt=ids,n_predict=count,events=events,reply=v)
    record(E/(name+".json"),job)
    return job
def acquire():
    store=R/"num-store"
    with window(R,"PREFIX-NUMERIC"):
        run=R/"runs"/("source-"+str(time.time_ns()))
        with engine(R,store,run,observed=True) as b:
            n=Native(b)
            def ids(k):
                return n.render([{"role":"user","content":" x"*k+"\nCount upward from one to one hundred, in words."}])
            prompt=ids(700)
            prompt=ids(700+750-len(prompt))
            assert 740<=len(prompt)<=755
            for tag,stops,points in [("create",STOPS,POINTS),("split1",STOPS,[]),("split2",STOPS,[]),("unsplit",[],[])]:
                n.erase()
                job=traced(n,run,tag,prompt,29,stops,points)
                if tag=="create":
                    ev=job["events"];assert [e["position"] for e in ev if e["bytes"]]==POINTS
                    assert next(e for e in ev if e["position"]==768)["phase"]=="decode"
                    assert all(e["ids"]==(prompt+job["reply"]["tokens"][:-1])[:e["position"]] for e in ev if e["bytes"])
            record(E/"original.json",{"run":run.name,"epoch":n.epoch,"identity":n.identity})
        restore_run(store)
def restore_run(store):
    base=json.loads((E/"create.json").read_text())
    ids=base["prompt"]+base["reply"]["tokens"][:-1]
    stops=sorted(set(STOPS+[len(base["prompt"])]))
    points={e["position"]:e for e in base["events"] if e["bytes"]}
    run=R/"runs"/("restore-"+str(time.time_ns()))
    with engine(R,store,run,observed=True) as b:
        n=Native(b);log=[]
        for p,e in points.items():
            n.erase()
            n.restore(e["sha"],pack(ids[:p]),e["bytes"])
            end=next(v for v in stops if v>p)
            name="resume-"+str(p)
            v=traced(n,run,name,ids[:end],1,[end],[])
            assert v["reply"]["timings"]["cache_n"]==p and v["reply"]["timings"]["prompt_n"]==end-p
            log.append(dict(position=p,sha=e["sha"],bytes=e["bytes"],end=end,new=end-p,epoch=n.epoch))
            if p==768:
                for end in (770,771):
                    v=traced(n,run,name+"-"+str(end),ids[:end],1,[end],[])
                    assert v["reply"]["timings"]["prompt_n"]==1
        record(E/"restorations.json",log)
if __name__=="__main__":
    acquire()

