"""Check logical B at every optional-save failure boundary on disposable synthetic slots."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import sys

R=Path(__file__).resolve().parent

def run(code):
    sys.path.insert(0,str(code))
    from test_checkpoints import TextSlot
    from manager import Manager
    from storage import Store
    from native import Error,Uncertain,unpack
    results={}
    for kind in ("budget","write_error","precommit","cleanup","unknown","boundary"):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);n=TextSlot(root);s=Store(root,n,10_000);m=Manager(s)
            def call(op,**kw):return m.handle({"op":op,"ticket":s.stamp(),**kw})
            def state():
                b=s.branch();return {"mode":s.control()["mode"],"gen":b["gen"],"rev":s.stamp()["revision"],"head":b["head"],"base":b["base"],
                    "tokens":unpack(b["tokens"]),"pending":unpack(b["pending"]),"messages":json.loads(b["messages"]),
                    "snapshots":s.db.execute("SELECT count(*) FROM snapshots").fetchone()[0]}
            call("recover");call("turn",text="P");parent=call("mark",name="parent",pin=True)["snapshot"]
            m=Manager(s,"fixed",1)
            if kind=="budget":call("budget",bytes=parent["image_bytes"]+100)
            if kind=="precommit":
                s.db.execute("CREATE TEMP TRIGGER refuse BEFORE INSERT ON snapshots BEGIN SELECT RAISE(ABORT,'injected test'); END")
            import contextlib
            with contextlib.ExitStack() as stack:
                if kind=="budget":
                    original=n.save
                    def large(ids):
                        from native import digest
                        p,_,_=original(ids)
                        with p.open("ab") as f:f.write(bytes(256))
                        return p,digest(p),p.stat().st_size
                    stack.enter_context(patch.object(n,"save",side_effect=large))
                if kind=="write_error":stack.enter_context(patch.object(n,"save",side_effect=OSError("synthetic confirmed write failure")))
                if kind=="cleanup":stack.enter_context(patch.object(s,"clean_files",side_effect=OSError("synthetic postcommit cleanup")))
                if kind=="unknown":stack.enter_context(patch.object(n,"save",side_effect=Uncertain("native_outcome_unknown","synthetic missing receipt")))
                if kind=="boundary":stack.enter_context(patch.object(n,"save",side_effect=Error("boundary_mismatch","synthetic invalid boundary")))
                before=state();q=call("turn",text="B");after=state()
            cp=q["checkpoint"];published=kind=="cleanup"
            expected="uncertain" if kind=="unknown" else "invalid" if kind=="boundary" else "ready"
            assert after["mode"]==expected and q["turn_committed"] and after["gen"]==before["gen"]+1
            assert q["ticket"]==s.stamp() and after["rev"]==before["rev"]+1+published
            assert len(after["messages"])==len(before["messages"])+2 and after["pending"]==[1] and after["tokens"]==unpack(n.live)
            if published:
                assert after["head"]==after["base"]==cp["snapshot_id"] and after["snapshots"]==before["snapshots"]+1 and cp["status"]=="saved_cleanup_warning"
            else:
                assert after["head"] is None and after["base"]==before["base"] and after["snapshots"]==before["snapshots"]
                assert cp["status"]==("unknown" if kind=="unknown" else "failed")
            assert m.policy.observed["optional_save"]==0
            results[kind]={"before":before,"after":after,"receipt":q}
            s.db.close()
    return results

if __name__=="__main__":
    code=R/"checkout/current/experiments/state-branches" if (R/"checkout").exists() else R/"cli"
    data=run(code)
    if "--check" not in sys.argv:(R/"evidence/faults.json").write_text(json.dumps(data,indent=2)+"\n")
    print(len(data),"committed-turn fault cases verified")
