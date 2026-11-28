"""Audit logical state around a repeated mark on the synthetic slot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def probe(feature):
    sys.path.insert(0, str(feature))
    from manager import Manager
    from native import Error
    from storage import Store
    from test_branches import SmallSlot
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); store=Store(root,SmallSlot(root),4096); manager=Manager(store)
        def call(op,**fields):
            return manager.handle({"op":op,"ticket":store.stamp(),**fields})
        def state():
            b=dict(store.branch())
            for k in ("tokens","pending"):
                b[k]=b[k].hex()
            return {"active":b,"ticket":store.stamp(),"mark":store.resolve("P")["id"],
                    "snapshots":store.db.execute("SELECT count(*) FROM snapshots").fetchone()[0],
                    "images":sorted(p.name for p in store.native.images.glob("*.slot"))}
        call("recover");call("turn",text="P");call("mark",name="P");call("budget",bytes=1);call("budget",bytes=4096)
        before=state()
        try:
            call("mark",name="P")
            error=None
        except Error as exc:
            error=exc.code
        after=state()
        result={"error":error,"before":before,"after":after,"logical_state_unchanged":before==after}
        result["old_ticket_accepted"]=manager.handle({"op":"fork","ticket":before["ticket"],"name":"child","from":"P"})["branch"]["name"]=="child"
        store.db.close()
        return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("feature",type=Path);parser.add_argument("output",type=Path)
    a=parser.parse_args();v=probe(a.feature.resolve());a.output.write_text(json.dumps(v,indent=2)+"\n")
    print(v["error"],"snapshots",v["before"]["snapshots"],v["after"]["snapshots"],"logical state equal",v["logical_state_unchanged"])
