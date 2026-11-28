"""Shared acquisition on x9ram; all inputs and catalogs are synthetic."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

R = Path(__file__).resolve().parent
LAB = R.parent
OLD = LAB / "engine-state-phase1-20261128"
FEATURE = R / "checkout/current/experiments/state-branches"
sys.path.insert(0, str(FEATURE))
from native import Native, Error, digest, encode, image_tokens, pack, unpack, sync_dir
from storage import Store, uid
from branchctl import rpc
from x9ram_pilot import engine, window, private
# The historical x9ram launcher prepends its acquisition directories; import this
# experiment's sibling modules from this directory after the launcher has loaded.
sys.path.insert(0, str(R))

STORE = R / "trial-store"
BUDGET = 1024**3
DIRTY = "この枝で扱う仮の控え番号は 842 です。OK とだけ答えてください。"
REGISTER = "この枝の保管箱番号は 731 です。後で参照するため覚えてください。今は半角の OK だけ返してください。"
QUERY = "この枝で登録した保管箱番号を答えてください。半角の数字だけを返してください。"


def read(p):
    return json.loads(Path(p).read_bytes())


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(p, value):
    with Path(p).open("a") as f:
        f.write(encode(value) + "\n")


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def blob(point):
    return point["sha"] + ".slot"


def asset(point):
    return R / "assets" / blob(point)


def add_asset(path, point):
    if path.stat().st_size != point["bytes"] or digest(path) != point["sha"] or image_tokens(path, 131072) != pack(point["tokens"]):
        raise ValueError("source image differs from its frozen native IDs or checksum")
    dst = asset(point)
    if not dst.exists():
        shutil.copyfile(path, dst)
        dst.chmod(0o444)
    assert digest(dst) == point["sha"]


def install_blob(native, point):
    path = native.images / blob(point)
    if not path.exists():
        os.link(asset(point), path)
    assert digest(path) == point["sha"]


def follow(messages, text):
    return [*messages, {"role": "user", "content": text}]


def prepare_case(native, size):
    if size == "short":
        prior = OLD / "perf-store"
        db = sqlite3.connect(f"file:{prior/'branches.sqlite3'}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        points = {}
        for key, name in (("P","P"),("T","T0")):
            v = db.execute("SELECT s.* FROM snapshots s JOIN marks m ON m.sid=s.id WHERE m.name=?", (name,)).fetchone()
            assert v["compat"] == native.compat
            point={"id":v["id"],"parent":v["parent"],"tokens":unpack(v["tokens"]),"pending":unpack(v["pending"]),
                   "messages":json.loads(v["messages"]),"sha":v["sha"],"bytes":v["bytes"]}
            add_asset(prior/"images"/blob(point),point)
            points[key]=point
        db.close()
    else:
        earlier=LAB/"warm-suffix-20261127"
        previous=read(earlier/"FIXTURES.json")["cases"]["warehouse"]
        img=read(earlier/"shared/IMAGES.json")["warehouse"]
        parent={"id":uid(),"parent":None,"tokens":previous["history_ids"],"pending":[],
                "messages":previous["messages"][:2],"sha":img["sha256"],"bytes":img["bytes"]}
        add_asset(earlier/"shared/images"/img["file"],parent)
        points={"P":parent}
    p=points["P"]
    install_blob(native,p)
    for key, question in (("T",REGISTER),("D",DIRTY)):
        if key in points:
            continue
        native.restore(p["sha"], pack(p["tokens"]), p["bytes"])
        msgs=follow(p["messages"],question);ids=native.render(msgs)
        assert ids[:len(p["tokens"])] == p["tokens"]
        out=native.generate(ids,1,lambda _:None)
        assert out["timings"]["cache_n"]==len(p["tokens"]) and out["sampled_count"]==1
        path,sha,size_bytes=native.save(pack(ids))
        point={"id":uid(),"parent":p["id"],"tokens":ids,"pending":out["sampled"],
               "messages":msgs+[{"role":"assistant","content":out["content"]}],"sha":sha,"bytes":size_bytes}
        add_asset(path,point);path.unlink()
        points[key]=point
    t=points["T"];install_blob(native,t);native.restore(t["sha"],pack(t["tokens"]),t["bytes"])
    next_ids=native.render(follow(t["messages"],QUERY))
    assert next_ids[:len(t["tokens"])]==t["tokens"]
    sample=native.generate(next_ids,8,lambda _:None)
    assert sample["timings"]["cache_n"]==len(t["tokens"])
    assert sample["content"].strip()=="731",sample["content"]
    if size=="short":assert len(points["D"]["tokens"])==285 and points["D"]["bytes"]==6193740
    else:assert 16300<=len(points["D"]["tokens"])<=16700
    assert len({z["sha"] for z in points.values()})==3
    out={"size":size,"points":points,"next_ids":next_ids,"answer":sample["content"],"answer_ids":sample["sampled"],
         "compat":native.compat,"next_request":QUERY,"dirty_request":DIRTY}
    write(R/"cases"/(size+".json"),out)
    return out


def make_template(native, case, duplicate):
    # Install a frozen template under a disposable catalog. All benchmark mutations
    # subsequently go through the real manager API. A known sibling owns D in duplicate trials.
    root=R/"cases"/(case["size"]+("-resident" if duplicate else "-fresh"))
    root.mkdir()
    st=Store(root,native,BUDGET)
    p,t,d=(case["points"][k] for k in ("P","T","D"))
    with st.tx():
        for z in [p,t]+([d] if duplicate else []):
            st.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)",
                (z["id"],z["parent"],native.compat,pack(z["tokens"]),pack(z["pending"]),encode(z["messages"]),z["sha"],z["bytes"],time.time()))
            st.db.execute("INSERT INTO blobs VALUES(?,?,1,?)",(z["sha"],z["bytes"],time.time()))
        st.db.execute("UPDATE branches SET name='depart',origin=?,base=?,head=?,tokens=?,pending=?,messages=? WHERE name='main'",
                      (p["id"],p["id"],p["id"],pack(p["tokens"]),pack(p["pending"]),encode(p["messages"])))
        st.db.execute("INSERT INTO branches VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                 (uid(),"target",p["id"],t["id"],t["id"],pack(t["tokens"]),pack(t["pending"]),encode(t["messages"]),0,0,time.time(),0))
        st.db.execute("INSERT INTO marks VALUES(?,?,1)",("origin",p["id"]))
        st.db.execute("INSERT INTO marks VALUES(?,?,1)",("target",t["id"]))
        if duplicate:
            st.db.execute("INSERT INTO marks VALUES(?,?,0)",("already-saved",d["id"]))
        st.state("ready")
    assert st.db.execute("PRAGMA synchronous").fetchone()[0]==2
    assert st.db.execute("PRAGMA journal_mode").fetchone()[0]=="wal"
    st.db.close()


def reset_trial(native, case, duplicate):
    # Only this experiment's private fixture catalog is replaced, with its writer stopped.
    if STORE!=R/"trial-store" or (STORE/"manager.sock").exists():
        raise RuntimeError("scratch catalog still has a manager")
    wanted=[case["points"][k] for k in ("P","T")]+([case["points"]["D"]] if duplicate else [])
    names={blob(z) for z in wanted}
    for path in native.images.iterdir():
        if path.name not in names:
            if path.suffix!=".slot":raise RuntimeError("unexpected scratch image filename")
            path.unlink()
    for z in wanted:
        install_blob(native,z)
    assert {p.name for p in native.images.iterdir()}==names
    sync_dir(native.images)
    root=R/"cases"/(case["size"]+("-resident" if duplicate else "-fresh"))
    db=STORE/"branches.sqlite3"
    if any((STORE/("branches.sqlite3"+ext)).exists() for ext in ("-wal","-shm")):
        raise RuntimeError("scratch SQLite has an open journal")
    temporary=STORE/".reset-db"
    shutil.copyfile(root/"branches.sqlite3",temporary)
    private(temporary,False)
    with temporary.open("rb") as f:os.fsync(f.fileno())
    temporary.replace(db);sync_dir(STORE)
