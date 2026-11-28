"""Assemble only synthetic fixtures, measured rows, model state images, source and receipts.
No model weights, API keys, live service configurations or user conversations are read.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

R=Path(__file__).resolve().parent
D=R/"sharing"
F=R/"checkout/current/experiments/state-branches"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
def copy(p,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dst)
    if p.stat().st_mode & 0o111:dst.chmod(0o755)
def pack(p,out,level=3):
    out.parent.mkdir(parents=True,exist_ok=True)
    with p.open("rb") as inp,out.open("wb") as sink,gzip.GzipFile(filename="",fileobj=sink,mode="wb",compresslevel=level,mtime=0) as stream:
        shutil.copyfileobj(inp,stream,1024*1024)
def tracing():
    idx=[]
    jobs=[
        ("compact",R,[s for s in (R/"runs").glob("source-*")]+list((R/"runs").glob("restore-*"))),
        ("physical",R/"layout",list((R/"layout/runs").glob("*"))),
        ("ring",R,[s for s in (R/"runs").glob("ring-*")])]
    for variant,base,runs in jobs:
        for run in sorted(runs):
            calls=run/"trace/calls.jsonl"
            if not calls.is_file():continue
            for line in calls.read_text().splitlines():
                v=json.loads(line);f=v.pop("logits_file",None)
                if f:
                    p=calls.parent/f;h=sha(p)
                    dest=D/"rows"/(h+".f32.gz")
                    if not dest.exists():pack(p,dest)
                    v["row_sha256"]=h
                idx.append(dict(variant=variant,run=run.name,**v))
    path=D/"native-calls.jsonl"
    with path.open("w") as out:
        for row in idx:out.write(json.dumps(row,separators=(",",":"))+"\n")
def images():
    out=[]
    for variant,root,job,store in [("compact",R/"evidence","create.json",R/"num-store/images"),
                                   ("physical",R/"layout/evidence","create.json",R/"layout/num-store/images"),
                                   ("ring",R/"ring-evidence","ring-create.json",R/"ring-store/images"),
                                   ("long",R/"long-evidence","anchor.json",R/"long-store/images")]:
        source=json.loads((root/job).read_bytes())
        for e in source["positions" if variant=="long" else "events"]:
            if not e["bytes"]:continue
            include=variant in ("compact","physical") or (variant=="ring" and e["position"]==1536) or (variant=="long" and e["position"]==3072)
            item=dict(variant=variant,position=e["position"],sha256=e["sha"],bytes=e["bytes"],parts=[])
            if include:
                orig=(R/"ring-evidence/ring-1536.slot") if variant=="ring" else store/(e["sha"]+".slot")
                assert sha(orig)==e["sha"]
                # Individual Git objects remain below 24 MiB. Split raw bytes before compression.
                with orig.open("rb") as inp:
                    for i in range((e["bytes"]+16*1024**2-1)//(16*1024**2)):
                        data=inp.read(16*1024**2)
                        name=f"images/{e['sha']}.{i:02}.gz";destination=D/name
                        destination.parent.mkdir(exist_ok=True)
                        with destination.open("wb") as sink,gzip.GzipFile(filename="",mode="wb",fileobj=sink,mtime=0,compresslevel=3) as z:z.write(data)
                        item["parts"].append(name)
            out.append(item)
    write(D/"image-index.json",out)
def main():
    D.mkdir(exist_ok=False)
    for p in F.iterdir():
        if p.is_file():copy(p,D/"source"/p.name)
    for p in ("LICENSE","THIRD_PARTY_NOTICES.md","ACKNOWLEDGMENTS.md"):
        copy(R/"checkout"/p,D/p)
    shutil.copytree(R/"workflow-evidence",D/"workflow-evidence")
    for p in (R/"long-evidence").glob("*.json"):copy(p,D/"long-evidence"/p.name)
    for variant,root in [("compact",R/"evidence"),("physical",R/"layout/evidence"),("ring",R/"ring-evidence")]:
        for p in sorted(root.glob("*.json")):
            if variant=="compact" and p.name.startswith("workflow-"):continue
            copy(p,D/"fixtures"/variant/p.name)
    for p in ("native_trial.py","physical_trial.py","ring_trial.py","cli_recovery.py","long_edit.py","workflow.py","compare_rows.py","summarize.py"):
        copy(R/p,D/"hardware"/p)
    for p in ("SUMMARY.json","RING.json","SERVICE.json","DESIGN.json","POLICY_REPLAY.json","RESULTS_JA.md","CODE.json","offline_replay.py"):
        copy(R/p,D/p)
    copy(R/"summarize.py",D/"summarize.py")
    for label,base in (("compact",R/"build-compact"),("server",R/"build-release"),("model",R/"build-release/physical")):
        copy(base/"BUILD.json",D/"build"/(label+".json"))
    import difflib
    text=""
    for name in ("server-context.cpp","server-task.cpp"):
        a=(R.parent/"deepseek41-bringup-20261101/source/tools/server"/name).read_text().splitlines(True)
        b=(R/"build-compact/src/tools/server"/name).read_text().splitlines(True)
        text+="".join(difflib.unified_diff(a,b,"a/tools/server/"+name,"b/tools/server/"+name))
    compact=D/"build/compact-native.patch";compact.write_text(text)
    assert sha(compact)==json.loads((D/"build/compact.json").read_text())["patch_sha256"]
    tracing();images()
    copy(R/"BUNDLE_JA.md",D/"README.md")
    records={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob("*")) if p.is_file()}
    write(D/"MANIFEST.json",records)
    print("files",len(records),"bytes",sum(v["bytes"] for v in records.values()))
if __name__=="__main__":main()

