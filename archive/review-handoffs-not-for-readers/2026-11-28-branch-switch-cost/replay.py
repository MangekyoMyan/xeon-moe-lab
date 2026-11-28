"""Check the complete ZIP, measured intervals, fixture identity and functional regressions."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from analyze import analyze,lines,partition,read
from check_validation import check

ROOT=Path(__file__).resolve().parent
def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def combined(root):
    from collections import defaultdict
    from analyze import describe
    data=lines(root/"evidence/confirm.jsonl")
    if (root/"evidence/extend.jsonl").exists():data+=lines(root/"evidence/extend.jsonl")
    groups=defaultdict(lambda:defaultdict(dict))
    for r in data:
        if r["variant"] in ("base","new"):
            key=r["size"]+("-resident" if r["duplicate"] else "-fresh")
            groups[key][r["pair"]][r["variant"]]=r["clocks"]
    out={}
    for k,p in groups.items():
        out[k]={}
        for clock in ("switch_end","first","done"):
            seq={v:[(z[v][clock]-z[v]["start"])/1e6 for i,z in sorted(p.items())] for v in ("base","new")}
            d=[a-b for a,b in zip(seq["base"],seq["new"])]
            out[k][clock]={"base":describe(seq["base"]),"new":describe(seq["new"]),
                           "base_minus_new":{**describe(d),"shorter":sum(x>0 for x in d),"raw":d}}
    return out


def structural_checks(root):
    cfg=read(root/"DECISION.json");before=read(root/"REMARK_BEFORE.json");after=read(root/"REMARK_AFTER.json")
    from prior_audit import audit
    assert audit(root/"prior/final-cost.json")==read(root/"PRIOR_AUDIT.json")
    for arm,dirname in (("baseline","baseline-cli"),("candidate","cli")):
        if (root/dirname).is_dir():
            for name,sha in cfg["source_fingerprints"][arm].items():
                assert digest(root/dirname/name)==sha
    assert not before["logical_state_unchanged"] and before["error"]=="mark_exists" and before["after"]["snapshots"]==before["before"]["snapshots"]+1
    assert after["logical_state_unchanged"] and after["before"]==after["after"] and after["error"]=="missing_image"
    points=read(root/"evidence/validation-cases.json")["points"].copy()
    for size in ("short","long"):
        for n,z in read(root/"cases"/(size+".json"))["points"].items():points[size+"-"+n]=z
    tool=read(root/"evidence/tool-edit.json")
    points.update({k:tool[k] for k in ("parent","A","B")})
    for z in points.values():
        f=root/"assets"/(z["sha"]+".slot")
        assert f.stat().st_size==z["bytes"] and digest(f)==z["sha"]
        with f.open("rb") as h:
            magic,ver,words,marker,codec,n=struct.unpack("<IIIiII",h.read(24))
            assert (magic,ver,marker,codec)==(0x67677371,4,-1,1) and n==len(z["tokens"]) and words==n+4
            assert list(struct.unpack("<"+str(n)+"i",h.read(4*n)))==z["tokens"] and h.read(4)==bytes(4)
    checks=0
    for phase in ("discover","confirm","timed","extend"):
        if phase=="extend" and not (root/"evidence/extend.jsonl").exists():continue
        assert analyze(root,phase)==read(root/(phase.upper()+".json"))
        for row in lines(root/"evidence"/(phase+".jsonl")):
            if "trace" not in row:continue
            _,counts=partition(row)
            isnew=row["variant"]=="new-trace"
            assert counts.get("sql_autocommit_used",0)==int(not isnew) and counts.get("sql_used_inside_tx",0)==int(isnew)
            assert all(counts.get(x,0)==1 for x in ("file_fsync","directory_fsync","sql_autocommit_mutating",
                       "sql_commit_outgoing","sql_commit_active","native_save_http","native_restore_http"))
            ev=row["trace"]["events"]
            def event(label):
                return next(x for x in ev if x["label"]==label)
            commits=[x for x in ev if x["label"]=="sql_commit"]
            assert event("sql_autocommit_mutating")["end"]<=event("native_save_http")["begin"]
            assert event("native_save_http")["end"]<=event("file_fsync")["begin"]<=event("file_fsync")["end"]<=commits[0]["begin"]
            assert event("directory_fsync")["end"]<=commits[0]["begin"]<=commits[0]["end"]<=event("native_restore_http")["begin"]
            use=event("sql_used_inside_tx" if isnew else "sql_autocommit_used")
            assert event("native_restore_http")["end"]<=use["begin"]<=use["end"]<=commits[-1]["begin"]
            checks+=1
    assert combined(root)==read(root/"PAIRED.json")
    assert check(root)==read(root/"VALIDATION.json")
    terminal=read(root/"evidence/terminal.json");extra=read(root/"evidence/terminal_extra.json")
    assert len(terminal)==8 and len(extra)==16
    assert {(r["size"],r["pair"],r["version"]) for r in extra}=={("short",i,v) for i in range(2,10) for v in ("base","new")}
    assert all(r["replayed"]==0 and r["answer"].strip()=="731" and r["start_ns"]<=r["switch_process_exited_ns"]<=r["first_pipe_byte_ns"]<=r["done_ns"] for r in terminal+extra)
    return checks


if __name__=="__main__":
    manifest=read(ROOT/"MANIFEST.json");n=0
    for g in manifest.values():
        for name,v in g.items():
            p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and p.stat().st_size==v["bytes"] and digest(p)==v["sha256"],name
            n+=1
    checks=structural_checks(ROOT)
    subprocess.run([sys.executable,"-B","-m","unittest","discover","-s",str(ROOT/"cli"),"-p","test_*.py"],check=True)
    print("Verified",n,"hashes and",checks,"exclusive checkout traces.")
