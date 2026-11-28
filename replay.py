"""Offline verification of the complete adaptive-checkpoint evidence bundle."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from analyze import read,analyze,equivalent
from numeric import verify as numeric
from summary import summarize
from supplement import verify as extra
from source_replay import check as decisions

R=Path(__file__).resolve().parent
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def verify(root,restore_images=False):
    result=analyze(root)
    assert decisions(root)>=99
    assert equivalent(result,read(root/"RESULT.json")) and equivalent(summarize(result),read(root/"SUMMARY.json"))
    assert equivalent(extra(root),read(root/"EXTRA.json"))
    num=numeric(root);assert num==read(root/"NUMERIC.json")
    ref=read(root/"reference/linear.json");image=root/"reference/linear.slot"
    data=image.read_bytes()
    assert hashlib.sha256(data).hexdigest()==ref["image"]["image_sha256"] and len(data)==ref["image"]["image_bytes"]
    magic,version,words,marker,codec,n=struct.unpack("<IIIiII",data[:24])
    assert (magic,version,marker,codec)==(0x67677371,4,-1,1) and words==n+4 and n==len(ref["tokens"])
    assert list(struct.unpack_from("<"+str(n)+"i",data,24))==ref["tokens"] and data[24+4*n:28+4*n]==bytes(4)
    long=read(root/"reference/late.json");pieces=sorted((root/"reference").glob("late.slot.part*"))
    if pieces:
        sha=hashlib.sha256();total=0
        for i,piece in enumerate(pieces):
            assert piece.name==f"late.slot.part{i:02d}"
            raw=piece.read_bytes();sha.update(raw);total+=len(raw)
            if i==0:
                h=struct.unpack("<IIIiII",raw[:24]);count=h[-1]
                assert h[:2]==(0x67677371,4) and list(struct.unpack_from("<"+str(count)+"i",raw,24))==long["tokens"]
        assert sha.hexdigest()==long["image"]["image_sha256"] and total==long["image"]["image_bytes"]
        if restore_images:
            import shutil,tempfile,os
            out=root/"reference/late.slot"
            if out.exists():assert digest(out)==sha.hexdigest()
            else:
                with tempfile.NamedTemporaryFile(dir=out.parent,delete=False) as tmp:
                    for p in pieces:
                        with p.open("rb") as inp:shutil.copyfileobj(inp,tmp)
                    tmp.flush();os.fsync(tmp.fileno())
                Path(tmp.name).replace(out)
    else:
        assert digest(root/"reference/late.slot")==long["image"]["image_sha256"]
    for kind,v in read(root/"evidence/faults.json").items():
        before,after,cp=v["before"],v["after"],v["receipt"]["checkpoint"]
        assert v["receipt"]["turn_committed"] and after["gen"]==before["gen"]+1 and after["pending"]==[1]
        assert len(after["messages"])==len(before["messages"])+2 and after["rev"]==before["rev"]+1+(kind=="cleanup")
        if kind=="cleanup":assert cp["status"]=="saved_cleanup_warning" and cp["snapshot_id"]==after["head"]==after["base"]
        else:assert after["head"] is None and after["base"]==before["base"] and after["snapshots"]==before["snapshots"]
    return {"sessions":len(result["runs"]),"cost_evaluations":len(result["actual_cost_comparisons"]),"numeric":num["equal"],"scores":num["scores"]}

if __name__=="__main__":
    mf=read(R/"MANIFEST.json");count=0
    for group in mf.values():
        for name,spec in group.items():
            p=(R/name).resolve();assert p.is_relative_to(R) and p.stat().st_size==spec["bytes"]
            with p.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==spec["sha256"],name
            count+=1
    v=verify(R,"--restore-images" in sys.argv)
    subprocess.run([sys.executable,"-B","-m","unittest","discover","-s",str(R/"cli"),"-p","test_*.py"],check=True)
    subprocess.run([sys.executable,"-B",str(R/"fault_cases.py"),"--check"],check=True)
    print("Verified",count,"file hashes;",v)
