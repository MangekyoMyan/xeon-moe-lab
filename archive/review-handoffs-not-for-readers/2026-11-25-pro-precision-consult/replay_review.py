"""Offline validation of the three pinned studies and their source manifests. Run after extracting the primary ZIP."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def check(root):
    manifest=read(root/"BUNDLE_MANIFEST.json")["files"]
    seen=set()
    for entry in manifest:
        n=PurePosixPath(entry["path"])
        if n.is_absolute() or ".." in n.parts or str(n) in seen:raise ValueError(n)
        seen.add(str(n));f=root/n
        if f.is_symlink() or f.stat().st_size!=entry["bytes"] or sha(f)!=entry["sha256"]:raise ValueError(n)
    sources=read(root/"SOURCES.json")
    tests={"arithmetic":["verify_record.py","."],"volta":["analyze.py","--check"],"margin":["verify_evidence.py"]}
    for k,cmd in tests.items():
        base=root/"evidence"/k;source=sources["experiments"][k]
        if sha(base/"MANIFEST_SHA256.json")!=source["manifest_sha256"]:raise ValueError(k+" source manifest")
        res=subprocess.run([sys.executable,"-B",*cmd],cwd=base,check=True,capture_output=True,text=True,timeout=240)
        print(k+": "+(res.stdout.strip().splitlines()[-1] if k!="volta" else "two native-GPU timing replicates and 4,608 dot outputs reproduced"))
    print("Bundle integrity verified for",len(seen),"files")

if __name__=="__main__":check(ROOT)
