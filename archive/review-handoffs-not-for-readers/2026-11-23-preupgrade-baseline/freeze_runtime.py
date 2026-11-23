"""Retain the exact local benchmark binaries and identify the server's mapped ELF dependencies."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from measure import read, sha, dump

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
ENGINE=LAB/"deepseek41-bringup-20261101"
def copy_verified(source,target,digest,executable=False):
    with source.open("rb") as f, target.open("xb") as g:shutil.copyfileobj(f,g,length=1048576)
    assert sha(target)==digest
    target.chmod(0o555 if executable else 0o444)
    return {"path":str(target),"sha256":digest,"bytes":target.stat().st_size}

def main():
    restored=read(ROOT/"production-after-live.json")
    pid=restored["pid"]
    assert read(ENGINE/"health.json")["pid"]==pid
    proc=Path("/proc")/str(pid)
    rows=[l.split(None,5) for l in (proc/"maps").read_text().splitlines()]
    paths={l[5] for l in rows if len(l)==6 and l[5].startswith("/") and ".so" in Path(l[5]).name}
    exe=(proc/"exe").resolve()
    assert exe==ENGINE/"build/bin/llama-server"
    paths.add(str(exe))
    pinned=restored["mapped_runtime_hashes"]
    assert set(pinned)<=paths
    dest=ROOT/"local-frozen";dest.mkdir(exist_ok=False)
    files=[]
    for file in sorted(paths):
        p=Path(file); st=p.stat()
        assert all(int(row[4])==st.st_ino and tuple(int(x,16) for x in row[3].split(":"))==(os.major(st.st_dev),os.minor(st.st_dev)) for row in rows if len(row)==6 and row[5]==file)
        h=sha(p)
        if file in pinned:assert h==pinned[file]
        assert (st.st_ino,st.st_size,st.st_mtime_ns)==(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
        rec={"path":file,"sha256":h,"bytes":st.st_size,"mapped_device_verified":True,"mapped_inode":st.st_ino}
        if file in pinned or p.is_relative_to(ENGINE/"build/bin"):
            rec["retained_on_host"]=copy_verified(p,dest/p.name,h,p==exe)
        files.append(rec)
    assert read(ENGINE/"health.json")["pid"]==pid
    b=read(ROOT/"trial/BUILD.json")
    driver=copy_verified(ROOT/"trial/driver",dest/"p12-benchmark-driver",b["driver_sha256"],True)
    dump(ROOT/"FROZEN_ASSETS.json",{"api_pid":pid,"driver":driver,"mapped_elf_files":files,
        "scope":"Local immutable-by-convention copies of the benchmark driver, API server, ggml/llama components and projector support library. Other installed, mapped system/toolkit shared objects are identified by SHA256. Filesystem modes are read-only. All GGUFs remain at their existing paths; their current sizes and historical content receipts are in the snapshots."})
    print(json.dumps({"runtime_files_fingerprinted":len(files),"retained_binaries":1+sum("retained_on_host" in f for f in files),"driver_sha256":driver["sha256"]}))
if __name__=="__main__":main()
