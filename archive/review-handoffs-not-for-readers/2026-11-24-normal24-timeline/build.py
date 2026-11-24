"""Build an isolated caller linked only to the previously verified inference libraries."""
from pathlib import Path
import hashlib
import json
import subprocess

R=Path(__file__).resolve().parent
OLD=R.parent/"preupgrade-baseline-20261123/trial"
D=R/"driver"


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


if __name__=="__main__":
    old=json.loads((OLD/"BUILD.json").read_text())
    assert not (D/"BUILD.json").exists()
    assert old["source_sha256"]["driver.cpp"]=="e1511e02b8c8004d79efd001b69b91f373a6f41100b49a4075430c4c36195e80"
    for n in ("CASES.json","affinity_check.h","cache_inspection.h"):
        assert sha(D/n)==old["source_sha256"][n]
    normal=json.loads((R.parent/"current/runtime.json").read_text())
    assert {str(R.parent/e["file"]):e["sha256"] for e in normal["libraries"].values()}=={v["path"]:v["sha256"] for v in old["libraries"].values()}
    for lib in old["libraries"].values():
        assert sha(Path(lib["path"]))==lib["sha256"]
    cmd=old["command"].copy()
    for before,after in (("-I"+str(OLD),"-I"+str(D)),(str(OLD/"driver.cpp"),str(D/"driver.cpp")),(str(OLD/"driver"),str(D/"driver"))):
        assert cmd.count(before)==1
        cmd[cmd.index(before)]=after
    with (D/"BUILD.stdout").open("x") as out,(D/"BUILD.stderr").open("x") as err:
        p=subprocess.run(cmd,stdout=out,stderr=err,timeout=240)
    rec=dict(command=cmd,returncode=p.returncode,libraries=old["libraries"],parent_source_sha256=old["source_sha256"],
             source_sha256={p.name:sha(p) for p in [D/n for n in ("driver.cpp","tracing.h","affinity_check.h","cache_inspection.h","CASES.json")]})
    if p.returncode==0:rec["driver_sha256"]=sha(D/"driver")
    (D/"BUILD.json").write_text(json.dumps(rec,indent=2)+"\n")
    print("build",p.returncode,rec.get("driver_sha256"))
    raise SystemExit(p.returncode)
