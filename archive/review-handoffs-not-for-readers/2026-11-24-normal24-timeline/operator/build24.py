import hashlib
import json
from pathlib import Path
import subprocess

D=Path(__file__).resolve().parent
OLD=D.parent.parent/"parallel-pilot-20261117/replay"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

if __name__=="__main__":
    b=json.loads((OLD/"BUILD.json").read_text())
    for n,h in b["source_sha256"].items():assert sha(OLD/n)==h
    libs=b["engine_libraries"]["diagnostic"]
    for v in libs.values():assert sha(Path(v["path"]))==v["sha256"]
    cmd=b["command"].copy()
    for a,z in (("-I"+str(OLD),"-I"+str(D)),("-I"+str(OLD/"route37"),"-I"+str(D/"route37")),
                (str(OLD/"driver.cpp"),str(D/"driver.cpp")),(str(OLD/"driver"),str(D/"driver"))):
        cmd[cmd.index(a)]=z
    assert not (D/"BUILD.json").exists()
    with (D/"BUILD.stdout").open("x") as out,(D/"BUILD.stderr").open("x") as err:
        p=subprocess.run(cmd,stdout=out,stderr=err,timeout=240)
    src={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob("*")) if p.is_file() and p.suffix in (".cpp",".h",".plan",".json")}
    j={"returncode":p.returncode,"command":cmd,"source_sha256":src,"libraries":libs,
       "historical_source_sha256":b["source_sha256"],"driver_sha256":sha(D/"driver") if not p.returncode else None}
    (D/"BUILD.json").write_text(json.dumps(j,indent=2)+"\n")
    print("operator build",p.returncode,j["driver_sha256"]);raise SystemExit(p.returncode)
