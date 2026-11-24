from pathlib import Path
import hashlib
import json
import subprocess

D=Path(__file__).resolve().parent
BASE=D.parent/"driver"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

if __name__=="__main__":
    orig=json.loads((BASE/"BUILD.json").read_text());cmd=orig["command"].copy()
    for a,z in (("-I"+str(BASE),"-I"+str(D)),(str(BASE/"driver.cpp"),str(D/"driver.cpp")),(str(BASE/"driver"),str(D/"driver"))):
        cmd[cmd.index(a)]=z
    with (D/"BUILD.log").open("x") as out:
        subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=240,check=True)
    r={"command":cmd,"returncode":0,"parent_driver_sha256":orig["driver_sha256"],"libraries":orig["libraries"],
       "driver_sha256":sha(D/"driver"),"source_sha256":{k:sha(D/k) for k in ("driver.cpp","prefix_cases.h","tracing.h","cache_inspection.h","affinity_check.h")}}
    (D/"BUILD.json").write_text(json.dumps(r,indent=2)+"\n")
    print("prefix probe built",r["driver_sha256"])
