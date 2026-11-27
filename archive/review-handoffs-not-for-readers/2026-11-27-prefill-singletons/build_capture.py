from pathlib import Path
import hashlib
import json
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
LAB=HERE.parent
D=HERE/"capture"
P=LAB/"worker-balance-20261124/precision"
def read(p):return json.loads(Path(p).read_bytes())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+"\n")
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

if __name__=="__main__":
    parent=read(P/"BUILD.json")
    for n in ("affinity_check.h","cache_inspection.h","tracing.h"):
        assert sha(P/n)==parent["source_sha256"][n]
        shutil.copyfile(P/n,D/n)
    src=(P/"driver.cpp").read_text()
    start=src.index('#include "prefix_cases.h"');end=src.index("int main(",start)
    src=src[:start]+'#include "layers.h"\n'+src[end:]
    start=src.index("        measure_prefixes(");end=src.index('        report["before_unload_KiB"]',start)
    src=src[:start]+"        run_capture(ctx,model,resources.batch,out,cases,report,nv);\n"+src[end:]
    (D/"driver.cpp").write_text(src)
    cmd=[s.replace(str(P),str(D)) for s in parent["command"]]
    with (D/"build.log").open("w") as f:subprocess.run(cmd,stdout=f,stderr=f,check=True)
    tokens=read(LAB/"normal24-timeline-20261124/prefill/TOKENS.json")["JA512"]
    assert len(tokens)==512 and hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()=="880e0f55cb789260ec93475aef601c3963aabde8f4fe8ff7793373a992826992"
    save(D/"FIXTURE.json",{"id":"JA512","tokens":tokens,"warm":read(P/"FIXTURES.json")["warmup_ja64"]})
    save(D/"BUILD.json",{"command":cmd,"driver_sha256":sha(D/"driver"),"source_sha256":{p.name:sha(p) for p in D.iterdir() if p.suffix in (".cpp",".h")},
         "fixture_sha256":sha(D/"FIXTURE.json"),"libraries":parent["libraries"],"parent_driver_sha256":parent["driver_sha256"]})
