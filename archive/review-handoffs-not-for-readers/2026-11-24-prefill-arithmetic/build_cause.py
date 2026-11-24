"""Build a diagnostic that changes only the existing buffers in an isolated layer-zero replay."""
import shutil
import subprocess
from shared import HERE,read,fresh,sha

P=HERE/"precision";D=HERE/"cause_replay"
old=read(P/"BUILD.json")
s=(P/"driver.cpp").read_text().replace('#include "prefix_cases.h"\n','').replace('#include "holdout.h"','#include "cause.h"')
before="""        measure_prefixes(ctx,resources.batch,out,cases,spec,report,nv);
        measure_chats(ctx,resources.batch,out,cases,spec,report,nv);
        capture_prefixes(ctx,model,resources.batch,out,cases,spec,report,nv);"""
assert s.count(before)==1
(D/"driver.cpp").write_text(s.replace(before,"        run_changes(ctx,model,resources.batch,out,cases,spec,report,nv);"))
for name in ("capture.h","affinity_check.h","tracing.h","cache_inspection.h"):
    shutil.copy2(P/name,D/name)
cmd=old["command"].copy()
for i,x in enumerate(cmd):
    if x in ("-I"+str(P),str(P/"driver.cpp"),str(P/"driver")):cmd[i]=x.replace(str(P),str(D))
with (D/"build.log").open("w") as fp:subprocess.run(cmd,stdout=fp,stderr=fp,timeout=240,check=True)
fresh(D/"BUILD.json",{"command":cmd,"driver":sha(D/"driver"),"sources":{p.name:sha(p) for p in D.iterdir() if p.suffix in (".cpp",".h")}})
print("single-layer diagnostic",sha(D/"driver"))
