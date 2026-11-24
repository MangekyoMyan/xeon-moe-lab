"""Assemble the factorial/ground-truth diagnostic from the prior pinned 24/24 prefix probe."""
from pathlib import Path
import shutil
import subprocess
from shared import HERE,PRIOR,read,sha,fresh

D=HERE/"precision"
P=PRIOR/"prefix_quality"
orig=read(P/"BUILD.json")
for name,h in orig["source_sha256"].items():assert sha(P/name)==h
for name in ("affinity_check.h","tracing.h","cache_inspection.h"):shutil.copyfile(P/name,D/name)
s=(P/"driver.cpp").read_text()
s=s.replace('#include "prefix_cases.h"','#include "prefix_cases.h"\n#include "capture.h"\n#include "holdout.h"')
a="        measure_prefixes(ctx,resources.batch,out,cases,spec,report,nv);"
assert s.count(a)==1
s=s.replace(a,a+"\n        measure_chats(ctx,resources.batch,out,cases,spec,report,nv);\n        capture_prefixes(ctx,model,resources.batch,out,cases,spec,report,nv);")
(D/"driver.cpp").write_text(s)
p=(P/"prefix_cases.h").read_text()
a=p.index("    const std::string policy=");z=p.index("    auto compute=",a)
p=p[:a]+'''    int index=0;
    for(const char *key:{"GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"}) {
        const char *actual=std::getenv(key);
        need(actual && std::to_string(spec.at("thresholds").at(index++).get<int>())==actual,"effective factorial settings");
    }
'''+p[z:]
(D/"prefix_cases.h").write_text(p)
cmd=orig["command"].copy()
for i,x in enumerate(cmd):
    if x in ("-I"+str(P),str(P/"driver.cpp"),str(P/"driver")):cmd[i]=x.replace(str(P),str(D))
with (D/"build.log").open("w") as log:subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=240)
fresh(D/"BUILD.json",{"command":cmd,"driver_sha256":sha(D/"driver"),"parent_driver":orig["driver_sha256"],
      "source_sha256":{p.name:sha(p) for p in D.iterdir() if p.suffix in (".cpp",".h")},"libraries":orig["libraries"]})
raw=read(P/"FIXTURES.json")
base=read(PRIOR/"driver/CASES.json")["cases"];items=[]
for n in (96,160):
    for c,prefix in zip(base,("JA","PY")):
        ids=c["prefix_ids"]+c["next_targets_prefix_and64T1"]
        item={"id":prefix+str(n),"prefix":ids[:n],"forced":ids[n:n+64],"targets":ids[n:n+65]}
        if n==96:assert item==next(k for k in raw["cases"] if k["id"]==item["id"])
        items.append(item)
chats=read(HERE/"CHAT_DESIGN.json")["prompts"]
fresh(D/"FIXTURES.json",{"warmup_ja64":raw["warmup_ja64"],"cases":items,"chats":[{"id":name,"ids":v["ids"]} for name,v in chats.items()]})
print("precision caller",sha(D/"driver"))
