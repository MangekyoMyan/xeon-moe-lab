"""Build a matched isolated full-model caller and freeze all prompt and sequence inputs."""
import shutil,subprocess
from prepare import ROOT,LAB,PREV,load,save,sha

D=ROOT/"driver";P=PREV/"precision"
ref=load(P/"BUILD.json")
for name,h in ref["source_sha256"].items():assert sha(P/name)==h
for name in ("affinity_check.h","cache_inspection.h","tracing.h"):shutil.copy2(P/name,D/name)
shutil.copy2(ROOT/"backend/margin_shared.h",D/"margin_shared.h")
src=(P/"driver.cpp").read_text()
a='#include "prefix_cases.h"\n#include "capture.h"\n#include "holdout.h"'
assert src.count(a)==1
src=src.replace(a,'#include "measure.h"')
a='        common_params p;'
assert src.count(a)==1
src=src.replace(a,'        MarginCalls margin(spec);\n'+a)
a='''        measure_prefixes(ctx,resources.batch,out,cases,spec,report,nv);
        measure_chats(ctx,resources.batch,out,cases,spec,report,nv);
        capture_prefixes(ctx,model,resources.batch,out,cases,spec,report,nv);'''
assert src.count(a)==1
src=src.replace(a,'        margin_trials(ctx,resources.batch,out,cases,spec,report,nv,margin);')
(D/"driver.cpp").write_text(src)
cmd=ref["command"].copy()
for i,x in enumerate(cmd):
    if x in ("-I"+str(P),str(P/"driver.cpp"),str(P/"driver")):cmd[i]=x.replace(str(P),str(D))
with (D/"build.log").open("w") as f:subprocess.run(cmd,stdout=f,stderr=f,check=True,timeout=240)
j=load(P/"FIXTURES.json");cases={v["id"]:v for v in j["cases"] if v["id"].endswith("96")}
gold=load(PREV/"runs/precision-64-64/GREEDY.json")
for q in j["chats"]:
    item=gold[q["id"]];ids=item["reference_ids"];assert item["last_is_eog"] and q["ids"]==item["prefix_ids"]
    cases[q["id"]]={"id":q["id"],"prefix":q["ids"],"forced":ids[:-1],"targets":ids,"reference_text":item["reference_text"],"reference_ids":ids}
save(D/"FIXTURES.json",{"cases":cases})
save(D/"BUILD.json",{"driver":sha(D/"driver"),"command":cmd,"sources":{p.name:sha(p) for p in D.iterdir() if p.suffix in (".cpp",".h")},
                     "fixture":sha(D/"FIXTURES.json"),"parent":ref["driver_sha256"]})
print("fixed caller",sha(D/"driver"))
