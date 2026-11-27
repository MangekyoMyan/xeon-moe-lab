from build_capture import HERE,LAB,read,save,sha
import shutil
import subprocess
p=HERE/"capture";d=HERE/"model-audit";d.mkdir(exist_ok=True)
for n in ("affinity_check.h","cache_inspection.h","tracing.h","model_audit.h","copy_ledger.cpp"):shutil.copyfile(p/n,d/n)
src=(p/"driver.cpp").read_text().replace('#include "layers.h"','#include <map>\n#include "model_audit.h"').replace("run_capture(ctx,model,resources.batch,out,cases,report,nv);","run_audit(ctx,model,resources.batch,out,cases,report,nv);")
(d/"driver.cpp").write_text(src)
parent=read(p/"BUILD.json");c=[x.replace(str(p)+"/",str(d)+"/").replace("-I"+str(p),"-I"+str(d)) for x in parent["command"]]
c[c.index(str(d/"driver.cpp"))+1:c.index(str(d/"driver.cpp"))+1]=[str(d/"copy_ledger.cpp"),"-Wl,--export-dynamic-symbol=ggml_backend_sched_set_copy_callback","-Wl,--export-dynamic-symbol=ggml_backend_tensor_set_async"]
with (d/"build.log").open("w") as f:subprocess.run(c,stdout=f,stderr=f,check=True)
save(d/"BUILD.json",{"command":c,"driver_sha256":sha(d/"driver"),"source_sha256":{f.name:sha(f) for f in d.iterdir() if f.suffix in (".h",".cpp")},"libraries":parent["libraries"]})
l=read(LAB/"parallel-pilot-20261117/replay/CASES.json")["cases"][1]["expected_token_ids"]
index=read(HERE/"RUNS_INDEX.json");cases=[]
for name,file in (("JA512","FIXTURE.json"),("PY512","PY512.json")):
    v=read(p/file)
    cases.append({"id":name,"tokens":v["tokens"],"forced":l[511:527] if name=="JA512" else [],"rendered":v.get("rendered",""),
        "capture":str(HERE/"runs"/("capture-"+name)),"candidates":{str(i):str(HERE/"runs"/index[name][str(i)]["primary"]/"P-sum.f32") for i in (4,24)}})
save(d/"CASES.json",{"cases":cases,"warm":read(p/"FIXTURE.json")["warm"]})
