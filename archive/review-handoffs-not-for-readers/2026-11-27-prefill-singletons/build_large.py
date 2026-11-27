from build_capture import HERE,LAB,read,save,sha
import shutil
import subprocess
p=HERE/"integrated";d=HERE/"large"
for n in ("affinity_check.h","cache_inspection.h","tracing.h","storage.h"):shutil.copyfile(p/n,d/n)
h=(p/"hook.cpp").read_text()
assert h.count("int T=512,K=6")==1
h=h.replace("int T=512,K=6","int T=1024,K=6")
anchor="if(plan->gs.empty()|| (mode!=1&&plan->cs.empty())){plan.reset();return false;}"
assert h.count(anchor)==1
h=h.replace(anchor,'if(plan->gs.empty()|| (mode!=1&&plan->cs.empty())){records.push_back({{"layer",l},{"cpu_pairs",0},{"gpu_pairs",T*K},{"native_fallback",true}});plan.reset();return false;}')
(d/"hook.cpp").write_text(h)
src=(p/"driver.cpp").read_text().replace('#include "trial.h"','#include "extended.h"').replace("actual_trial(ctx,model,resources.batch,out,cases,report,nv);","run_extended(ctx,model,resources.batch,out,cases,report,nv);")
(d/"driver.cpp").write_text(src)
cmds=[[s.replace(str(p)+"/",str(d)+"/").replace("-I"+str(p),"-I"+str(d)) for s in cmd] for cmd in read(p/"BUILD.json")["commands"]]
with (d/"build.log").open("w") as f:
    for cmd in cmds:subprocess.run(cmd,stdout=f,stderr=f,check=True)
save(d/"BUILD.json",{"commands":cmds,"binary_sha256":sha(d/"driver"),"source_sha256":{f.name:sha(f) for f in d.iterdir() if f.suffix in (".cpp",".h")},
    "gather_source_sha256":sha(HERE/"operator/gather.cu"),"libraries":read(p/"BUILD.json")["libraries"]})
