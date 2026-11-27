from build_capture import HERE,LAB,read,save,sha
import shutil
import subprocess

p=HERE/"capture";d=HERE/"integrated"
for n in ("affinity_check.h","cache_inspection.h","tracing.h"):shutil.copyfile(p/n,d/n)
src=(p/"driver.cpp").read_text().replace('#include "layers.h"','#include <map>\n#include <tuple>\n#include "trial.h"').replace("run_capture(ctx,model,resources.batch,out,cases,report,nv);","actual_trial(ctx,model,resources.batch,out,cases,report,nv);")
(d/"driver.cpp").write_text(src)
c=[x.replace(str(p)+"/",str(d)+"/").replace("-I"+str(p),"-I"+str(d)) for x in read(p/"BUILD.json")["command"]]
cuda=LAB/"v100-prep-20261024/cuda-12.9.2"
c1=[str(cuda/"bin/nvcc"),"-O2","-std=c++17","-arch=sm_70","-Xcompiler","-fPIC","-c",str(HERE/"operator/gather.cu"),"-o",str(d/"gather.o")]
c[1:1]=["-I"+str(cuda/"include"),"-L"+str(cuda/"lib"),"-Wl,-rpath,"+str(cuda/"lib")]
i=c.index(str(d/"driver.cpp"))+1
c[i:i]=[str(d/"hook.cpp"),str(d/"gather.o"),"-lcudart",*["-Wl,--export-dynamic-symbol="+s for s in ("ggml_backend_sched_set_copy_callback","ggml_backend_sched_alloc_graph","ggml_backend_graph_compute_async")]]
with (d/"build.log").open("w") as f:
    for cmd in (c1,c):subprocess.run(cmd,stdout=f,stderr=f,check=True)
save(d/"BUILD.json",{"commands":[c1,c],"binary_sha256":sha(d/"driver"),"gather_source_sha256":sha(HERE/"operator/gather.cu"),
     "source_sha256":{f.name:sha(f) for f in d.iterdir() if f.suffix in (".h",".cpp")},"libraries":read(p/"BUILD.json")["libraries"]})
v=read(HERE/"model-audit/CASES.json");v["audit_folder"]=str(HERE/"runs/full-model-check");save(d/"CASES.json",v)
