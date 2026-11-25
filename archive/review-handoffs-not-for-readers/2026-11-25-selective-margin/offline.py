"""Compile offline analysis tools and time them outside real-model inference windows."""
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare import ROOT,LAB,PREV,load,sha,save

src=LAB/"deepseek41-bringup-20261101/source"
cfg=load(LAB/"deepseek41-bringup-20261101/server-config.json")
paths=cfg["environment"]["LD_LIBRARY_PATH"]
env={**os.environ,"LD_LIBRARY_PATH":paths}
def build(name,links):
    cmd=["g++","-std=c++17","-O3","-march=ivybridge","-ffp-contract=off","-I"+str(src/"ggml/include"),"-I"+str(src/"include"),
         str(ROOT/(name+".cpp")),"-o",str(ROOT/name),*["-L"+p for p in paths.split(":")],"-Wl,-rpath,"+paths,"-Wl,-rpath-link,"+paths,*links]
    subprocess.run(cmd,check=True,timeout=120,env=env)
    return cmd


def local_stage():
    build_at=time.monotonic_ns()
    cmds=[build("local_error",["-lggml-cpu","-lggml-base"]),build("output_cost",["-lllama"])]
    weights=load(PREV/"precision/WEIGHT_SOURCE.json")
    shard=next(iter({v["shard"] for v in weights.values()}))
    d=PREV/"runs/precision-64-64"
    subprocess.run(["/usr/bin/taskset","-c","23",str(ROOT/"local_error"),shard,str(d),str(ROOT/"dot_reference.tsv")],check=True,env=env)
    save(ROOT/"OFFLINE_BUILD.json",{"start_ns":build_at,"end_ns":time.monotonic_ns(),"commands":cmds,"source":{k:sha(ROOT/(k+".cpp")) for k in ("local_error","output_cost")}})


def sampler_stage():
    cases=load(ROOT/"driver/FIXTURES.json")["cases"];r=PREV/"runs/precision-64-64"
    names=sorted(cases)
    paths=[r/((n if n in ("JA96","PY96") else "hold-"+n)+".f32") for n in names]
    begun=time.monotonic_ns()
    subprocess.run(["/usr/bin/taskset","-c","23",str(ROOT/"output_cost"),str(ROOT/"output_cost.tsv"),*map(str,paths)],check=True,env=env,timeout=120)
    save(ROOT/"SAMPLER_INPUTS.json",{"clock_begin_ns":begun,"clock_end_ns":time.monotonic_ns(),"order":names,
                "rows":{n:len(cases[n]["targets"]) for n in names},"input_hashes":{n:sha(p) for n,p in zip(names,paths)}})

if __name__=="__main__":
    if sys.argv[1]=="local":local_stage()
    elif sys.argv[1]=="sampler":sampler_stage()
