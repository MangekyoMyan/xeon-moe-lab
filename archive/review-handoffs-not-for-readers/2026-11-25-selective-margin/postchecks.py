"""Keep supplemental sampler and GPU microbenchmarks outside all three model timing processes."""
import csv
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

from prepare import ROOT,LAB,PREV,bias,load,sha,save
from offline import sampler_stage
D=ROOT/"router_micro";E=LAB/"deepseek41-bringup-20261101"
sys.path.insert(0,str(LAB/"volta-residual-20261125"));import run as live


def build():
    original=E/"source/ggml/src/ggml-cuda/topk-moe.cuh"
    text=original.read_text();begin=text.index("struct ggml_cuda_topk_moe_args {");end=text.index("\n};",begin)+3
    (D/"native_router_args.h").write_text(text[begin:end]+"\n")
    b,r=bias();raw=struct.pack("<384f",*b);assert r==load(ROOT/"DESIGN.json")["bias"]
    (D/"bias.local.bin").write_bytes(raw)
    cfg=load(E/"server-config.json");paths=str(ROOT/"backend/runtime")+":"+cfg["environment"]["LD_LIBRARY_PATH"];tk=LAB/"v100-prep-20261024/cuda-12.9.2"
    cmd=["g++","-std=c++17","-O2","-march=ivybridge","-I"+str(ROOT/"backend"),"-I"+str(E/"source/ggml/include"),
         "-I"+str(E/"source/ggml/src"),"-I"+str(tk/"include"),str(D/"router.cpp"),
         *["-L"+p for p in paths.split(":")],"-Wl,-rpath,"+paths,"-Wl,-rpath-link,"+paths,"-lggml-cuda","-lggml-base","-lcudart","-o",str(D/"router")]
    with (D/"build.log").open("w") as f:subprocess.run(cmd,stdout=f,stderr=f,check=True,timeout=90)
    save(D/"BUILD.json",{"command":cmd,"router":sha(D/"router"),"source":sha(D/"router.cpp"),"copied_args_header":sha(original),"bias_sha256":sha(D/"bias.local.bin")})


def main():
    until=time.monotonic()+1800
    while time.monotonic()<until:
        if (ROOT/"RESTORED_MARGIN.json").exists() and load(ROOT/"STATE.json").get("status")=="measurements complete":break
        time.sleep(2)
    else:raise RuntimeError("full-model window not closed")
    cfg=load(E/"server-config.json");who=load(ROOT/"RESTORED_MARGIN.json")["pid"]
    assert live.adopt.process(who)["comm"]=="llama-server"
    baseline=live.api_activity()
    sampler_stage();assert live.api_activity()==baseline
    build()
    run_env={"PATH":"/usr/bin:/bin","LANG":"C",**cfg["environment"]}
    run_env["LD_LIBRARY_PATH"]=str(ROOT/"backend/runtime")+":"+run_env["LD_LIBRARY_PATH"]
    cmd=["taskset","-c","23",str(D/"router"),str(PREV/"runs/precision-64-64"),str(ROOT/"runs/candidate"),str(D/"bias.local.bin"),str(load(ROOT/"DESIGN.json")["threshold"]),str(D/"samples.tsv")]
    snap,host,ecc=live.sample(cfg);assert snap["free_MiB"]>1600 and live.api_activity()==baseline
    observed=[snap]
    with (D/"run.log").open("w") as f:
        child=subprocess.Popen(cmd,env=run_env,stdout=f,stderr=f,stdin=subprocess.DEVNULL)
        try:
            end=time.monotonic()+60
            while child.poll() is None:
                if time.monotonic()>=end or live.api_activity()!=baseline:raise RuntimeError("ordinary work or microbench timeout")
                v,h,e=live.sample(cfg);assert h==host and e==ecc;observed.append(v);time.sleep(.18)
            assert child.returncode==0,(D/"run.log").read_text()[-500:]
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    after,h,e=live.sample(cfg);assert h==host and e==ecc and live.api_activity()==baseline
    observed.append(after)
    with (D/"samples.tsv").open() as f:raw=list(csv.DictReader(f,delimiter="\t"))
    assert len(raw)==96
    stats={}
    for tag in ("JA96","PY96"):
        rows=[r for r in raw if r["case"]==tag];table={}
        for key in ("GPU_us_per_call","wall_us_per_call"):
            a={int(r["pair"]):float(r[key]) for r in rows if r["decision"]=="0"}
            b={int(r["pair"]):float(r[key]) for r in rows if r["decision"]=="1"}
            assert len(a)==len(b)==24 and all(int(r["calls"])==128 for r in rows)
            table[key]={"ordinary_median":statistics.median(a.values()),"decision_median":statistics.median(b.values()),
                        "paired_delta_median":statistics.median(b[i]-a[i] for i in a)}
        stats[tag]=table
    save(ROOT/"ROUTER_MICRO.json",{"native_router_capture_and_ids_verified":True,"calls_per_timed_block":128,"pairs_per_case":24,
          "same_DSO":load(ROOT/"backend/BUILD.json")["sha256"],"stats":stats,"API_pid":who,"ordinary_request_metrics_before_after":baseline,
          "samples":observed,"host_counter_count":len(host),"scope":"CUDA-event and host wall time for standalone fused scoring+selection of 96 token rows x 384 experts. Both arms returned identical selected IDs/weights; decision also matched full-model gap receipts."})
    print(stats,flush=True)


if __name__=="__main__":main()
