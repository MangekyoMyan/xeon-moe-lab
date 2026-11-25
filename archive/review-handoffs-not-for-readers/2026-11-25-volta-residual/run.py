"""Run only our small resident GPU process while the existing API is ready and idle."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from build import ROOT,LAB,read,save,sha
sys.path.insert(0,str(LAB/"adopt24-20261124"))
import adopt


def api_activity():
    status,slots=adopt.get("/slots",True)
    assert status==200 and slots
    status,metrics=adopt.get("/metrics",True,True)
    assert status==200
    values={k:float(v) for k,v in re.findall(r"(?m)^llamacpp:(requests_processing|requests_deferred|prompt_tokens_total|tokens_predicted_total)\s+([0-9.eE+-]+)\s*$",metrics)}
    assert len(values)==4,sorted(values)
    if any(v.get("is_processing") for v in slots) or values["requests_processing"] or values["requests_deferred"]:
        raise RuntimeError("ordinary API is processing a request")
    return values


def sample(cfg):
    w=adopt.monitor()
    g=subprocess.check_output(["nvidia-smi","-i",cfg["environment"]["CUDA_VISIBLE_DEVICES"],
        "--query-gpu=temperature.gpu,memory.free,power.limit,ecc.mode.current,power.draw","--format=csv,noheader,nounits"],
        text=True,timeout=3).strip().split(", ")
    snap={"unix":time.time(),"gpu_C":float(g[0]),"free_MiB":float(g[1]),"power_limit_W":float(g[2]),"ecc":g[3],"power_W":float(g[4]),
          "cpu_C":max(v for k,v in w["cpu"].items() if "Package" in k),"hbm_C":w["gpu"]["hbm_C"],
          "fan_pct":max(w["bmc"]["fan_percent"]),"DIMM_C":max(w["bmc"]["dimms_C"].values())}
    if snap["gpu_C"]>=60 or snap["hbm_C"]>=65 or snap["cpu_C"]>=60 or snap["free_MiB"]<1024 or snap["power_limit_W"]!=150 or snap["ecc"]!="Enabled":
        raise RuntimeError("room or protection condition changed")
    return snap,w["host_counters"],[w["gpu"][k] for k in ("ecc_single","ecc_double")]


def run(order):
    assert order in ("forward","reverse")
    spec=read(ROOT/"DESIGN.json");build=read(ROOT/"BUILD.json");before=read(ROOT/"API_BEFORE.json")
    cfg=read(adopt.CONFIG)
    assert sha(adopt.CONFIG)==spec["normal_config_sha256"]==before["config_sha256"]==adopt.NEW
    assert sha(ROOT/"probe")==build["probe_sha256"] and build["spec_sha256"]==sha(ROOT/"DESIGN.json")
    assert all(sha(Path(p))==v for p,v in spec["frozen_captures"].items())
    assert all(sha(Path(p))==v for p,v in spec["mapped_runtime"].items())
    who=before["pid"];assert adopt.unit()["ActiveState"]=="active" and adopt.process(who)["comm"]=="llama-server"
    ready,body=adopt.get("/health");assert ready==200 and body["status"]=="ok"
    live={int(v) for v in subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).split()}
    assert live=={who},"other GPU activity"
    counts=api_activity();s,host,ecc=sample(cfg)
    assert s["free_MiB"]>=1600
    folder=ROOT/("run-"+time.strftime("%H%M%S"));folder.mkdir()
    command=["/usr/bin/taskset","-c","23",str(ROOT/"probe"),str(ROOT/"DESIGN.json"),str(folder),order]
    env={"PATH":"/usr/bin:/bin","LANG":"C",**cfg["environment"]}
    receipt={"command":command,"server_pid":who,"begin":s,"initial_metrics":counts,"samples":[s],"passed":False}
    save(folder/"GUARD.json",receipt)
    with (folder/"probe.log").open("w") as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            deadline=time.monotonic()+120
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError("bounded GPU replay")
                if api_activity()!=counts:raise RuntimeError("ordinary API received work; release diagnostic resources")
                snap,host2,ecc2=sample(cfg);assert host==host2 and ecc==ecc2,"ECC or thermal counter changed"
                receipt["samples"].append(snap)
                time.sleep(.25)
            if child.returncode:raise RuntimeError("diagnostic process failed; inspect "+str(folder/"probe.log"))
            assert read(folder/"RUN.json")["passed"] and api_activity()==counts
            ready,body=adopt.get("/health");assert ready==200 and body["status"]=="ok"
            assert adopt.process(who)["comm"]=="llama-server" and sha(adopt.CONFIG)==before["config_sha256"]
            end,host2,ecc2=sample(cfg);assert host==host2 and ecc==ecc2
            receipt.update(passed=True,final_metrics=counts,final=end,unchanged_host_counters=len(host),gpu_ECC_delta=[0,0])
            save(ROOT/("MEASURED-"+order+".json"),{"run":folder.name,"server_pid":who,"unchanged_normal_config":sha(adopt.CONFIG)})
        except BaseException as exc:
            receipt["error"]=repr(exc)
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            raise
        finally:
            receipt["probe_pid"]=child.pid;receipt["probe_exit"]=child.poll();save(folder/"GUARD.json",receipt)
    print(folder.name,read(folder/"RUN.json"),flush=True)

if __name__=="__main__":
    orders=("forward","reverse") if sys.argv[1:]==["both"] else tuple(sys.argv[1:])
    if not orders or any(o not in ("forward","reverse") for o in orders):
        raise SystemExit("choose forward, reverse or both")
    for order in orders:
        run(order)
