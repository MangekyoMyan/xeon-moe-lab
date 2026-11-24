"""Four output-checked F/S/S/F model processes, same isolated DSO and fixture."""
import json
import os
from pathlib import Path
import subprocess
import time
from study import R,D,LAB,E,inputs,condition,read,write,fresh,sha,guarded_child
from telemetry import start as start_telemetry

W=R/"whole"
STAGES=(("F1",False,[0,1]),("S1",True,[0,1]),("S2",True,[1,0]),("F2",False,[1,0]))


def setup(bank):
    b=read(W/"BUILD.json")
    assert sha(W/"driver")==b["driver_sha256"] and sha(R/"operator/layer7.plan")==read(R/"operator/BUILD.json")["source_sha256"]["layer7.plan"]
    for n,h in b["source_sha256"].items():assert sha(W/n)==h
    for v in b["libraries"].values():assert sha(v["path"])==v["sha256"]
    _,prefix,args,env=inputs()
    ids=condition()["process_cpu_ids"]
    env["LD_LIBRARY_PATH"]=str(W/"runtime")+":"+env["LD_LIBRARY_PATH"]
    env["GGML_XEON_HOT_FFN"]="1" if bank else "0"
    env["GGML_XEON_HOT_PLAN"]=str(R/"operator/layer7.plan")
    env["XEON_L7_STUDY_CPUS"]=" ".join(map(str,ids))
    assert "LD_PRELOAD" not in env
    return b,prefix,args,env


def providers(out,bank):
    b,_,_,env=setup(bank)
    subprocess.run([str(W/"driver"),"--check-providers",str(out)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
    j=read(out);assert not j["model_loaded"] and len(j["providers"])==5
    for v in j["providers"]:
        lib="libllama." if v["symbol"].startswith("llama_") else "libggml-base." if v["symbol"].startswith("ggml_backend_sched_") else "libggml-cpu."
        p=Path(next(x["path"] for k,x in b["libraries"].items() if k.startswith(lib)));s=p.stat()
        assert (v["path"],v["device"],v["inode"],v["bytes"])==(str(p),s.st_dev,s.st_ino,s.st_size)
    names=[("CUDA0","libggml-cuda."),*([("XEON_HOT_FFN","libllama.")] if bank else []),("CPU","libggml-cpu.")]
    j["active_backend_providers"]=[{"name":name,"graph_compute_provider":next(x["path"] for k,x in b["libraries"].items() if k.startswith(lib))} for name,lib in names]
    write(out,j)


def preflight():
    assert read(R/"OPERATOR_STATE.json")["status"].endswith("restored")
    effects=read(R/"OPERATOR.json")["paired_by_membership_and_hits"]
    assert all(effects["331-H"+str(h)]["O_minus_S_ms"]["median_ms"]>0 for h in (1,2)) and effects["331-H1"]["S_minus_P_ms"]["median_ms"]<0
    for bank in (False,True):providers(W/("PREFLIGHT_S.json" if bank else "PREFLIGHT_F.json"),bank)
    design={"stages":STAGES,"membership":{"7":[19,154,331]},"tested_candidate":"serial partition; unchanged resident weights for both language cases",
        "reasons":"Both H1 and H2 saved time in local serial vs O, H1 did not benefit from parallel; 331 much more common than 85 in historical training. Use exactly one candidate.",
        "qualification":"Compare corresponding F/S in both time orders. Use 64-call per-case aggregate and p95, direct finite rows, argmax and target NLL. Require consistent improvement across both workloads/orders for policy adoption; experimental DSOs differ from production."}
    fresh(W/"DESIGN.json",design)


def run_all():
    assert read(W/"DESIGN.json")["stages"]==[list(s) for s in STAGES]
    cases=read(W/"CASES.json")["cases"]
    for name,bank,order in STAGES:
        b,prefix,args,env=setup(bank)
        run=R/"runs"/("whole-"+name);run.mkdir();owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
        fresh(run/"SPEC.json",{"name":name,"stage":"quality","bank":bank,"condition":condition(),"cases":order,"rows":65,"quality_host_observation":False})
        providers(run/"PROVIDER_EXPECTED.json",bank);assert read(run/"PROVIDER_EXPECTED.json")==read(W/("PREFLIGHT_S.json" if bank else "PREFLIGHT_F.json"))
        cmd=prefix+[str(W/"driver"),"--study-output",str(run),"--study-cases",str(W/"CASES.json"),"--run-spec",str(run/"SPEC.json"),*args]
        fresh(run/"server-config.json",{"command":cmd,"environment":env})
        fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"libraries":b["libraries"],"driver_sha256":b["driver_sha256"],"design_sha256":sha(W/"DESIGN.json")})
        write(W/"STATE.json",{"running":name,"unix":time.time()})
        stop,threads=start_telemetry(run,W/"driver",False)
        try:guarded_child(run)
        finally:
            stop.set()
            for th in threads:th.join(20)
        j=read(run/"RUN.json");assert j["passed"] and len(j["blocks"])==2 and j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==24
        assert j["providers"]==read(run/"PROVIDER_EXPECTED.json")["providers"] and j["active_backend_providers"]==read(run/"PROVIDER_EXPECTED.json")["active_backend_providers"]
        assert j["compact_bytes"]==127401984000 and j["after_unload_KiB"]["VmLck"]==j["compact_bytes_after_unload"]==0
        if bank:
            import re
            entries=re.findall(rb"l7-private-worker ordinal=(\d+) cpu=(\d+) tid=(\d+)",(run/"server.stderr.log").read_bytes())
            assert len(entries)==24 and {int(v[1]) for v in entries}==set(range(24)) and len({v[2] for v in entries})==24
        for block,ci in zip(j["blocks"],order):
            case=cases[ci]
            assert block["rows"]==65 and block["all_finite"] and block["status"]==[0]*64 and block["case_id"]==case["case_id"]
            assert block["input_tokens"]==case["used_T1_ids"] and block["prefix_ids"]==case["prefix_ids"]
            f=block["file"];assert (run/f).stat().st_size==65*129280*4
            if not bank:assert (run/f).read_bytes()==(R/"runs/control"/f).read_bytes(),"inactive experimental runtime differs from adopted full control"
            elif name=="S2":assert (run/f).read_bytes()==(R/"runs/whole-S1"/f).read_bytes(),"serial result failed reproducibility"
        fresh(run/"VERIFIED.json",{"pid":j["linux_entry"]["pid"],"bank":bank,"same_numerical_arm_verified":not bank or name=="S2",
              "full_f32_hashes":{q["file"]:sha(run/q["file"]) for q in j["blocks"]}})
        print("whole model",name,"verified",flush=True)
    write(W/"STATE.json",{"status":"four whole-model runs complete","unix":time.time()})
