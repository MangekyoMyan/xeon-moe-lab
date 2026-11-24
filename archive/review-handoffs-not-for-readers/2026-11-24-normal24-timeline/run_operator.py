"""Matched layer-7 assignments on two actual 24-worker private CPU pools."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from study import R,LAB,E,P,NEW,adopt,read,write,fresh,sha,inputs,guarded_child,maintenance,condition
from telemetry import start as start_telemetry

D=R/"operator";OLD=LAB/"parallel-pilot-20261117/replay"
HIST=OLD/"runs/20261117T053052Z-U1024-1563d936"
REF=LAB/"route37-calibration-20261117/runs/20261117T012730Z-F-before-a30e72f7"


def prepare():
    timeline=read(R/"TIMELINE.json")
    assert read(R/"RESTORED_TIMELINE.json")["config_sha256"]==NEW
    assert all(v["by_layer"]["7"]["cpu_ns"]>0 and v["by_layer"]["7"]["device_overlap_ns"]==0 for v in timeline["calls"])
    assert timeline["median_ms"]["routed37_ns"]>timeline["median_ms"]["device_ns"]
    b=read(D/"BUILD.json"); assert b["returncode"]==0 and sha(D/"driver")==b["driver_sha256"]
    for n,h in b["source_sha256"].items():assert sha(D/n)==h,n
    for e in b["libraries"].values():assert sha(e["path"])==e["sha256"]
    history=read(HIST/"LAUNCH_MANIFEST.json")
    for path,h in history["guard_source_sha256"].items():assert sha(path)==h
    assert sha(REF/"RESULTS_SHA256.json")==history["reference_manifest_sha256"]
    for n,h in read(REF/"RESULTS_SHA256.json").items():assert sha(REF/n)==h
    cap=read(D/"CAPTURE_PIN.json");fixture=Path(cap["directory"])
    assert sha(D/"CAPTURE_PIN.json")==history["capture_pin_sha256"]
    for n,h in cap["files"].items():
        assert Path(n).name==n and sha(fixture/n)==h
    _,prefix,args,baseenv=inputs()
    env=history["environment"].copy()
    # The historical operator did not declare the BLAS ceiling added to the
    # normal environment. Match the adopted runtime explicitly.
    assert baseenv["OPENBLAS_NUM_THREADS"]=="1" and env.get("OPENBLAS_NUM_THREADS","1")=="1"
    env["OPENBLAS_NUM_THREADS"]=baseenv["OPENBLAS_NUM_THREADS"]
    for k,v in baseenv.items():
        if k!="LD_LIBRARY_PATH":assert env[k]==v,k
    env["GGML_XEON_HOT_PLAN"]=str(D/"layer7.plan");assert env["GGML_XEON_HOT_FFN"]=="1" and "LD_PRELOAD" not in env
    return b,fixture,prefix,args,env


def providers(out):
    b,_,_,_,env=prepare()
    subprocess.run([str(D/"driver"),"--check-providers",str(out)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
    j=read(out);assert len(j["providers"])==16 and j["model_loaded"] is False
    for item in j["providers"]:
        k=item["symbol"]
        name="libllama.so.0.8.0" if k.startswith("llama_") else "libggml-base.so.0.28.0" if k.startswith("ggml_backend_sched_") else "libggml-cpu.so.0.28.0"
        path=Path(b["libraries"][name]["path"]);st=path.stat()
        assert (item["path"],item["device"],item["inode"],item["bytes"])==(str(path),st.st_dev,st.st_ino,st.st_size)
    j["active_backend_providers"]=[{"name":n,"graph_compute_provider":b["libraries"][lib]["path"]} for n,lib in (
        ("CUDA0","libggml-cuda.so.0.28.0"),("CPU","libggml-cpu.so.0.28.0"))]
    j["CUDA_VISIBLE_DEVICES"]=env["CUDA_VISIBLE_DEVICES"]
    write(out,j)


def main():
    assert __debug__ and len(sys.argv)==2
    b,fixture,prefix,args,env=prepare()
    design={"physical_ids":condition()["process_cpu_ids"],"stage":"one layer O/C/S/P, local endpoint only",
            "layers":[7],"period_physical_slot2":[331,85,85,331],"slots0_1":[19,154],"native_bytes":36218880,
            "cell_indices":[0,1,2],"expected_hits":[0,1,2],"active_choices_per_token":6,
            "fixed_comparisons":"O/C all CPU, S/P identical partition; O=C=captured, S=P, non-offloaded slots byte exact. F64 reference outside clocks.",
            "endpoint":"host x + route IDs ready to canonical 6 host outputs; includes partition, CPU, GPU, joins and scatter.",
            "timing":"12 balanced blocks per cell per period, four modes, four first-use and 4 untimed warmups each mode.",
            "production_profile_sha256":NEW,"diagnostic_libraries":b["libraries"],
            "capacity":"existing GPU guards strictly before vision, 3072/2560 MiB, verify all three uploaded triplets per period"}
    if sys.argv[1]=="preflight":
        fresh(D/"DESIGN24.json",design);providers(D/"PREFLIGHT.json");print("Layer 7 diagnostic 16 provider pins passed");return
    assert sys.argv[1]=="run" and os.geteuid()==0 and read(D/"DESIGN24.json")==design
    run=R/"runs/operator"
    def interrupted(sig,frame):raise KeyboardInterrupt(f"operator controller signal {sig}")
    for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,interrupted)
    with (R/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not run.exists()
        with maintenance("OPERATOR"):
            run.mkdir();owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
            providers(run/"PROVIDER_EXPECTED.json");assert read(run/"PROVIDER_EXPECTED.json")==read(D/"PREFLIGHT.json")
            cmd=prefix+[str(D/"driver"),"--study-output",str(run),"--study-cases",str(D/"CASES.json"),
                "--wiring-reference",str(REF),"--fixture",str(fixture),*args]
            fresh(run/"server-config.json",{"command":cmd,"environment":env})
            fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"driver_sha256":b["driver_sha256"],
                                    "reference_manifest":sha(REF/"RESULTS_SHA256.json"),"capture_pin":sha(D/"CAPTURE_PIN.json"),"design_sha256":sha(D/"DESIGN24.json")})
            stop,threads=start_telemetry(run,D/"driver",False)
            try:guarded_child(run)
            finally:
                stop.set()
                for t in threads:t.join(20)
            rep=read(run/"RUN.json");h=read(run/"health.json")
            assert rep["passed"] and rep["entry"]["pid"]==h["pid"] and h["child_returncode"]==0 and h["status"]=="exited"
            assert rep["cparams"]["threads"]==rep["cparams"]["threads_batch"]==rep["projector_threads_argument"]==24
            for i in range(1,5):
                j=read(run/f"period-{i}/REPLAY.json")
                assert j["passed"] and len(j["cells"])==3 and j["bank_native_bytes_verified"]==36218880
                assert j["private_initialized_layers"]==[7] and len(j["empty_layers_bypassed"])==36
                for field in ("O_worker_receipt","CSP_worker_receipt"):
                    assert {w["cpu"] for w in j[field]}==set(range(24)) and len({w["tid"] for w in j[field]})==24
                assert [c["matched_H"] for c in j["cells"]]==[0,1,2]
                assert all(c["passed"] for c in j["cells"])
        write(R/"OPERATOR_STATE.json",{"status":"four periods passed and normal 24/24 restored","unix":time.time()})


if __name__=="__main__":main()
