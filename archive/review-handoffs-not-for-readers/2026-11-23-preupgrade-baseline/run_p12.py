"""Run the three predeclared old-CPU processes in a guarded one-model window."""
import argparse
import fcntl
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

from api_window import active, clear_monitor, window
from measure import compare_quality, dump, floats, read, sha
from observe_quality import observe

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
TRIAL=ROOT/"trial"
ENGINE=LAB/"deepseek41-bringup-20261101"
OLD=LAB/"ecc9-simplification-20261119"
INPUT=ROOT/"intake"
CPU="libggml-cpu.so.0.28.0"
CUDA="libggml-cuda.so.0.28.0"
BASE="libggml-base.so.0.28.0"
LLAMA="libllama.so.0.8.0"

def prepare(stage):
    initial=read(ROOT/"production-before-live.json")
    config=read(ENGINE/"server-config.json")
    assert sha(ENGINE/"server-config.json")==initial["config_sha256"]
    profile=read(INPUT/"reference/current_runtime_633e.json")
    for field in ["model","vision"]:
        group=profile[field]; rec=group["download_receipt"]
        assert sha(LAB/rec["file"])==rec["sha256"]
        for f in group["files"]:assert (LAB/f["file"]).stat().st_size==f["size"]
    baseline=read(INPUT/"reference/timing-0-control.LAUNCH.json")
    for p,h in baseline["guard_source_sha256"].items():assert sha(p)==h,p
    b=read(TRIAL/"BUILD.json")
    for name,h in b["source_sha256"].items():assert sha(TRIAL/name)==h,name
    assert sha(TRIAL/"driver")==b["driver_sha256"] and b["returncode"]==0
    libs=b["libraries"]
    assert {e["path"]:e["sha256"] for e in libs.values()}==initial["mapped_runtime_hashes"]
    for e in libs.values():assert sha(Path(e["path"]))==e["sha256"]
    condition=stage["condition"]
    assert condition==read(ROOT/"TOPOLOGY.json")["measurement"]
    assert len(condition["process_cpu_ids"])==12 and all(condition[k]==12 for k in ("decode_threads","prefill_threads","projector_threads"))
    expected=read(TRIAL/"DESIGN.json")
    assert stage in expected["stages"] and sha(INPUT/"BASELINE_PROTOCOL.json")==expected["protocol_sha256"]
    assert sha(TRIAL/"CASES.json")==read(INPUT/"BASELINE_PROTOCOL.json")["fixed_cases_sha256"]
    env=config["environment"].copy()
    assert int(env["OMP_THREAD_LIMIT"])==condition["OMP_THREAD_LIMIT"]
    assert not any(k in env for k in ("LD_PRELOAD","GGML_CPU_IQ_ROW_LOCAL","GGML_CPU_IQ_AUDIT_DIR","XEON_169_SHARED_PREFIX"))
    env["LD_BIND_NOW"]="1"
    assert sha(Path(env["GGML_CPU_IQ_PLAN"]))==initial["plan_sha256"]==profile["plan"]["sha256"]
    cpulist=",".join(map(str,condition["process_cpu_ids"]))
    prefix=baseline["prefix"].copy()
    assert prefix==["/usr/bin/python3",str(LAB/"moe-implementation-20261026/release/without_thp.py"),"numactl","--physcpubind=0-23","--interleave=0,1"]
    prefix[3]="--physcpubind="+cpulist
    args=baseline["args"].copy()
    # The normal config was verified against the live server before the window.
    for flag in ("-m","-t","-tb","--cpu-mask","--cpu-mask-batch","--cpu-strict","--cpu-strict-batch","-c","-b","-ub","--n-cpu-moe","--mmproj"):
        orig=config["command"];assert args[args.index(flag)+1]==orig[orig.index(flag)+1]
    for flag,key in [("-t","decode_threads"),("-tb","prefill_threads"),("--cpu-mask","decode_mask"),("--cpu-mask-batch","prefill_mask")]:
        assert args.count(flag)==1
        args[args.index(flag)+1]=str(condition[key])
    return {"spec":stage,"prefix":prefix,"args":args,"environment":env,"libraries":libs,
            "original_config_sha256":initial["config_sha256"],"model_download_receipt_sha256":initial["model_download_manifest_sha256"],
            "plan_sha256":initial["plan_sha256"],"driver_sha256":b["driver_sha256"],"build_sha256":sha(TRIAL/"BUILD.json"),
            "design_sha256":sha(TRIAL/"DESIGN.json"),"cases_sha256":sha(TRIAL/"CASES.json"),
            "guard_source_sha256":baseline["guard_source_sha256"],"runner_sha256":sha(Path(__file__)),"reserve_GiB":64}

def providers(launch,out):
    clean={k:v for k,v in os.environ.items() if not k.startswith(("LLVM_","LLAMA_","GGML_","OMP_","GOMP_","LD_","CUDA_"))}
    clean.update(launch["environment"])
    subprocess.run([str(TRIAL/"driver"),"--check-providers",str(out)],env=clean,check=True,timeout=30)
    j=read(out);assert not j["model_loaded"] and len(j["providers"])==5
    for sym in j["providers"]:
        name=sym["symbol"]
        soname=LLAMA if name.startswith("llama_") else BASE if name.startswith("ggml_backend_sched_") else CPU
        path=Path(launch["libraries"][soname]["path"]);st=path.stat()
        assert (sym["path"],sym["device"],sym["inode"],sym["bytes"])==(str(path),st.st_dev,st.st_ino,st.st_size)
    j["active_backend_providers"]=[dict(name=n,graph_compute_provider=launch["libraries"][lib]["path"]) for n,lib in [("CUDA0",CUDA),("CPU",CPU)]]
    dump(out,j)

def verify(run):
    j=read(run/"RUN.json");s=read(run/"SPEC.json");pin=read(run/"PROVIDER_EXPECTED.json");h=read(run/"health.json");c=s["condition"]
    assert h["status"]=="exited" and h["child_returncode"]==0 and j["passed"]
    assert j["spec"]==s and j["requested"]==c and j["providers"]==pin["providers"] and j["active_backend_providers"]==pin["active_backend_providers"]
    assert j["linux_entry"]["pid"]==h["pid"] and j["linux_entry"]["effective_cpu_ids"]==c["process_cpu_ids"]
    assert j["linux_entry"]["mempolicy_mode"]==3 and j["linux_entry"]["mempolicy_nodes_mask"]==3 and j["linux_entry"]["thp_disabled"]
    assert j["parsed_cpu"]=={"decode_count":12,"prefill_count":12,"decode_ids":c["process_cpu_ids"],"prefill_ids":c["process_cpu_ids"],"decode_strict":True,"prefill_strict":True}
    assert j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==12
    assert (j["context"]["n_ctx"],j["context"]["n_batch"],j["context"]["n_ubatch"])==(131072,1024,1024)
    assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==0 and j["after_unload_KiB"]["VmLck"]==0 and j["vision_loaded"]
    assert [p["routed_host"] for p in j["placement"]]==[True]*37+[False]*3 and len(j["shared_inventory"])==37
    assert min(j[k] for k in ("text_model_context_load_ns","vision_init_ns","unload_ns"))>0
    assert j["preparation"]["warmup_end_ns"]>j["preparation"]["warmup_start_ns"] and j["preparation"]["JA_reference_T1"]==4
    q=(s["stage"]=="quality")
    if q:
        reg=j["registration"]
        assert {k:v for k,v in reg["experts"].items() if v}==read(OLD/"PLAN_MEMBERSHIP.json")
        assert len(reg["experts"])==120 and reg["selected_projections"]==19200 and reg["registered_tensors"]==80
        assert reg["sampled_vma_policies"]=={"interleave:0-1":1440} and reg["sampled_pages"]==1440
    else:assert "registration" not in j and "ja_quality_task_observations" not in j
    cases=read(TRIAL/"CASES.json")["cases"]
    assert len(j["blocks"])==len(s["cases"])==2
    comparisons={}
    for b,i in zip(j["blocks"],s["cases"],strict=True):
        case=cases[i]
        assert b["case_index"]==i and b["case_id"]==case["case_id"] and b["prefix_ids"]==case["prefix_ids"] and b["input_tokens"]==case["used_T1_ids"]
        assert b["positions_begin"]==case["prefix_count"] and b["prefill_ready_ns"]>b["prefill_begin_ns"]
        assert len(b["start_ns"])==len(b["output_ready_ns"])==64 and b["status"]==[0]*64 and b["traced_calls"]==0
        assert all(x<y for x,y in zip(b["start_ns"],b["output_ready_ns"])) and all(x>y for x,y in zip(b["start_ns"][1:],b["output_ready_ns"][:-1]))
        assert b["file"]==case["case_id"]+".f32" and b["rows"]==s["rows"]==(65 if q else 1)
        path=run/b["file"]
        assert path.stat().st_size==b["rows"]*129280*4
        raw=path.read_bytes();assert all(map(math.isfinite,floats(raw)))
        if not q:
            ref=TRIAL/"runs/old-p12-quality-01"/b["file"]
            with ref.open("rb") as f:
                f.seek(-len(raw),2);tail=f.read()
            import hashlib
            comparisons[b["file"]]={"byte_equal":raw==tail,"observed_sha256":sha(path),"p12_quality_tail_sha256":hashlib.sha256(tail).hexdigest()}
    return comparisons

def one(stage):
    clear_monitor(ROOT)
    assert not any(active(u) for u in ("llm-deepseek41-api.service","llm-openai-api.service"))
    mem={l.split(":")[0]:int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith(("MemTotal:","MemAvailable:","Mlocked:"))}
    assert mem["MemTotal"]>=480*1024**2 and mem["MemAvailable"]>=360*1024**2 and mem["Mlocked"]<=32*1024**2
    gpu=subprocess.check_output(["/usr/bin/nvidia-smi","--query-gpu=uuid,pci.bus_id","--format=csv,noheader"],text=True,timeout=8).strip().lower()
    assert gpu=="gpu-6622f7a2-ee1a-a79e-1277-cee96d850aa3, 00000000:84:00.0",gpu
    run=TRIAL/"runs"/stage["name"]
    run.mkdir(parents=True);owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    launch=prepare(stage);providers(launch,run/"PROVIDER_EXPECTED.json")
    dump(run/"SPEC.json",stage)
    command=launch["prefix"]+[str(TRIAL/"driver"),"--study-output",str(run),"--study-cases",str(TRIAL/"CASES.json"),"--run-spec",str(run/"SPEC.json")]+launch["args"]
    launch.update(command=command,host_memory_before_kib=mem)
    dump(run/"server-config.json",dict(command=command,environment=launch["environment"]));dump(run/"LAUNCH.json",launch)
    state=read(TRIAL/"STATE.json");state["phase"]=stage["name"];state["runs"][stage["name"]]={"status":"running","run":str(run)}
    dump(TRIAL/"STATE.json",state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(ENGINE))
    spec=importlib.util.spec_from_file_location("preupgrade_guard",ENGINE/"supervise.py")
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.ROOT=run
    host_check=guard.check_host;deadline=time.monotonic()+7200
    def bounded(initial):
        temps,available=host_check(initial)
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError("P12 RAM reserve/time guard")
        return temps,available
    guard.check_host=bounded
    stop=threading.Event();watcher=None
    if stage["quality_host_observation"]:
        watcher=threading.Thread(target=observe,args=(run,ROOT,stop));watcher.start()
    try:
        print("Starting",stage["name"],flush=True);rc=guard.main()
    finally:
        stop.set()
        if watcher is not None:watcher.join(25)
    state["runs"][stage["name"]].update(status="child-exited",returncode=rc)
    dump(TRIAL/"STATE.json",state);assert rc==0,"child or guard failed"
    tails=verify(run)
    if stage["stage"]=="quality":dump(TRIAL/"QUALITY_COMPARISON.json",compare_quality(run))
    else:dump(run/"TAIL_COMPARISON.json",tails)
    dump(run/"RESULTS_SHA256.json",{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!="RESULTS_SHA256.json"})
    state["runs"][stage["name"]].update(status="passed",manifest_sha256=sha(run/"RESULTS_SHA256.json"))
    dump(TRIAL/"STATE.json",state);print("Completed",stage["name"],flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");opts=ap.parse_args()
    stages=read(TRIAL/"DESIGN.json")["stages"]
    if opts.preflight:
        for s in stages:prepare(s)
        providers(prepare(stages[0]),TRIAL/"PREFLIGHT.json")
        print("Three launch specs and original library providers verified without model load.")
        return
    assert os.geteuid()==0
    with (ROOT/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(TRIAL/"STATE.json")
        if state["completed"]:return
        assert not state["runs"],"Review an earlier attempt before resuming"
        for stage in stages:prepare(stage)
        assert read(ROOT/"stream/summary.json")["status"] in ("complete","partial")
        assert read(ROOT/"production-before/snapshot.json")["summary"]["all_small_reference_files_match"]
        assert active("llm-deepseek41-api.service"),"normal API state changed since before snapshot"
        assert read(ENGINE/"health.json")["pid"]==read(ROOT/"production-before-live.json")["pid"]
        with window(ROOT,"P12"):
            for stage in stages:one(stage)
        state=read(TRIAL/"STATE.json")
        state.update(phase="three runs complete, ordinary API restored",completed=True)
        dump(TRIAL/"STATE.json",state)
if __name__=="__main__":main()
