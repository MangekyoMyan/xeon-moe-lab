"""Run the three predeclared old-CPU processes in a guarded one-model window."""
import argparse
import signal
import postswap_design as ps
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
    expected_condition=ps.conditions(read(ROOT/"TOPOLOGY.json"))[condition["name"]]
    assert condition==expected_condition
    assert initial["config_sha256"]==ps.CONFIG_SHA and b["driver_sha256"]==ps.DRIVER_SHA
    expected=read(TRIAL/"DESIGN.json")
    assert stage in expected["stages"] and sha(INPUT/"BASELINE_PROTOCOL.json")==expected["protocol_sha256"]
    assert sha(TRIAL/"CASES.json")==read(INPUT/"BASELINE_PROTOCOL.json")["fixed_cases_sha256"]
    env=config["environment"].copy()
    assert int(env["OMP_THREAD_LIMIT"])==condition["OMP_THREAD_LIMIT"]
    assert not any(k in env for k in ("LD_PRELOAD","GGML_CPU_IQ_ROW_LOCAL","GGML_CPU_IQ_AUDIT_DIR","XEON_169_SHARED_PREFIX"))
    env["LD_BIND_NOW"]="1"
    original_p12=read(ps.BASELINE/"trial/runs/old-p12-quality-01/LAUNCH.json")
    assert env==original_p12["environment"], "Original P12 performance environment changed"
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
    assert j["parsed_cpu"]=={"decode_count":c["decode_threads"],"prefill_count":c["prefill_threads"],"decode_ids":c["decode_cpu_ids"],"prefill_ids":c["prefill_cpu_ids"],"decode_strict":True,"prefill_strict":True}
    assert j["context"]["threads"]==c["decode_threads"] and j["context"]["threads_batch"]==c["prefill_threads"]
    assert j["projector_init_threads_argument"]==c["projector_threads"]==12
    assert (j["context"]["n_ctx"],j["context"]["n_batch"],j["context"]["n_ubatch"])==(131072,1024,1024)
    assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==0 and j["after_unload_KiB"]["VmLck"]==0 and j["vision_loaded"]
    assert [p["routed_host"] for p in j["placement"]]==[True]*37+[False]*3 and len(j["shared_inventory"])==37
    assert min(j[k] for k in ("text_model_context_load_ns","vision_init_ns","unload_ns"))>0
    assert j["preparation"]["warmup_end_ns"]>j["preparation"]["warmup_start_ns"] and j["preparation"]["JA_reference_T1"]==4
    q=(s["stage"]=="quality")
    if q:
        reg=j["registration"]
        assert {k:v for k,v in reg["experts"].items() if v}==read(ps.BASELINE/"PLAN_MEMBERSHIP.json")
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
            ref=TRIAL/"runs"/(ps.SLUGS[c["name"]]+"-quality-01")/b["file"]
            with ref.open("rb") as f:
                f.seek(-len(raw),2);tail=f.read()
            import hashlib
            comparisons[b["file"]]={"byte_equal":raw==tail,"observed_sha256":sha(path),"condition_quality_tail_sha256":hashlib.sha256(tail).hexdigest()}
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
    for name in ("SPEC.json","PROVIDER_EXPECTED.json"):
        os.chown(run/name,owner.st_uid,owner.st_gid)
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
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError("Post-swap RAM reserve/time guard")
        return temps,available
    guard.check_host=bounded
    stop=threading.Event();watcher=None
    if stage["quality_host_observation"]:
        watcher=threading.Thread(target=observe,args=(run,ROOT,stop));watcher.start()
    watched_signals=(signal.SIGINT,signal.SIGTERM)
    saved_handlers={sig:signal.getsignal(sig) for sig in watched_signals}
    try:
        print("Starting",stage["name"],flush=True);rc=guard.main()
    finally:
        for sig,handler in saved_handlers.items():signal.signal(sig,handler)
        stop.set()
        if watcher is not None:watcher.join(25)
    state["runs"][stage["name"]].update(status="child-exited",returncode=rc)
    dump(TRIAL/"STATE.json",state);assert rc==0,"child or guard failed"
    if guard.stop_requested:raise KeyboardInterrupt("Guard stopped this study by request")
    tails=verify(run)
    if stage["stage"]=="quality":
        condition_name=stage["condition"]["name"]
        if condition_name=="NEW_P12":
            reference=ps.BASELINE/"trial/runs/old-p12-quality-01"
            reference_name="OLD_P12"
        else:
            reference=TRIAL/"runs/new-p12-quality-01"
            reference_name="NEW_P12"
            assert state["runs"]["new-p12-quality-01"]["status"]=="passed"
        result=compare_quality(run,reference,reference_name,condition_name,ps.QUALITY_SHA)
        dump(run/"QUALITY_COMPARISON.json",result)
        gate=set(result["cases"])==set(ps.QUALITY_SHA) and all(r["byte_equal"] for r in result["cases"].values())
    else:
        dump(run/"TAIL_COMPARISON.json",tails)
        gate=set(tails)==set(ps.QUALITY_SHA) and all(r["byte_equal"] for r in tails.values())
    dump(run/"RESULTS_SHA256.json",{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!="RESULTS_SHA256.json"})
    state["runs"][stage["name"]].update(status="passed" if gate else "quality difference",manifest_sha256=sha(run/"RESULTS_SHA256.json"))
    dump(TRIAL/"STATE.json",state)
    assert gate, f'{stage["name"]}: F32 comparison differs; retained evidence' 
    dump(TRIAL/"STATE.json",state);print("Completed",stage["name"],flush=True)

def preflight(stages):
    import tempfile
    ps.verify_live_server_identity()
    ps.verify_design()
    ps.verify_installation()
    for stage in stages:
        prepare(stage)
    with tempfile.TemporaryDirectory(dir=TRIAL) as folder:
        receipt=Path(folder)/"providers.json"
        providers(prepare(stages[0]),receipt)
        saved=TRIAL/"PREFLIGHT.json"
        if saved.exists():
            assert read(receipt)==read(saved), "DSO provider identity changed since preflight"
        else:
            dump(saved,read(receipt))


def main():
    ap=argparse.ArgumentParser(description="Post-swap six-run study; preflight is the default.")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--run",action="store_true",help="Run six fixed stages in one guarded maintenance window")
    opts=ap.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("This controller requires enabled Python checks")
    assert os.geteuid()==0, "Use the existing host's root-controlled benchmark workflow"
    with (ROOT/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(TRIAL/"STATE.json")
        if state["completed"]:
            print("Study completed:",TRIAL/"STATE.json")
            return
        assert not state["runs"], "Partial study: inspect existing stages and the restore receipt before another attempt"
        stages=read(TRIAL/"DESIGN.json")["stages"]
        preflight(stages)
        if not opts.run:
            print("Six fixed specs, runtime and driver providers verified; no model loaded by the study.")
            return
        assert active("llm-deepseek41-api.service")
        assert not active("llm-openai-api.service")
        ps.verify_live_server_identity()
        label="POSTSWAP"
        def interrupt_window(signum,frame):
            raise KeyboardInterrupt(f"Post-swap controller received signal {signum}")
        previous_term=signal.signal(signal.SIGTERM,interrupt_window)
        try:
            with window(ROOT,label):
                for stage in stages:
                    one(stage)
        except BaseException as exc:
            state=read(TRIAL/"STATE.json")
            state.update(completed=False,last_error=repr(exc))
            window_path=ROOT/("API_WINDOW_"+label+".json")
            state["window"]=read(window_path) if window_path.exists() else None
            dump(TRIAL/"STATE.json",state)
            raise
        finally:
            signal.signal(signal.SIGTERM,previous_term)
        state=read(TRIAL/"STATE.json")
        assert len(state["runs"])==6 and all(r["status"]=="passed" for r in state["runs"].values())
        receipt=ROOT/("RESTORED_"+label+".json")
        assert read(receipt)["normal_API_restored"] is True
        state.update(phase="six runs complete; original API restored",completed=True,restore_receipt=str(receipt))
        dump(TRIAL/"STATE.json",state)


if __name__=="__main__":main()
