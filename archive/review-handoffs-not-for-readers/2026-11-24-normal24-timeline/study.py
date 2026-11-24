"""One guarded 24/24 control and one matched CUPTI process; restore the adopted API."""
import array
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from telemetry import read,write,start as start_telemetry

R=Path(__file__).resolve().parent
LAB=R.parent;D=R/"driver";E=LAB/"deepseek41-bringup-20261101";A=LAB/"adopt24-20261124"
P=LAB/"postload-measurement-20261124"
sys.path.insert(0,str(A))
import adopt
NEW=adopt.NEW
CHECK=LAB/"results-review-20261124/tools/check_live_identity.py"
EXPECTED=LAB/"results-review-20261124/reference/EXPECTED_NORMAL_24_24.json"
OBS=LAB/"169a-exploration-20261119/astra62/observer/libtrace169.so"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def fresh(path,data):
    with path.open("x") as f:json.dump(data,f,indent=2);f.write("\n")


def condition():
    cfg=read(E/"server-config.json")
    assert sha(E/"server-config.json")==NEW and read(LAB/"current/runtime.json")["service_config"]["sha256"]==NEW
    topology=json.loads(subprocess.check_output([sys.executable,"-I",str(LAB/"current/check.py"),"topology"]))
    ids=topology["one_available_thread_per_core"]
    assert ids==list(range(24)) and topology["physical_cores_online"]==24 and topology["logical_cpus_online"]==48
    cmd=cfg["command"]
    v=lambda opt:cmd[cmd.index(opt)+1]
    assert all(v(arg)=="24" for arg in ("-t","-tb")) and all(v(arg)==topology["physical_cpu_mask"] for arg in ("--cpu-mask","--cpu-mask-batch"))
    assert all(v(arg)=="1" for arg in ("--cpu-strict","--cpu-strict-batch"))
    return dict(name="adopted_normal24",decode_threads=24,prefill_threads=24,projector_threads=24,packing_threads_at_load=12,
                decode_cpu_ids=ids,prefill_cpu_ids=ids,process_cpu_ids=ids,decode_mask=v("--cpu-mask"),prefill_mask=v("--cpu-mask-batch"),
                strict=True,interleave_nodes=[0,1],OMP_THREAD_LIMIT=24)


def inputs():
    build=read(D/"BUILD.json")
    assert build["returncode"]==0 and sha(D/"driver")==build["driver_sha256"]
    for name,h in build["source_sha256"].items():assert sha(D/name)==h
    pin=read(A/"DIAGNOSTIC_INPUTS.json")["observer_and_cupti"]
    for path,h in pin.items():assert sha(path)==h
    before=read(R/"BEFORE.json")
    assert before["config_sha256"]==NEW and before["mapped_runtime_hashes"]==read(EXPECTED)["mapped_runtime_hashes"]
    for entry in build["libraries"].values():
        assert sha(entry["path"])==entry["sha256"]==before["mapped_runtime_hashes"][entry["path"]]
    inherited=read(P/"trial/runs/new-d24-b12-quality-01/LAUNCH.json")
    for path,h in inherited["guard_source_sha256"].items():assert sha(path)==h
    current=read(E/"server-config.json");cmd=current["command"]
    args=inherited["args"].copy()
    for k in ("-tb","--cpu-mask-batch"):args[args.index(k)+1]=cmd[cmd.index(k)+1]
    # All diagnostic inference flags are present in the actual server invocation, in the same order.
    assert len(args)==len(inherited["args"]) and args[args.index("-t")+1]==args[args.index("-tb")+1]=="24"
    pos=cmd.index(str(E/"build/bin/llama-server"))+1
    for arg in args:
        pos=cmd.index(arg,pos)+1
    env=current["environment"].copy()
    assert all(k not in env for k in ("LD_PRELOAD","GGML_CPU_IQ_ROW_LOCAL","XEON_169_SHARED_PREFIX","GGML_CPU_IQ_AUDIT_DIR"))
    env["LD_BIND_NOW"]="1"
    assert env==inherited["environment"]
    assert sha(env["GGML_CPU_IQ_PLAN"])==before["plan_sha256"]
    assert sha(D/"CASES.json")=="0458b7a358d0eb3a0c289eb1a1aa9903ee0bb972158acc64abbf0952f472d746"
    return build,cmd[:cmd.index(str(E/"build/bin/llama-server"))],args,env


def providers(stage,out):
    b,_,_,env=inputs()
    if stage=="observe":env["LD_PRELOAD"]=str(OBS)
    subprocess.run([str(D/"driver"),"--check-providers",str(out)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
    data=read(out);assert data["model_loaded"] is False and len(data["providers"])==5
    for sym in data["providers"]:
        lib="llama" if sym["symbol"].startswith("llama_") else "ggml-base" if sym["symbol"].startswith("ggml_backend_sched_") else "ggml-cpu"
        target=next(v for k,v in b["libraries"].items() if k.startswith("lib"+lib+".so"))
        st=Path(target["path"]).stat()
        assert (sym["path"],sym["device"],sym["inode"],sym["bytes"])==(target["path"],st.st_dev,st.st_ino,st.st_size)
    data["active_backend_providers"]=[{"name":name,"graph_compute_provider":next(v["path"] for k,v in b["libraries"].items() if k.startswith(lib+".so"))}
                      for name,lib in (("CUDA0","libggml-cuda"),("CPU","libggml-cpu"))]
    write(out,data)


def plan():
    return {"profile_sha256":NEW,"cases_sha256":sha(D/"CASES.json"),"condition":condition(),"stages":["control","observe"],
            "early_T1s_per_case":4,"full_reference_rows_per_case":65,"normal_placement":"37 CPU / 3 GPU routed; 37 shared GPU",
            "measurement":"Output-ready clocks; device timestamps from CUPTI; IMC counter read/write system-wide; sampled CPU PCs only on observed calls 5..64.",
            "next_operator_gate":"Consider saved layer 7 H0/H1/H2 if routed CPU occupies meaningful wall time and GPU activity leaves room in those same dispatches. Evaluate endpoint with packing, copy and join included."}


def verify_run(run):
    rep=read(run/"RUN.json");spec=read(run/"SPEC.json");health=read(run/"health.json");pin=read(run/"PROVIDER_EXPECTED.json")
    assert rep["passed"] and health["status"]=="exited" and health["child_returncode"]==0 and rep["linux_entry"]["pid"]==health["pid"]
    assert rep["requested"]==spec["condition"]==condition() and rep["spec"]==spec and rep["providers"]==pin["providers"] and rep["active_backend_providers"]==pin["active_backend_providers"]
    assert rep["context"]["threads"]==rep["context"]["threads_batch"]==rep["projector_init_threads_argument"]==24
    assert rep["linux_entry"]["effective_cpu_ids"]==list(range(24)) and rep["linux_entry"]["mempolicy_mode"]==3 and rep["linux_entry"]["mempolicy_nodes_mask"]==3 and rep["linux_entry"]["thp_disabled"]
    assert [row["routed_host"] for row in rep["placement"]]==[True]*37+[False]*3 and len(rep["shared_inventory"])==37
    assert rep["compact_bytes"]==127401984000 and rep["compact_bytes_after_unload"]==rep["after_unload_KiB"]["VmLck"]==0
    reg=rep["registration"];assert {k:v for k,v in reg["experts"].items() if v}==read(LAB/"preupgrade-baseline-20261123/PLAN_MEMBERSHIP.json")
    assert len(rep["blocks"])==2
    hashes={}
    for i,(b,c) in enumerate(zip(rep["blocks"],read(D/"CASES.json")["cases"])):
        assert b["case_index"]==i and b["case_id"]==c["case_id"] and b["prefix_ids"]==c["prefix_ids"] and b["input_tokens"]==c["used_T1_ids"] and b["positions_begin"]==len(c["prefix_ids"])
        assert b["status"]==[0]*64 and b["rows"]==65 and b["traced_calls"]==(4 if spec["stage"]=="observe" else 0)
        assert len(b["start_ns"])==len(b["output_ready_ns"])==64 and all(x<y for x,y in zip(b["start_ns"],b["output_ready_ns"])) and all(x<y for x,y in zip(b["output_ready_ns"],b["start_ns"][1:]))
        assert b["file"]==c["case_id"]+".f32"
        raw=(run/b["file"]).read_bytes();vals=array.array("f");vals.frombytes(raw)
        assert sys.byteorder=="little" and len(vals)==65*129280 and all(map(math.isfinite,vals))
        hashes[b["file"]]=hashlib.sha256(raw).hexdigest()
        if spec["stage"]=="observe":assert raw==(R/"runs/control"/b["file"]).read_bytes(),"observer changed fixed logits"
    fresh(run/"VERIFIED.json",{"pid":health["pid"],"shape_per_case":[65,129280],"all_values_finite":True,"output_sha256":hashes,
        "compared_byte_for_byte_to_matched_control":spec["stage"]=="observe"})


def guarded_child(run):
    assert __debug__ and os.geteuid()==0
    assert adopt.unit()["ActiveState"]=="inactive" and not (run/"STOP_REQUESTED").exists()
    assert not any(p.exists() for p in (E/"STOP_REQUESTED",LAB/"moe-engine-20261029/STOP_REQUESTED"))
    mem={k:int(v.split()[0]) for k,v in (s.split(":",1) for s in Path("/proc/meminfo").read_text().splitlines())}
    assert mem["MemAvailable"]>=360*1024**2 and mem["Mlocked"]<=32*1024**2
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    sys.path.insert(0,str(E))
    src=E/"supervise.py";spec=importlib.util.spec_from_file_location("timeline_native_guard",src)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.ROOT=run
    check=guard.check_host;deadline=time.monotonic()+7200
    def bounded(initial):
        cpu,available=check(initial)
        if available<64*1024**2 or time.monotonic()>deadline:raise RuntimeError("diagnostic RAM reserve or time limit")
        return cpu,available
    guard.check_host=bounded
    saved={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    try:ret=guard.main()
    finally:
        for sig,handler in saved.items():signal.signal(sig,handler)
    if guard.stop_requested:raise KeyboardInterrupt("guard received stop request")
    assert ret==0,"diagnostic guard or child failed"


def one(stage):
    b,prefix,args,env=inputs()
    run=R/"runs"/stage;run.mkdir();owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    spec={"stage":stage,"cases":[0,1],"rows":65,"condition":condition(),"observer_library":str(OBS)}
    fresh(run/"SPEC.json",spec)
    providers(stage,run/"PROVIDER_EXPECTED.json")
    if stage=="observe":env["LD_PRELOAD"]=str(OBS)
    cmd=prefix+[str(D/"driver"),"--study-output",str(run),"--study-cases",str(D/"CASES.json"),"--run-spec",str(run/"SPEC.json")]+args
    fresh(run/"server-config.json",{"command":cmd,"environment":env})
    fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"driver_sha256":b["driver_sha256"],"profile_sha256":NEW,
                              "observer_sha256":sha(OBS) if stage=="observe" else None,"reserve_GiB":64})
    write(R/"STATE.json",{"phase":stage,"status":"running","start_unix":time.time()})
    stop,threads=start_telemetry(run,D/"driver",stage=="observe")
    try:guarded_child(run)
    finally:
        stop.set()
        for thread in threads:thread.join(20)
    verify_run(run)
    print(stage,"verified",flush=True)


def restore():
    adopt.monitor()
    if any((R/"runs").glob("*/STOP_REQUESTED")):raise RuntimeError("hardware guard stopped a test; inspect before restart")
    assert sha(E/"server-config.json")==NEW
    if adopt.unit()["ActiveState"]=="inactive":
        adopt.released()
        subprocess.run([sys.executable,str(E/"start.py")],check=True,timeout=90)
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        adopt.monitor()
        assert adopt.unit()["ActiveState"] in ("active","activating"),"ordinary service stopped"
        try:
            who=adopt.identity(NEW);status,body=adopt.get("/health")
        except (AssertionError,OSError,ValueError,KeyError):
            time.sleep(4);continue
        if status==200 and body.get("status")=="ok":break
        time.sleep(4)
    else:raise TimeoutError("keep existing normal loader running")
    return who


@contextlib.contextmanager
def maintenance(label):
    path=R/("WINDOW_"+label+".json");assert not path.exists()
    assert sha(E/"server-config.json")==NEW and sha(CHECK)==adopt.CHECKER
    who,idle=adopt.idle(NEW)
    record={"initial":who,"idle":idle,"profile_sha256":NEW,"restore_required":True,"started_unix":time.time()};write(path,record)
    try:
        subprocess.run(["systemctl","stop",adopt.UNIT],check=True,timeout=110)
        adopt.released(who)
        yield
    finally:
        try:
            restored=restore()
            out=R/("RESTORED_"+label+".json")
            subprocess.run([sys.executable,"-I",str(CHECK),str(out),"--expected",str(EXPECTED)],check=True,timeout=90)
            assert read(out)["pid"]==restored["pid"] and restored["decode"]==restored["prefill"]==24
            record.update(restore_required=False,restored=restored,finished_unix=time.time());write(path,record)
        except BaseException as exc:
            record["restore_exception"]=repr(exc);write(path,record);raise


def main():
    assert __debug__ and len(sys.argv)==2
    if sys.argv[1]=="preflight":
        expected=plan();fresh(R/"DESIGN.json",expected)
        for stage in expected["stages"]:providers(stage,R/("PREFLIGHT_"+stage+".json"))
        print("24/24 topology and both provider sets verified without loading a model")
        return
    assert sys.argv[1]=="run" and os.geteuid()==0
    with (R/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(R/"DESIGN.json")==plan() and not list((R/"runs").iterdir())
        for stage in ("control","observe"):assert (R/("PREFLIGHT_"+stage+".json")).is_file()
        inputs()
        def interrupted(sig,frame):raise KeyboardInterrupt(f"signal {sig}")
        for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
        try:
            with maintenance("TIMELINE"):
                for stage in ("control","observe"):one(stage)
            write(R/"STATE.json",{"status":"both processes passed and normal 24/24 restored","finished_unix":time.time()})
        except BaseException as exc:
            write(R/"STATE.json",{"status":"incomplete","exception":repr(exc),"unix":time.time()});raise


if __name__=="__main__":main()
