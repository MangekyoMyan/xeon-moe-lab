"""One guarded maintenance regression on the installed E5-2630L v2 pair."""
import argparse
import array
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from api_window import active, clear_monitor, window
from live import ENGINE, LAB, digest, read

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT / "trial"
PREVIOUS = LAB / "169a-exploration-20261119"
GOLD = PREVIOUS / "main/runs/baseline"
CPU = "libggml-cpu.so.0.28.0"
CUDA = "libggml-cuda.so.0.28.0"
BASE = "libggml-base.so.0.28.0"
LLAMA = "libllama.so.0.8.0"

def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    temporary.replace(path)

def prepare(stage):
    initial = read(ROOT/"INITIAL_RUNTIME.json")
    config = read(ENGINE/"server-config.json")
    assert digest(ENGINE/"server-config.json")==initial["config_sha256"]
    previous = read(GOLD/"LAUNCH.json")
    for name, measurement in previous["guard_source_sha256"].items():
        assert digest(name)==measurement, name
    recorded_command = previous["command"]
    target_args = recorded_command[recorded_command.index("-m"):]
    assert config["model"]==target_args[1]
    model_dir = Path(config["model"]).parent.parent
    assert digest(model_dir/"download-verified.json")==previous["model_manifest_sha256"]
    for f in read(model_dir/"download-verified.json")["files"]:
        assert (model_dir/f["name"]).stat().st_size==f["size"]
    plan = Path(config["environment"]["GGML_CPU_IQ_PLAN"])
    assert digest(plan)==initial["plan_sha256"]
    build = read(ROOT/"build/BUILD.json")
    variant = build[stage["arm"]]
    driver = read(TRIAL/"BUILD.json")
    assert digest(TRIAL/"driver")==driver["driver_sha256"]
    for name,h in driver["source_sha256"].items():
        assert digest(TRIAL/name)==h
    assert read(ROOT/"compact-check/RESULT.json")["outputs_byte_equal"]
    libraries = {Path(p).name:{"path":p,"sha256":h} for p,h in initial["mapped_runtime_hashes"].items()}
    libraries[CPU]={"path":variant["file"],"sha256":variant["sha256"]}
    for v in libraries.values():
        assert digest(v["path"])==v["sha256"]
    env = config["environment"].copy()
    cpu_dir = Path(variant["file"]).parent
    env["LD_LIBRARY_PATH"]=str(cpu_dir)+":"+env["LD_LIBRARY_PATH"]
    env["LD_BIND_NOW"]="1"
    assert all(k not in env for k in ("LD_PRELOAD","GGML_CPU_IQ_ROW_LOCAL","GGML_CPU_IQ_AUDIT_DIR"))
    launch={"prefix":recorded_command[:recorded_command.index(str(PREVIOUS/"main/driver"))],
            "args":target_args,"environment":env,"libraries":libraries,"initial_config_sha256":initial["config_sha256"],
            "plan_sha256":initial["plan_sha256"],"download_manifest_sha256":previous["model_manifest_sha256"],
            "driver_sha256":driver["driver_sha256"],"cpu_build_sha256":digest(ROOT/"build/BUILD.json"),
            "case_file_sha256":digest(TRIAL/"CASES.json"),"design_sha256":digest(TRIAL/"DESIGN.json"),
            "guard_source_sha256":previous["guard_source_sha256"],"runner_sha256":digest(__file__)}
    return launch

def providers(launch, out):
    clean={k:v for k,v in os.environ.items() if not k.startswith(("GGML_","LLAMA_","LD_","OMP_","GOMP_","CUDA_"))}
    clean.update(launch["environment"])
    subprocess.run([str(TRIAL/"driver"),"--check-providers",str(out)],env=clean,check=True,timeout=30)
    actual=read(out)
    assert actual["model_loaded"] is False and len(actual["providers"])==5
    for symbol in actual["providers"]:
        name=symbol["symbol"]
        lib=LLAMA if name.startswith("llama_") else BASE if name.startswith("ggml_backend_sched_") else CPU
        path=Path(launch["libraries"][lib]["path"])
        st=path.stat()
        assert symbol["path"]==str(path) and (symbol["device"],symbol["inode"],symbol["bytes"])==(st.st_dev,st.st_ino,st.st_size)
    actual["active_backend_providers"]=[{"name":name,"graph_compute_provider":launch["libraries"][lib]["path"]} for name,lib in [("CUDA0",CUDA),("CPU",CPU)]]
    save(out,actual)

def expected_ids():
    plan=Path(read(ENGINE/"server-config.json")["environment"]["GGML_CPU_IQ_PLAN"])
    lines=plan.read_text().splitlines()
    result={}
    for line in lines[1:]:
        name,_type,_k,_m,e,count,*rest=line.split()
        ids=sorted(map(int,rest))
        assert int(e)==384 and len(ids)==len(set(ids))==int(count)
        result[name]=ids
    assert len(result)==80 and sum(map(len,result.values()))==19200
    return result

def verify(run):
    j=read(run/"RUN.json")
    s=read(run/"SPEC.json")
    pin=read(run/"PROVIDER_EXPECTED.json")
    info=read(run/"health.json")
    assert info["status"]=="exited" and info["child_returncode"]==0 and j["passed"]
    assert j["spec"]==s and j["providers"]==pin["providers"] and j["active_backend_providers"]==pin["active_backend_providers"]
    assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==0 and j["vision_loaded"]
    assert [p["routed_host"] for p in j["placement"]]==[True]*37+[False]*3 and len(j["shared_inventory"])==37
    assert min(j[k] for k in ["text_model_context_load_ns","unload_ns","vision_init_ns"])>0
    quality=s["stage"]=="quality"
    if quality:
        reg=j["registration"]
        expected=expected_ids()
        assert len(reg["experts"])==120 and {n:ids for n,ids in reg["experts"].items() if ids}==expected
        assert reg["registered_tensors"]==80 and reg["selected_projections"]==19200 and reg["sampled_pages"]==1440
        assert reg["sampled_vma_policies"]=={"interleave:0-1":1440}
        if reg["page_location_errno"]==0:
            assert set(reg["sampled_page_nodes"])=={"0","1"} and sum(reg["sampled_page_nodes"].values())==1440
    else:
        assert "registration" not in j
    cases=read(TRIAL/"CASES.json")["cases"]
    assert len(j["blocks"])==len(s["cases"])==2
    for b,i in zip(j["blocks"],s["cases"],strict=True):
        c=cases[i]
        assert b["case_id"]==c["case_id"] and b["prefix_ids"]==c["prefix_ids"] and b["input_tokens"]==c["used_T1_ids"]
        assert b["positions_begin"]==c["prefix_count"] and b["prefill_ready_ns"]>b["prefill_begin_ns"]
        assert len(b["start_ns"])==len(b["output_ready_ns"])==64 and b["status"]==[0]*64 and b["traced_calls"]==0
        assert all(x<y for x,y in zip(b["start_ns"],b["output_ready_ns"])) and all(x>y for x,y in zip(b["start_ns"][1:],b["output_ready_ns"][:-1]))
        assert b["file"]==c["case_id"]+".f32" and b["rows"]==(65 if quality else 1)
        raw=(run/b["file"]).read_bytes()
        assert len(raw)==b["rows"]*129280*4
        gold=(GOLD/b["file"]).read_bytes()
        assert hashlib.sha256(gold).hexdigest()==read(PREVIOUS/"BASELINE_REVERIFIED.json")["outputs"][b["file"]]
        assert raw==(gold if quality else gold[-len(raw):]), "fixed logits differ"
        values=array.array("f")
        values.frombytes(raw)
        assert all(map(math.isfinite,values))

def run_stage(stage):
    clear_monitor(ROOT)
    assert not any(active(n) for n in ["llm-openai-api.service","llm-deepseek41-api.service"])
    memory={l.split(":")[0]:int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith(("MemTotal:","MemAvailable:","Mlocked:"))}
    assert memory["MemTotal"]>=480*1024**2 and memory["MemAvailable"]>=360*1024**2 and memory["Mlocked"]<=32*1024**2
    launch=prepare(stage)
    run=TRIAL/"runs"/stage["name"]
    run.mkdir(parents=True)
    owner=LAB.stat()
    os.chown(run,owner.st_uid,owner.st_gid)
    providers(launch,run/"PROVIDER_EXPECTED.json")
    save(run/"SPEC.json",stage)
    command=launch["prefix"]+[str(TRIAL/"driver"),"--study-output",str(run),"--study-cases",str(TRIAL/"CASES.json"),"--run-spec",str(run/"SPEC.json")]+launch["args"]
    launch.update(command=command,spec=stage,reserve_GiB=64)
    save(run/"server-config.json",{"command":command,"environment":launch["environment"]})
    save(run/"LAUNCH.json",launch)
    state=read(TRIAL/"STATE.json")
    state["runs"][stage["name"]]={"path":str(run),"status":"running"}
    save(TRIAL/"STATE.json",state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    sys.path.insert(0,str(ENGINE))
    spec=importlib.util.spec_from_file_location("cache_trial_guard",ENGINE/"supervise.py")
    guard=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.ROOT=run
    host_check=guard.check_host
    deadline=time.monotonic()+7200
    def check(initial):
        cpus,available=host_check(initial)
        if available<64*1024**2 or time.monotonic()>deadline:
            raise RuntimeError("RAM reserve/time guard")
        return cpus,available
    guard.check_host=check
    print("Starting",stage["name"],flush=True)
    rc=guard.main()
    state["runs"][stage["name"]].update(status="child-exited",returncode=rc)
    save(TRIAL/"STATE.json",state)
    if rc:
        raise RuntimeError("Model process failed: "+stage["name"])
    verify(run)
    save(run/"RESULTS_SHA256.json",{p.name:digest(p) for p in run.iterdir() if p.is_file() and p.name!="RESULTS_SHA256.json"})
    state["runs"][stage["name"]].update(status="passed",manifest_sha256=digest(run/"RESULTS_SHA256.json"))
    save(TRIAL/"STATE.json",state)
    print("Passed",stage["name"],flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    opts=parser.parse_args()
    design=read(TRIAL/"DESIGN.json")
    if opts.preflight:
        for arm in ["control","ordinary"]:
            stage=next(s for s in design["stages"] if s["arm"]==arm)
            providers(prepare(stage),TRIAL/("PREFLIGHT-"+arm+".json"))
        print("Both CPU providers verified without model load.")
        return
    assert os.geteuid()==0
    with (ROOT/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(TRIAL/"STATE.json")["completed"]:
            return
        for stage in design["stages"]: prepare(stage)
        with window(ROOT,"ROW_CLEANUP"):
            for stage in design["stages"]:
                old=read(TRIAL/"STATE.json")["runs"].get(stage["name"])
                if old:
                    assert old["status"]=="passed"
                    verify(Path(old["path"]))
                else:
                    if stage["stage"]=="timing":
                        assert read(TRIAL/"QUALITY_GATE.json")["passed"]
                    run_stage(stage)
                if stage["name"]=="quality-ordinary":
                    for arm in ["control","ordinary"]:
                        verify(TRIAL/"runs"/("quality-"+arm))
                    save(TRIAL/"QUALITY_GATE.json",{"passed":True,"all_65_rows_and_membership_verified":True})
        state=read(TRIAL/"STATE.json")
        state.update(completed=True,phase="Six processes complete; service window closed")
        save(TRIAL/"STATE.json",state)

if __name__=="__main__":main()
