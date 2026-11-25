"""Full-model matched maintenance window; restore the verified original 24/24 API on exit."""
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare import ROOT,LAB,PREV,load,save,sha

NORMAL=LAB/"normal24-timeline-20261124"
sys.path.insert(0,str(NORMAL));import study
from telemetry import start as telemetry
study.R=ROOT
adopt=study.adopt
D=ROOT/"driver";B=ROOT/"backend"
POLICIES=("DSAR","SRDA","RADS","ADRS")
CASES=("JA96","reservation","inventory","py-slice","PY96","sql-sales","jp-mail","units")
EXTRA=("short-control","long-control")


def trials(label):
    rows=[]
    def record(name,arm,case,quality=True):
        rows.append({"name":name+"_"+arm+"_"+case,"arm":arm,"case":case,"quality":quality,"free":quality})
    if label!="candidate":
        names=(*CASES,*EXTRA)
        for n in names if label=="F0" else reversed(names):record("q","F",n)
    else:
        for i,n in enumerate(CASES):
            for a in POLICIES[i%4]:record("q",a,n)
        for i,n in enumerate(reversed(CASES)):
            for a in ("SD" if i%2==0 else "DS"):record("t",a,n,False)
        for n in EXTRA:record("q","D",n)
    return rows


def setup(candidate):
    b=load(D/"BUILD.json");assert sha(D/"driver")==b["driver"] and sha(D/"FIXTURES.json")==b["fixture"]
    assert all(sha(D/n)==h for n,h in b["sources"].items())
    original,prefix,args,env=study.inputs()
    libs={k:dict(v) for k,v in original["libraries"].items()}
    cuda=next(k for k in libs if k.startswith("libggml-cuda."))
    if candidate:
        c=load(B/"BUILD.json");lib=Path(c["library"]);assert sha(lib)==c["sha256"] and c["ordinary"]==libs[cuda]["sha256"]
        libs[cuda]={"path":str(lib),"sha256":c["sha256"]};env["LD_LIBRARY_PATH"]=str(lib.parent)+":"+env["LD_LIBRARY_PATH"]
    return prefix,args,env,libs,libs[cuda]["path"]


def providers(candidate,path):
    _,_,env,libs,_=setup(candidate)
    subprocess.run([str(D/"driver"),"--check-providers",str(path)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
    j=load(path);baseline=load(NORMAL/"PREFLIGHT_control.json")
    assert not j["model_loaded"] and j["providers"]==baseline["providers"]
    j["active_backend_providers"]=[{"name":name,"graph_compute_provider":next(v["path"] for k,v in libs.items() if k.startswith(prefix))}
                                   for name,prefix in (("CUDA0","libggml-cuda."),("CPU","libggml-cpu."))]
    save(path,j);return j


def check(run,label):
    j=load(run/"RUN.json");h=load(run/"health.json");pin=load(run/"PROVIDER_EXPECTED.json");spec=load(run/"SPEC.json")
    assert h["status"]=="exited" and h["child_returncode"]==0 and h["pid"]==j["linux_entry"]["pid"] and j["passed"]
    assert j["providers"]==pin["providers"] and j["active_backend_providers"]==pin["active_backend_providers"]
    assert j["compact_bytes"]==127401984000 and j["after_unload_KiB"]["VmLck"]==j["compact_bytes_after_unload"]==0
    assert j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==24
    assert [r["routed_host"] for r in j["placement"]]==[True]*37+[False]*3
    assert j["linux_entry"]["effective_cpu_ids"]==list(range(24)) and j["linux_entry"]["mempolicy_mode"]==3
    fs=load(D/"FIXTURES.json")["cases"]
    entries=[__import__("json").loads(line) for line in (run/"measurements.jsonl").read_text().splitlines()]
    assert j["margin_trial_count"]==len(entries)==len(spec["trials"])
    hashes={};native=PREV/"runs/precision-64-64"
    for e,t in zip(entries,spec["trials"],strict=True):
        assert all(e[k]==t[k] for k in ("name","arm","case","quality")) and e["start_ns"]<e["ready_ns"]
        c=fs[e["case"]];n=e["case"];assert e["prefix"]==len(c["prefix"])
        if not e["quality"]:
            p=run/(e["name"]+".head");cmp=run/("q_"+e["arm"]+"_"+n+".f32")
            with cmp.open("rb") as f:assert p.read_bytes()==f.read(129280*4)
            continue
        p=run/(e["name"]+".f32")
        assert e["rows"]==len(c["targets"]) and p.stat().st_size==e["rows"]*129280*4
        hashes[p.name]=sha(p)
        if e["arm"] in ("F","D"):
            base=n+".f32" if n in ("JA96","PY96") else "hold-"+n+".f32"
            assert hashes[p.name]==sha(native/base),("no-op changed model",label,n)
        if "reference_ids" in c:assert "free_text" in e
    study.fresh(run/"VERIFIED.json",{"pid":h["pid"],"label":label,"hashes":hashes,"checked_ordinary_or_decision_only_against_prior":True})
    print(label,len(entries),"full-model executions verified",flush=True)


def one(label):
    candidate=label=="candidate"
    prefix,args,env,libs,cuda=setup(candidate);run=ROOT/"runs"/label;run.mkdir();os.chown(run,LAB.stat().st_uid,LAB.stat().st_gid)
    pin=providers(candidate,run/"PROVIDER_EXPECTED.json");assert pin==load(D/("PREFLIGHT_"+str(candidate)+".json"))
    spec={"stage":"control","observer_library":str(study.OBS),"condition":study.condition(),"candidate":candidate,"cuda_path":cuda,
          "cut":load(ROOT/"DESIGN.json")["threshold"],"trials":trials(label)}
    study.fresh(run/"SPEC.json",spec)
    cmd=prefix+[str(D/"driver"),"--study-output",str(run),"--study-cases",str(D/"FIXTURES.json"),"--run-spec",str(run/"SPEC.json"),*args]
    study.fresh(run/"server-config.json",{"command":cmd,"environment":env})
    study.fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"driver_sha256":sha(D/"driver"),"fixtures":sha(D/"FIXTURES.json"),"libraries":libs})
    stop,workers=telemetry(run,D/"driver",False)
    try:study.guarded_child(run)
    finally:
        stop.set()
        for w in workers:w.join(15)
    check(run,label)


def main():
    if sys.argv[1]=="preflight":
        for candidate in (False,True):providers(candidate,D/("PREFLIGHT_"+str(candidate)+".json"))
        study.fresh(ROOT/"MODEL_DESIGN.json",{"order":["F0","candidate","F1"],"fixture":sha(D/"FIXTURES.json"),"policy_design":sha(ROOT/"DESIGN.json"),
            "budgets":"D builds the selection plan and moves the flag list on existing synchronization. S applies it to true near positions, R rotates it by 37, A applies to all. All compute sixth/seventh at the 40 routers in eligible prefixes.",
            "trials":{k:trials(k) for k in ("F0","candidate","F1")},"normal_target":adopt.NEW})
        return
    assert sys.argv[1]=="run" and os.geteuid()==0 and __debug__
    plan=load(ROOT/"MODEL_DESIGN.json")
    assert plan["fixture"]==sha(D/"FIXTURES.json") and plan["policy_design"]==sha(ROOT/"DESIGN.json")
    with (NORMAL/".study.lock").open("r") as mainlock,(ROOT/".study.lock").open("a") as own:
        for lock in (mainlock,own):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def abort(sig,frame):raise KeyboardInterrupt(f"signal {sig}")
        for sig in (signal.SIGHUP,signal.SIGTERM):signal.signal(sig,abort)
        try:
            with study.maintenance("MARGIN"):
                for label in plan["order"]:
                    save(ROOT/"STATE.json",{"running":label,"time":time.time()});one(label)
        except BaseException as e:
            save(ROOT/"STATE.json",{"phase":"interrupted","error":repr(e),"time":time.time()});raise
        save(ROOT/"STATE.json",{"status":"measurements complete","normal_pid":load(ROOT/"RESTORED_MARGIN.json")["pid"]})

if __name__=="__main__":main()
