"""Conditional full-logit check for the one native threshold that wins its affected band."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from study import LAB,R,D,OBS,read,fresh,write,sha,inputs,condition,maintenance,guarded_child
from telemetry import start as start_telemetry

Q=R/"prefix_quality"
FLAGS=("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")


def select():
    results=read(R/"PREFILL.json")["compared_to_bracketing_64"]
    wins=[]
    for setting,n in ((32,48),(128,96)):
        metrics=[results[str(setting)][language+str(n)]["metrics"] for language in ("JA","PY")]
        if all(m[k] and m[k]["candidate_below_both_controls_in_each_order"] for m in metrics for k in ("prompt_ms","first_text_ms")):
            wins.append((max(m["prompt_ms"]["ratio"] for m in metrics),setting,n))
    return min(wins) if wins else None


def fixtures(setting,n):
    lengths=(32,40,48,56,63,64,96) if setting==32 else (64,80,96,112,127,128,160)
    original=read(D/"CASES.json")["cases"]
    cases=[]
    for i,length in enumerate(lengths):
        for ci in ((0,1) if i%2==0 else (1,0)):
            case=original[ci]
            all_ids=case["prefix_ids"]+case["next_targets_prefix_and64T1"]
            forced=64 if length==n else 0
            cases.append({"id":("JA","PY")[ci]+str(length),"prefix":all_ids[:length],
                "forced":all_ids[length:length+forced],"targets":all_ids[length:length+forced+1]})
    return {"warmup_ja64":original[0]["prefix_ids"][:64],"cases":cases,"target_length":n,
            "native_threshold_candidate":setting,"source_cases_sha256":sha(D/"CASES.json")}


def prepare(threshold):
    b=read(Q/"BUILD.json")
    for name,h in b["source_sha256"].items():assert sha(Q/name)==h
    assert sha(Q/"driver")==b["driver_sha256"]
    _,prefix,args,env=inputs()
    for k in FLAGS:env[k]=str(threshold)
    return b,prefix,args,env


def providers(out,threshold):
    b,_,_,env=prepare(threshold)
    subprocess.run([str(Q/"driver"),"--check-providers",str(out)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},timeout=30,check=True)
    j=read(out);p=read(R/"PREFLIGHT_control.json")
    assert j["providers"]==p["providers"] and j["model_loaded"] is False
    j["active_backend_providers"]=p["active_backend_providers"];write(out,j)


def run_one(threshold):
    run=R/"runs"/f"qpf-{threshold}";run.mkdir();own=LAB.stat();os.chown(run,own.st_uid,own.st_gid)
    b,prefix,args,env=prepare(threshold)
    spec={"stage":"control","prefill_threshold":threshold,"condition":condition(),"observer_library":str(OBS)}
    fresh(run/"SPEC.json",spec);providers(run/"PROVIDER_EXPECTED.json",threshold)
    cmd=prefix+[str(Q/"driver"),"--study-output",str(run),"--study-cases",str(Q/"FIXTURES.json"),"--run-spec",str(run/"SPEC.json"),*args]
    fresh(run/"server-config.json",{"command":cmd,"environment":env})
    fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"driver_sha256":b["driver_sha256"],"fixtures_sha256":sha(Q/"FIXTURES.json")})
    stop,threads=start_telemetry(run,Q/"driver",False)
    try:guarded_child(run)
    finally:
        stop.set()
        for th in threads:th.join(20)
    report=read(run/"RUN.json");fixture=read(Q/"FIXTURES.json");h=read(run/"health.json")
    assert report["passed"] and report["spec"]==spec and report["context"]["threads"]==report["context"]["threads_batch"]==report["projector_init_threads_argument"]==24
    assert report["providers"]==read(R/"PREFLIGHT_control.json")["providers"] and report["active_backend_providers"]==read(R/"PREFLIGHT_control.json")["active_backend_providers"]
    assert report["linux_entry"]["pid"]==h["pid"] and h["status"]=="exited" and h["child_returncode"]==0
    assert report["compact_bytes"]==127401984000 and report["compact_bytes_after_unload"]==report["after_unload_KiB"]["VmLck"]==0
    assert len(report["blocks"])==len(fixture["cases"])
    hashes={}
    for b,case in zip(report["blocks"],fixture["cases"]):
        assert b["id"]==case["id"] and b["prefix_ids"]==case["prefix"] and b["forced_inputs"]==case["forced"]
        assert b["rows"]==len(case["targets"]) and b["all_finite"] and (run/(case["id"]+".f32")).stat().st_size==b["rows"]*129280*4
        hashes[case["id"]]=sha(run/(case["id"]+".f32"))
    fresh(run/"VERIFIED.json",{"pid":h["pid"],"threshold":threshold,"hashes":hashes})
    print("prefix distribution check",threshold,"complete",flush=True)


def main():
    assert len(sys.argv)==2
    winner=select()
    if sys.argv[1]=="preflight":
        if winner is None:
            fresh(Q/"DECISION.json",{"run":False,"reason":"no native candidate robust at both matching input families"});return
        ratio,t,n=winner;fresh(Q/"FIXTURES.json",fixtures(t,n));providers(Q/"PREFLIGHT.json",t)
        fresh(Q/"DECISION.json",{"run":True,"candidate":t,"target_length":n,"worst_of_two_prompt_ratio":ratio,
            "scope":"Both native language fixtures beat both controls in both orders. Compare complete prefix logits at seven lengths per language and 64 additional controlled steps for the two affected native prompts."})
        return
    assert sys.argv[1]=="run" and os.geteuid()==0 and __debug__ and winner
    decision=read(Q/"DECISION.json");assert decision["run"] and decision["candidate"]==winner[1] and read(Q/"FIXTURES.json")==fixtures(winner[1],winner[2])
    with (R/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def abort(sig,frame):raise KeyboardInterrupt(f"signal {sig}")
        for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,abort)
        with maintenance("PREFIX_QUALITY"):
            for t in (64,decision["candidate"]):run_one(t)
        fresh(Q/"STATE.json",{"status":"prefix quality checked and 24/24 restored","unix":time.time()})

if __name__=="__main__":main()
