"""Isolate general and MXFP4-specific GPU thresholds on fixed histories and real chat prompts."""
import os
from pathlib import Path
import subprocess
import sys
import time
from shared import HERE, PRIOR, LAB, adopt, fresh, read, sha, study, window, write
from telemetry import start as telemetry

D=HERE/"precision"
ORDER=((64,64),(128,128),(64,128),(128,64))
FLAGS=("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")


def run_one(a,b):
    name=f"precision-{a}-{b}";r=HERE/"runs"/name;r.mkdir();os.chown(r,LAB.stat().st_uid,LAB.stat().st_gid)
    build=read(D/"BUILD.json");assert sha(D/"driver")==build["driver_sha256"]
    assert all(sha(D/key)==h for key,h in build["source_sha256"].items())
    _,prefix,args,env=study.inputs()
    for key,v in zip(FLAGS,(a,b)):env[key]=str(v)
    pin=r/"PROVIDER_EXPECTED.json"
    subprocess.run([str(D/"driver"),"--check-providers",str(pin)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
    info=read(pin);base=read(PRIOR/"PREFLIGHT_control.json")
    assert info["providers"]==base["providers"] and not info["model_loaded"]
    info["active_backend_providers"]=base["active_backend_providers"];write(pin,info)
    spec={"stage":"control","observer_library":str(study.OBS),"condition":study.condition(),"thresholds":[a,b],
          "greedy_reference":a==b==64,"reference_file":str(HERE/"runs/precision-64-64/GREEDY.json"),"capture":a==b}
    fresh(r/"SPEC.json",spec)
    cmd=prefix+[str(D/"driver"),"--study-output",str(r),"--study-cases",str(D/"FIXTURES.json"),"--run-spec",str(r/"SPEC.json"),*args]
    fresh(r/"server-config.json",{"command":cmd,"environment":env})
    fresh(r/"LAUNCH.json",{"command":cmd,"environment":env,"driver":sha(D/"driver"),"fixture":sha(D/"FIXTURES.json"),"mapped_expected":build["libraries"]})
    stop,workers=telemetry(r,D/"driver",False)
    try:study.guarded_child(r)
    finally:
        stop.set()
        for t in workers:t.join(15)
    j=read(r/"RUN.json");h=read(r/"health.json")
    assert j["passed"] and h["status"]=="exited" and h["child_returncode"]==0 and h["pid"]==j["linux_entry"]["pid"]
    assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==j["after_unload_KiB"]["VmLck"]==0
    assert j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==24
    assert j["providers"]==info["providers"] and j["active_backend_providers"]==info["active_backend_providers"]
    assert [v["routed_host"] for v in j["placement"]]==[True]*37+[False]*3
    hashes={}
    for block in j["blocks"]:
        n=block["id"];f=r/(n+".f32")
        assert block["rows"]==65 and block["all_finite"] and f.stat().st_size==65*129280*4
        hashes[n]=sha(f)
        if a==b:
            prev=PRIOR/"runs"/f"qpf-{a}"/f.name
            if n.endswith("96"):assert hashes[n]==sha(prev)
            else:
                with f.open("rb") as handle:assert handle.read(129280*4)==prev.read_bytes()
    texts=read(r/"GREEDY.json");canonical=read(HERE/"runs/precision-64-64/GREEDY.json")
    fixture=read(D/"FIXTURES.json")["chats"];assert j["holdout_count"]==len(texts)==len(fixture)==8
    for q in fixture:
        n=q["id"];v=texts[n];f=r/("hold-"+n+".f32")
        assert v["prefix_ids"]==q["ids"] and v["reference_ids"]==canonical[n]["reference_ids"] and v["reference_text"]==canonical[n]["reference_text"]
        assert len(v["reference_ids"])==len(v["observed_top_ids"])==len(v["clock_ns"]) and f.stat().st_size==len(v["reference_ids"])*129280*4
        hashes["hold-"+n]=sha(f)
    fresh(r/"VERIFIED.json",{"thresholds":[a,b],"pid":h["pid"],"full_logit_sha256":hashes,"holdout_rows":sum(len(v["reference_ids"]) for v in texts.values()),
        "capture_checks":j.get("capture_comparison"),"full_prior_96_checks":a==b})
    print(name,"ordinary full distributions checked",j.get("capture_comparison"),flush=True)


if __name__=="__main__":
    assert len(sys.argv)==2
    if sys.argv[1]=="preflight":
        spec={"order":ORDER,"general_then_MXFP4":FLAGS,"driver":sha(D/"driver"),"fixtures":sha(D/"FIXTURES.json"),
              "controls":"original 96-token JA/Python+64 forced; 160+64 where both policies offload; eight independent chat histories frozen from ordinary greedy 64/64",
              "capture":"Endpoint policies only: replay identical 96 inputs; hold prefix logits against no-callback run. Collect layer-0 actual inputs, IDs, outputs, dequantized GGUF F64 dot and Q8_K-input F64 dot; all 40 layers' active expert IDs.",
              "production":"return to ordinary thresholds 64/64 and 24/24; no numerical change deployed"}
        fresh(D/"DESIGN.json",spec)
    elif sys.argv[1]=="run":
        d=read(D/"DESIGN.json");assert d["driver"]==sha(D/"driver") and d["fixtures"]==sha(D/"FIXTURES.json")
        assert read(HERE/"WORKER_STATE.json")["status"]=="sampled and restored"
        with window("PRECISION"):
            for a,b in ORDER:
                write(HERE/"PRECISION_STATE.json",{"running":[a,b]});run_one(a,b)
        write(HERE/"PRECISION_STATE.json",{"status":"complete","pid":read(HERE/"RESTORED_PRECISION.json")["pid"]})
