"""Replay first-layer-only numerical changes in a guarded model at thresholds 128/128."""
import os
import subprocess
import sys
import time
from shared import HERE,LAB,PRIOR,study,adopt,read,write,fresh,sha,window
from telemetry import start as telemetry

D=HERE/"cause_replay"
MODES=("control","gpu-gates","residual","gpu-layer-output")


def main():
    deadline=time.monotonic()+4000
    while time.monotonic()<deadline:
        state=HERE/"PRECISION_STATE.json"
        if state.exists() and read(state).get("status")=="complete":break
        time.sleep(5)
    else:raise RuntimeError("factorial controller has not restored the ordinary service")
    b=read(D/"BUILD.json");assert sha(D/"driver")==b["driver"]
    for name,h in b["sources"].items():assert sha(D/name)==h
    before=read(HERE/"PRECISION_CAUSE.json")
    assert all(q["validated_callbacks"]=={"early":True,"routes":True} for q in before.values())
    saved={}
    for case in before:
        for policy in (64,128):
            leaf=HERE/"runs"/f"precision-{policy}-{policy}"/(case+"-early")
            for n in ("ffn_moe_gate-0.bin","ffn_moe_up-0.bin","l_out-0.bin","ffn_moe_gate-0-input.bin","ffn_moe_up-0-input.bin","ffn_moe_gate-0-ids.bin","ffn_moe_up-0-ids.bin","CAPTURE.json"):
                saved[str((leaf/n).relative_to(HERE))]=sha(leaf/n)
    fresh(D/"DESIGN.json",{"purpose":"One 128/128 diagnostic process, fixed 96+64 histories.",
        "arms":{"control":"same callback, exact CPU outputs, no write","gpu-gates":"copy only the recorded ordinary GPU layer-0 gate/up outputs from identical operands",
                "residual":"add a second ordinary IQ2_S CPU dot on the Q8_K quantization residual of the input, only layer-0 gate/up.",
                "gpu-layer-output":"copy only the entire recorded layer-0 GPU-policy final activation"},
        "expected_inputs":saved,"driver":b["driver"]})
    with window("CAUSAL"):
        r=HERE/"runs/causal";r.mkdir();os.chown(r,LAB.stat().st_uid,LAB.stat().st_gid)
        _,prefix,args,env=study.inputs()
        for key in ("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"):env[key]="128"
        spec={"stage":"control","observer_library":str(study.OBS),"condition":study.condition(),"endpoint_runs":str(HERE/"runs")}
        fresh(r/"SPEC.json",spec);p=r/"PROVIDER_EXPECTED.json"
        subprocess.run([str(D/"driver"),"--check-providers",str(p)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True,timeout=30)
        pin=read(p);orig=read(PRIOR/"PREFLIGHT_control.json")
        assert pin["providers"]==orig["providers"];pin["active_backend_providers"]=orig["active_backend_providers"];write(p,pin)
        cmd=prefix+[str(D/"driver"),"--study-output",str(r),"--study-cases",str(HERE/"precision/FIXTURES.json"),"--run-spec",str(r/"SPEC.json"),*args]
        fresh(r/"server-config.json",{"command":cmd,"environment":env})
        fresh(r/"LAUNCH.json",{"command":cmd,"environment":env,"driver":b["driver"],"fixture":sha(HERE/"precision/FIXTURES.json")})
        stop,threads=telemetry(r,D/"driver",False)
        try:study.guarded_child(r)
        finally:
            stop.set()
            for th in threads:th.join(20)
        j=read(r/"RUN.json");h=read(r/"health.json")
        assert j["passed"] and j["linux_entry"]["pid"]==h["pid"] and h["status"]=="exited" and h["child_returncode"]==0
        assert j["providers"]==pin["providers"] and j["active_backend_providers"]==pin["active_backend_providers"]
        assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==j["after_unload_KiB"]["VmLck"]==0
        assert j["context"]["threads"]==j["context"]["threads_batch"]==24
        hashes={}
        for case in ("JA96","PY96"):
            assert sha(r/(case+"-control.f32"))==sha(HERE/"runs/precision-128-128"/(case+".f32"))
            for mode in MODES:
                f=r/(case+"-"+mode+".f32");q=j["counterfactual"][case][mode]
                assert f.stat().st_size==65*129280*4 and q["rows"]==65
                assert len(q["visits"])==(1 if mode=="gpu-layer-output" else 2)
                hashes[f.name]=sha(f)
        fresh(r/"VERIFIED.json",{"hashes":hashes,"control_equals_normal_128_all_rows":True,"pid":h["pid"]})
    fresh(HERE/"CAUSAL_STATE.json",{"status":"complete","normal_pid":read(HERE/"RESTORED_CAUSAL.json")["pid"]})


if __name__=="__main__":main()
