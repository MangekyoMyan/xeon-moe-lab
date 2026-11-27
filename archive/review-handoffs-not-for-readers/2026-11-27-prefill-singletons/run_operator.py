from pathlib import Path
import json
import os
import subprocess
import statistics
from build_capture import HERE, LAB, read, save, sha
import study

D=HERE/"operator";CAP=HERE/"runs/capture-JA512"
rt=read(LAB/"current/runtime.json")

def settings():
    b=read(D/"BUILD.json")
    assert sha(D/"replay")==b["binary_sha256"] and all(sha(D/n)==h for n,h in b["source_sha256"].items())
    expected={k:str(LAB/v["file"]) for k,v in rt["libraries"].items()}
    assert all(sha(p)==rt["libraries"][k]["sha256"] for k,p in expected.items())
    _,prefix,_,env=study.inputs()
    # Only the tiny scheduler shell is loaded by llama_model_load_from_file.
    # The measured real weights use the captured per-projection registry.
    env={k:v for k,v in env.items() if k not in ("GGML_CPU_IQ_PLAN","GGML_CPU_IQ3_COMPACT_MIB")}
    cupti=LAB/"strata-tc-study-20261031/cupti/cuda_cupti-linux-x86_64-12.9.79-archive/lib"
    env["LD_LIBRARY_PATH"]+=":"+str(cupti)
    return prefix,env,expected

def preflight(label):
    prefix,env,expected=settings()
    assert json.loads(subprocess.check_output([str(D/"replay"),"--providers"],env={"PATH":"/usr/bin:/bin",**env}))==expected
    cap=read(CAP/"CHECKED.json")
    assert cap["prefix_logits_match"] and all(sha(CAP/p)==h for p,h in cap["raw_sha256"].items())
    plan=read(HERE/"DESIGN.json")
    assert plan["operator_sha256"]==sha(D/"replay") and plan["capture_hashes"]==cap["raw_sha256"]
    assert plan["reps"]==7 and plan["layers"]==[4,24]

def run(label):
    prefix,env,expected=settings();plan=read(HERE/"DESIGN.json")
    if label.startswith("profile"):stages=[(True,l) for l in (4,24)]
    elif label.startswith("timing"):stages=[(False,l) for l in (4,24)]
    else:stages=[(trace,l) for trace in (False,True) for l in (4,24)]
    for trace,layer in stages:
        mode="profile" if trace else "primary"
        out=HERE/"runs"/f"{label}-{mode}-L{layer}";out.mkdir();os.chown(out,LAB.stat().st_uid,LAB.stat().st_gid)
        spec={"trace":trace,"layer":layer,"reps":plan["reps"],"providers":expected,"capture":str(CAP),
              "shell":str(LAB/"gpu-stream-20261103/tiny-moe.gguf"),"model_dir":str((LAB/rt["model"]["first_shard"]).parent)}
        save(out/"SPEC.json",spec)
        cmd=prefix+[str(D/"replay"),str(out/"SPEC.json"),str(out)]
        save(out/"server-config.json",{"command":cmd,"environment":env})
        save(out/"LAUNCH_PUBLIC.json",{"driver_sha256":sha(D/"replay"),"design_sha256":sha(HERE/"DESIGN.json"),"spec_sha256":sha(out/"SPEC.json")})
        study.guarded_child(out)
        r=read(out/"RUN.json");h=read(out/"health.json");assert r["passed"] and h["child_returncode"]==0 and h["status"]=="exited"
        assert r["layer"]==layer and r["S_P_slots"]["bit_different"]==r["S_P_sum"]["bit_different"]==0
        print(label,mode,layer,"model equivalence:",r["native_matches_model"],flush=True)

def advances(path):
    r=read(path/"RUN.json")
    if not r["passed"] or not r["native_matches_model"]:return False
    for block in (0,1):
        for press in (False,True):
            med={}
            for arm in ("G","C","S","P"):
                rows=[(s["end"]-s["start"]) for s in r["samples"] if s["phase"]=="primary" and s["arm"]==arm and s["block"]==block and s["llc_pressure"]==press]
                assert len(rows)==7
                med[arm]=statistics.median(rows)
            if min(med["S"],med["P"])>=min(med["G"],med["C"]):return False
    return True

def resume():
    prefix,env,expected=settings();plan=read(HERE/"DESIGN.json");d=HERE/"runs"
    index={"JA512":{"4":{"primary":"compare-JA512-primary-L4"},"24":{}},"PY512":{}}
    def one(case,layer,trace):
        mode="profile" if trace else "primary"
        run_name=f"final-{case}-{mode}-L{layer}";out=d/run_name;out.mkdir();os.chown(out,LAB.stat().st_uid,LAB.stat().st_gid)
        folder=d/("capture-"+case);ck=read(folder/"CHECKED.json")
        assert ck["prefix_logits_match"] and all(sha(folder/k)==h for k,h in ck["raw_sha256"].items())
        exe=D/("replay-registered-v2" if layer==4 else "replay")
        assert sha(exe)==plan["binary_by_layer"][str(layer)]
        spec={"trace":trace,"layer":layer,"reps":7,"providers":expected,"capture":str(folder),
              "shell":str(LAB/"gpu-stream-20261103/tiny-moe.gguf"),"model_dir":str((LAB/rt["model"]["first_shard"]).parent)}
        save(out/"SPEC.json",spec)
        cmd=prefix+[str(exe),str(out/"SPEC.json"),str(out)]
        save(out/"server-config.json",{"command":cmd,"environment":env})
        save(out/"LAUNCH_PUBLIC.json",{"driver_sha256":sha(exe),"design_sha256":sha(HERE/"DESIGN.json"),"spec_sha256":sha(out/"SPEC.json")})
        study.guarded_child(out)
        r=read(out/"RUN.json");h=read(out/"health.json");assert r["passed"] and h["child_returncode"]==0 and h["status"]=="exited"
        assert r["S_P_slots"]["bit_different"]==r["S_P_sum"]["bit_different"]==0
        index[case].setdefault(str(layer),{})[mode]=run_name;save(HERE/"RUNS_INDEX.json",index)
        print(run_name,"model equivalence",r["native_matches_model"],flush=True)
    one("JA512",24,False)
    eligible=[l for l in (4,24) if advances(d/index["JA512"][str(l)]["primary"])]
    for l in eligible:one("PY512",l,False)
    # Complete primary timings before independent profiling launches.
    for case,layers in (("JA512",[4,24]),("PY512",eligible)):
        for l in layers:one(case,l,True)
