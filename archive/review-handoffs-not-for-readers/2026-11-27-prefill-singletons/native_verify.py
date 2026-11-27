"""Validate the installed model's real copy decisions before interpreting a cropped graph."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from build_capture import HERE,LAB,read,save,sha
import study

def audit(label):
    d=HERE/"model-audit";b=read(d/"BUILD.json")
    assert sha(d/"driver")==b["driver_sha256"] and all(sha(d/k)==v for k,v in b["source_sha256"].items())
    out=HERE/"runs"/label;out.mkdir();os.chown(out,LAB.stat().st_uid,LAB.stat().st_gid)
    _,prefix,args,env=study.inputs();env["PP512_COPY_LEDGER"]=str(out/"native-copy.tsv")
    pin=out/"PROVIDER_EXPECTED.json"
    subprocess.run([str(d/"driver"),"--check-providers",str(pin)],env={"PATH":"/usr/bin:/bin",**env},check=True)
    v=read(pin);orig=read(LAB/"normal24-timeline-20261124/PREFLIGHT_control.json")
    assert v["providers"]==orig["providers"];v["active_backend_providers"]=orig["active_backend_providers"];save(pin,v)
    spec=out/"SPEC.json";save(spec,{"condition":study.condition(),"stage":"control","observer_library":str(study.OBS)})
    cmd=prefix+[str(d/"driver"),"--study-output",str(out),"--study-cases",str(d/"CASES.json"),"--run-spec",str(spec),*args]
    save(out/"server-config.json",{"command":cmd,"environment":env})
    save(out/"LAUNCH_PUBLIC.json",{"driver_sha256":b["driver_sha256"],"fixture_sha256":sha(d/"CASES.json"),"libraries":b["libraries"]})
    study.guarded_child(out)
    j=read(out/"RUN.json");assert j["passed"] and len(j["audit"])==8
    ledger=out/"native-copy.tsv";assert ledger.is_file()
    return reduce_ledger(out)

def reduce_ledger(out):
    routes=read(HERE/"ROUTING.json");result={};meta={x["case"]:x for x in read(out/"RUN.json")["audit"] if x["phase"]=="boundary"}
    for case in ("JA512","PY512"):
        a={}
        for layer in (4,24):
            native=next(v for v in routes[case] if v["layer"]==layer)
            active=[v["id"] for v in native["groups"]];spans=[]
            for e in active:
                if spans and e==spans[-1][1]+1:spans[-1][1]=e
                else:spans.append([e,e])
            captures=read(HERE/"runs"/("capture-"+case)/"RUN.json")["weight_meta"][str(layer)]
            rows=[]
            for lines in (out/"native-copy.tsv").read_text().splitlines():
                phase,beg,end,name,backend,handled,count,first,op,nodes,ranges=lines.split("\t")
                if phase not in (case+"-plain",case+"-boundary") or not name.startswith(f"blk.{layer}."):continue
                geo=next(z for z in captures if z["name"]==name)
                size=geo["nb"][2];intervals=[tuple(map(int,it.split("+"))) for it in ranges.split(",") if it]
                expected=[(lo*size,(hi-lo+1)*size+(512 if hi<383 else 0)) for lo,hi in spans]
                ok=(intervals==expected)
                boundary=meta[case]["layer_times"][str(layer)]
                rows.append(dict(phase=phase,name=name,handled=bool(int(handled)),first=first,first_op=op,selective_ranges_exact=ok,
                    requested_B=sum(z[1] for z in intervals),copy_calls=len(intervals),
                    all_copy_calls_inside_boundary=(boundary[0]<=int(beg)<=int(end)<=boundary[1]) if phase.endswith("boundary") else None))
            assert len(rows)==6,(case,layer,len(rows))
            a[str(layer)]={"copies":rows,"native_selective_every_projection":all(row["handled"] and row["selective_ranges_exact"] for row in rows),
                    "direct_model_boundary_ms":(meta[case]["layer_times"][str(layer)][1]-meta[case]["layer_times"][str(layer)][0])/1e6}
        result[case]=a
    save(HERE/"MODEL_COPY.json",result);return result

def corrected(case,l,trace,env,expected,prefix):
    d=HERE/"operator";out=HERE/"runs"/f"ready-{case}-{'profile' if trace else 'primary'}-L{l}"
    out.mkdir();os.chown(out,LAB.stat().st_uid,LAB.stat().st_gid)
    rt=read(LAB/"current/runtime.json");cap=HERE/"runs"/("capture-"+case)
    spec={"trace":trace,"layer":l,"reps":7,"providers":expected,"capture":str(cap),
          "shell":str(LAB/"gpu-stream-20261103/tiny-moe.gguf"),"model_dir":str((LAB/rt["model"]["first_shard"]).parent)}
    save(out/"SPEC.json",spec);cmd=prefix+[str(d/"replay"),str(out/"SPEC.json"),str(out)]
    save(out/"server-config.json",{"command":cmd,"environment":env})
    save(out/"LAUNCH_PUBLIC.json",{"operator_sha256":sha(d/"replay"),"design_sha256":sha(HERE/"READY_DESIGN.json")})
    study.guarded_child(out)
    r=read(out/"RUN.json");assert r["passed"] and r["native_matches_model"] and r["S_P_sum"]["bit_different"]==0 and r["S_P_slots"]["bit_different"]==0
    return out.name

def run():
    from importlib import util
    f=util.spec_from_file_location("local_pp_run",HERE/"run_operator.py");mod=util.module_from_spec(f);f.loader.exec_module(mod)
    prefix,env,expected=mod.settings()
    design=read(HERE/"READY_DESIGN.json")
    assert sha(HERE/"operator/replay")==design["binary_sha256"]
    ledger=audit("full-model-check")
    idx=read(HERE/"RUNS_INDEX.json")
    for trace in (False,True):
        for case in ("JA512","PY512"):
            for l in (4,24):
                if not ledger[case][str(l)]["native_selective_every_projection"]:continue
                target=corrected(case,l,trace,env,expected,prefix)
                idx[case][str(l)]["profile" if trace else "primary"]=target
                save(HERE/"READY_INDEX.json",idx)
    save(HERE/"READY_INDEX.json",idx)
