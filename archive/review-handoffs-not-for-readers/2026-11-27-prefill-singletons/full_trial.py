from build_capture import HERE,LAB,read,save,sha
import os
import subprocess
import study

def validate_start():
    d=HERE/"integrated";b=read(d/"BUILD.json")
    assert sha(d/"driver")==b["binary_sha256"] and sha(HERE/"operator/gather.cu")==b["gather_source_sha256"]
    assert all(sha(d/k)==v for k,v in b["source_sha256"].items())
    assert all(x["native_selective_every_projection"] for ca in read(HERE/"MODEL_COPY.json").values() for x in ca.values())
    for case in ("JA512","PY512"):
        result=read(HERE/("MATCHED_"+case+".json"))
        assert all(v["passes_comparison_of_block_medians"] for v in result.values()),"local result does not advance"
        assert all(not v["CUPTI"]["G"]["range_sizes_missing_from_observed_H2D"] for v in result.values()),"unverified native weight transfers"

def run(label):
    large=label.startswith("live-large-")
    d=HERE/("large" if large else "integrated");b=read(d/"BUILD.json");out=HERE/"runs"/label;out.mkdir();os.chown(out,LAB.stat().st_uid,LAB.stat().st_gid)
    assert sha(d/"driver")==b["binary_sha256"] and all(sha(d/k)==v for k,v in b["source_sha256"].items())
    if large:assert sha(d/"CASES.json")==read(HERE/"LARGE_DESIGN.json")["case_sha256"]
    _,prefix,args,env=study.inputs()
    pin=out/"PROVIDER_EXPECTED.json";subprocess.run([str(d/"driver"),"--check-providers",str(pin)],env={"PATH":"/usr/bin:/bin",**env},check=True)
    p=read(pin);orig=read(LAB/"normal24-timeline-20261124/PREFLIGHT_control.json");assert p["providers"]==orig["providers"]
    p["active_backend_providers"]=orig["active_backend_providers"];save(pin,p)
    spec=out/"SPEC.json";save(spec,{"condition":study.condition(),"stage":"control","observer_library":str(study.OBS)})
    cmd=prefix+[str(d/"driver"),"--study-output",str(out),"--study-cases",str(d/"CASES.json"),"--run-spec",str(spec),*args]
    save(out/"server-config.json",{"command":cmd,"environment":env})
    save(out/"LAUNCH_PUBLIC.json",{"driver_sha256":b["binary_sha256"],"libraries":b["libraries"],"design_sha256":sha(HERE/("LARGE_DESIGN.json" if large else "INTEGRATED_DESIGN.json"))})
    study.guarded_child(out);j=read(out/"RUN.json");h=read(out/"health.json")
    assert h["child_returncode"]==0 and j["passed"]
    if large:assert len(j["quality"])==8 and len(j["timings"])==8 and len(j["conversation"])==4
    else:assert len(j["quality"])==12 and len(j["timings"])==16 and len(j["free_greedy"])==4
