"""Guarded maintenance window for either a full-prefix capture or standalone FFN replays."""
import argparse
import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
from build_capture import HERE, LAB, read, save, sha

pr=LAB/"normal24-timeline-20261124"
sys.path.insert(0,str(pr))
import study
study.R=HERE

def capture(name,fixture):
    d=HERE/"capture";build=read(d/"BUILD.json")
    assert sha(d/"driver")==build["driver_sha256"] and all(sha(d/k)==v for k,v in build["source_sha256"].items())
    if name=="capture-JA512":assert sha(fixture)==build["fixture_sha256"]
    _,prefix,args,env=study.inputs()
    folder=HERE/"runs"/name;folder.mkdir(exist_ok=False)
    os.chown(folder,LAB.stat().st_uid,LAB.stat().st_gid)
    pin=folder/"PROVIDER_EXPECTED.json"
    subprocess.run([str(d/"driver"),"--check-providers",str(pin)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},check=True)
    actual=read(pin);normal=read(pr/"PREFLIGHT_control.json");assert actual["providers"]==normal["providers"]
    actual["active_backend_providers"]=normal["active_backend_providers"];save(pin,actual)
    spec=folder/"SPEC.json";save(spec,{"condition":study.condition(),"stage":"control","observer_library":str(study.OBS)})
    cmd=prefix+[str(d/"driver"),"--study-output",str(folder),"--study-cases",str(fixture),"--run-spec",str(spec),*args]
    save(folder/"server-config.json",{"command":cmd,"environment":env})
    save(folder/"LAUNCH_PUBLIC.json",{"binary":build["driver_sha256"],"fixture":sha(fixture),"libraries":build["libraries"],"profile":study.NEW})
    study.guarded_child(folder)
    j=read(folder/"RUN.json");h=read(folder/"health.json")
    assert j["passed"] and h["child_returncode"]==0 and h["pid"]==j["linux_entry"]["pid"] and j["capture"]["native_logits_equal"]
    assert sha(folder/"capture-logits.f32")==sha(folder/"native-logits.f32") and j["compact_bytes"]==127401984000
    save(folder/"CHECKED.json",{"weights":j["weight_meta"],"tensors":j["capture"]["tensors"],
         "raw_sha256":{p.name:sha(p) for p in folder.iterdir() if p.suffix in (".bin",".f32")},"prefix_logits_match":True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument("label");parser.add_argument("--fixture",type=type(HERE),default=HERE/"capture/FIXTURE.json")
    a=parser.parse_args();assert os.geteuid()==0 and a.label.replace("-","").replace("_","").isalnum()
    assert sha(study.E/"server-config.json")==study.NEW
    before=HERE/"BEFORE.json"
    if not before.exists():
        subprocess.run([sys.executable,"-I",str(study.CHECK),str(before),"--expected",str(study.EXPECTED)],check=True)
    def stop(sig,frame):raise KeyboardInterrupt(f"controller signal {sig}")
    for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    with (pr/".study.lock").open() as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.label.startswith(("live-two-","live-large-")):
            import full_trial
            full_trial.validate_start()
            with study.maintenance(a.label):full_trial.run(a.label)
        elif a.label=="check-native":
            import native_verify
            assert (HERE/"READY_DESIGN.json").is_file()
            with study.maintenance(a.label):native_verify.run()
        elif a.label.startswith("capture-"):
            study.inputs()
            with study.maintenance(a.label):
                capture(a.label,a.fixture)
        else:
            loader=importlib.util.spec_from_file_location("pp512_operator",HERE/"run_operator.py")
            run_operator=importlib.util.module_from_spec(loader);loader.loader.exec_module(run_operator)
            run_operator.preflight(a.label)
            with study.maintenance(a.label):
                if a.label=="resume512":
                    capture("capture-PY512",HERE/"capture/PY512.json")
                    run_operator.resume()
                else:run_operator.run(a.label)
    print("Completed",a.label,"and restored normal API",flush=True)

if __name__=="__main__":main()
