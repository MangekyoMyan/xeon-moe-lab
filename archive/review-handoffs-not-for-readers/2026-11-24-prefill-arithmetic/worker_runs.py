"""Guarded real-model CPU worker study on the ordinary 24/24 profile."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from shared import HERE, LAB, PRIOR, adopt, fresh, read, sha, study, window, write
from telemetry import start as telemetry

D = HERE / "driver"
BASE = "original"


def prepare(kind):
    saved, prefix, args, env = study.inputs()
    new = read(D / "BUILD.json")
    assert sha(D / "driver") == new["driver_sha256"]
    assert all(sha(D/name)==h for name,h in new["source_sha256"].items())
    libs = {k:dict(v) for k,v in saved["libraries"].items()}
    cpu = next(k for k in libs if k.startswith("libggml-cpu."))
    if kind!="trace":
        env["OMP_DISPLAY_ENV"]="VERBOSE"
    if kind=="spin3000":
        env["GOMP_SPINCOUNT"]="3000"
    elif kind!=BASE:
        rec=read(HERE/"cpu"/kind/"BUILD.json")
        path=Path(rec["library"])
        assert rec["kind"]==kind and sha(path)==rec["sha256"] and rec["original_library"]==libs[cpu]["sha256"]
        libs[cpu]={"path":str(path),"sha256":rec["sha256"]}
        env["LD_LIBRARY_PATH"]=str(path.parent)+":"+env["LD_LIBRARY_PATH"]
    return prefix,args,env,libs,libs[cpu]["path"]


def pin(kind,out):
    _,_,env,libs,_=prepare(kind)
    p = subprocess.run([str(D/"driver"),"--check-providers",str(out)],env={"PATH":"/usr/bin:/bin","LANG":"C",**env},
                       capture_output=True,text=True,timeout=30)
    if p.returncode:
        raise RuntimeError(f"provider preflight failed ({p.returncode}): {p.stderr[-600:]}")
    data=read(out)
    assert not data["model_loaded"] and len(data["providers"])==5
    for e in data["providers"]:
        ident=("libllama." if e["symbol"].startswith("llama_") else "libggml-base." if e["symbol"].startswith("ggml_backend_sched_") else "libggml-cpu.")
        v=next(v for k,v in libs.items() if k.startswith(ident))
        p=Path(v["path"]);st=p.stat()
        assert sha(p)==v["sha256"] and (e["path"],e["device"],e["inode"],e["bytes"])==(str(p),st.st_dev,st.st_ino,st.st_size)
    data["active_backend_providers"]=[{"name":name,"graph_compute_provider":next(v["path"] for k,v in libs.items() if k.startswith(ident))}
             for name,ident in (("CUDA0","libggml-cuda."),("CPU","libggml-cpu."))]
    write(out,data)
    return data


def verify(run,kind,order):
    j=read(run/"RUN.json");h=read(run/"health.json");p=read(run/"PROVIDER_EXPECTED.json")
    assert j["passed"] and j["linux_entry"]["pid"]==h["pid"] and h["status"]=="exited" and h["child_returncode"]==0
    assert j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==24
    assert j["linux_entry"]["effective_cpu_ids"]==list(range(24)) and j["linux_entry"]["mempolicy_mode"]==j["linux_entry"]["mempolicy_nodes_mask"]==3 and j["linux_entry"]["thp_disabled"]
    assert j["providers"]==p["providers"] and j["active_backend_providers"]==p["active_backend_providers"]
    assert [v["routed_host"] for v in j["placement"]]==[True]*37+[False]*3
    assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==j["after_unload_KiB"]["VmLck"]==0
    assert {k:v for k,v in j["registration"]["experts"].items() if v}==read(LAB/"preupgrade-baseline-20261123/PLAN_MEMBERSHIP.json")
    ref=read(PRIOR/"runs/control/VERIFIED.json")["output_sha256"]; outputs={}
    for idx,b in enumerate(j["blocks"]):
        ci=order[idx];c=read(D/"CASES.json")["cases"][ci]
        assert b["case_index"]==ci and b["case_id"]==c["case_id"] and b["input_tokens"]==c["used_T1_ids"] and b["prefix_ids"]==c["prefix_ids"]
        assert b["status"]==[0]*64 and b["rows"]==65 and b["all_finite"] and b["traced_calls"]==0
        assert (run/b["file"]).stat().st_size==65*129280*4
        outputs[b["file"]]=sha(run/b["file"])
    files=sorted(run.glob("worker-*.tsv"))
    assert len(files)==(16 if kind=="trace" else 0)
    if kind=="trace":
        for ci in (0,1):
            assert set(j["worker_graphs"][str(ci)])=={str(i) for i in range(5,65,8)}
            assert all(1<=n<=100 for n in j["worker_graphs"][str(ci)].values())
    proof={"pid":h["pid"],"kind":kind,"expected":ref,"outputs":outputs,"all_full_rows_match":outputs==ref,"worker_files":len(files)}
    if kind!="trace":
        display=(run/"server.stderr.log").read_text(errors="replace")
        value=re.search(r"GOMP_SPINCOUNT\s*=\s*'?(\d+)",display)
        assert value,"libgomp runtime settings missing"
        proof["GOMP_SPINCOUNT_from_runtime"]=int(value[1])
        if kind=="spin3000":assert int(value[1])==3000
    fresh(run/"VERIFIED.json",proof)
    if kind in (BASE,"trace"):
        assert outputs==ref,"baseline/observation changed output"
    return proof


def one(label,kind,order):
    pre=read(HERE/"cpu"/("PREFLIGHT-"+kind+".json"))
    prefix,args,env,libs,cpu=prepare(kind)
    run=HERE/"runs"/label
    run.mkdir();os.chown(run,LAB.stat().st_uid,LAB.stat().st_gid)
    assert pin(kind,run/"PROVIDER_EXPECTED.json")==pre
    spec={"stage":"control","condition":study.condition(),"cases":order,"worker_trace":kind=="trace",
          "cpu_path":cpu,"observer_library":str(study.OBS),"kind":kind}
    fresh(run/"SPEC.json",spec)
    cmd=prefix+[str(D/"driver"),"--study-output",str(run),"--study-cases",str(D/"CASES.json"),"--run-spec",str(run/"SPEC.json"),*args]
    fresh(run/"server-config.json",{"command":cmd,"environment":env})
    fresh(run/"LAUNCH.json",{"command":cmd,"environment":env,"driver_sha256":sha(D/"driver"),"libraries":libs,"normal_config":adopt.NEW})
    stop,threads=telemetry(run,D/"driver",False)
    energy=None
    if kind!="trace":
        import rapl
        energy=rapl.start(run)
    try:study.guarded_child(run)
    finally:
        stop.set()
        for th in threads:th.join(20)
        if energy:
            done,thread,errors=energy
            done.set()
            thread.join(10)
            assert not errors and not thread.is_alive()
    proof=verify(run,kind,order)
    print(label,proof["all_full_rows_match"],"worker files",proof["worker_files"],flush=True)
    return proof


def main():
    action=sys.argv[1]
    if action=="preflight":
        for kind in sys.argv[2:]:
            dest=HERE/"cpu"/("PREFLIGHT-"+kind+".json")
            with tempfile.TemporaryDirectory() as tmp:
                receipt=pin(kind,Path(tmp)/"providers.json")
            if dest.exists():assert read(dest)==receipt
            else:fresh(dest,receipt)
    elif action=="trace":
        until=time.monotonic()+3600
        while time.monotonic()<until:
            if (HERE/"RESTORED_CHAT.json").is_file() and read(HERE/"CHAT_STATE.json").get("status")=="complete":break
            time.sleep(4)
        else:raise RuntimeError("chat window did not close")
        with window("WORKERS"):
            one("worker-trace","trace",[0,1])
        fresh(HERE/"WORKER_STATE.json",{"status":"sampled and restored","pid":read(HERE/"RESTORED_WORKERS.json")["pid"]})
    elif action=="compare":
        plan=read(HERE/"CPU_COMPARISON.json")
        assert read(HERE/"WORKER_STATE.json")["status"]=="sampled and restored"
        with window("CPU_COMPARE"):
            for label,kind,order in plan["periods"]:
                write(HERE/"CPU_STATE.json",{"phase":label,"kind":kind})
                proof=one(label,kind,order)
                if not proof["all_full_rows_match"]:break
        write(HERE/"CPU_STATE.json",{"status":"restored","last_period":label,"last_exact":proof["all_full_rows_match"],
             "pid":read(HERE/"RESTORED_CPU_COMPARE.json")["pid"]})
    else:raise SystemExit(action)


if __name__=="__main__":
    main()
