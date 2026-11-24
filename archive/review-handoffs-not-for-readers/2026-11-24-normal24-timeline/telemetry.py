"""Read the host IMCs; collect CPU PCs on calls 5–64 after each CUPTI window."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def write(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_bytes())


def memory_traffic(run,stop):
    source=Path(__file__).resolve().parent.parent/"xeon_imc.py"
    spec=importlib.util.spec_from_file_location("study_imc",source)
    imc=importlib.util.module_from_spec(spec);spec.loader.exec_module(imc)
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    reader=imc.ImcReader()
    result={"events":reader.events,"open_error":reader.error,"scope":"system-wide channels, raw count/enabled/running/monotonic_s"}
    seen_guard=set();seen_watch=set()
    try:
        with (run/"host-counters.jsonl").open("x",buffering=1) as f,(run/"guard-samples.jsonl").open("x",buffering=1) as g:
            next_guard=0
            while not stop.is_set():
                t=time.monotonic()
                if reader.fds:
                    ticks=[imc.read_counter(fd) for fd in reader.fds]
                    f.write(json.dumps({"t":t,"raw":ticks},separators=(",",":"))+"\n")
                if t>=next_guard:
                    next_guard=t+5
                    if (run/"health.json").exists():
                        health=read(run/"health.json")
                        for s in health.get("recent_samples",[]):
                            if s["unix"] in seen_guard:continue
                            seen_guard.add(s["unix"])
                            g.write(json.dumps({"source":"guard","unix":s["unix"],"cpu_max_C":max(s["cpu_C"].values()),
                              "gpu_C":s["gpu"]["gpu_C"],"hbm_C":s["gpu"]["hbm_C"],"gpu_W":s["gpu"]["power_W"],
                              "fan_max_pct":max(s["fans"]["percent"]),"available_kib":s["available_kib"]})+"\n")
                    w=read(Path("/var/lib/xeon-watch/latest.json"))
                    if w["unix"] not in seen_watch:
                        seen_watch.add(w["unix"])
                        g.write(json.dumps({"source":"watch","unix":w["unix"],"bmc_unix":w["bmc"]["observed_unix"],
                            "max_dimm_C":max(w["bmc"]["dimms_C"].values()),"alert":bool(w["active_alerts"]),"error":bool(w["errors"] or w["bmc"]["bad_sensor_status"])})+"\n")
                stop.wait(max(.01,.40-(time.monotonic()-t)))
    except Exception as exc:
        result["error"]=repr(exc)
    finally:
        reader.close()
        write(run/"IMC.json",result)


def pc_samples(run,driver,stop):
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    observations=[]
    for c in (0,1):
        base=f"pc-{c}"
        obs={"case":c};observations.append(obs)
        proc=None
        try:
            ready=run/(base+"-ready.json")
            while not ready.is_file():
                if stop.wait(.05):return
            pid=read(ready)["pid"]
            assert read(run/"health.json")["pid"]==pid and (Path("/proc")/str(pid)/"exe").resolve()==driver
            cmd=["/usr/bin/perf","record","-q","--no-buildid-cache","--no-buildid","--sample-cpu","-T","-k","mono","-F","79",
                 "-e","cpu-clock:u","-p",str(pid),"-o",str(run/(base+".local.data"))]
            obs.update(pid=pid,command=cmd,started_ns=time.monotonic_ns())
            with (run/(base+".stderr")).open("xb") as err:
                proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=err)
                time.sleep(.20)
                obs["running_at_arm"]=proc.poll() is None
                write(run/(base+"-armed.json"),{"ns":time.monotonic_ns(),"running":obs["running_at_arm"]})
                limit=time.monotonic()+120
                while not (run/(base+"-done.json")).is_file():
                    if stop.wait(.04) or time.monotonic()>limit:break
        except Exception as exc:
            obs["error"]=repr(exc)
        finally:
            if proc:
                if proc.poll() is None:proc.send_signal(signal.SIGINT)
                try:obs["exit_code"]=proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill();obs["exit_code"]=proc.wait()
            write(run/"PC_RECORDS.json",observations)


def start(run,driver,observe):
    stop=threading.Event()
    threads=[threading.Thread(target=memory_traffic,args=(run,stop))]
    if observe:threads.append(threading.Thread(target=pc_samples,args=(run,driver,stop)))
    for thread in threads:thread.start()
    return stop,threads
