"""Bounded use of Linux perf during the Japanese quality case only."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

def read(p):return json.loads(p.read_text())
def stamp():return time.monotonic_ns()

def observe(run, root, stop):
    folder=run/"quality-host"
    folder.mkdir(exist_ok=False)
    procs={}
    files=[]
    record={"scope":"Japanese quality prefill plus 64 forced calls. PC samples are for this PID. APERF/MPERF/TSC are CPU-wide on the process CPUs in this run SPEC.",
            "timebase":"CLOCK_MONOTONIC","commands":{},"start_brackets_ns":{},"exit_codes":{}}
    try:
        deadline=time.monotonic()+7200
        ready=run/"quality-observer-ready.json"
        while not ready.is_file():
            if stop.wait(.2) or time.monotonic()>deadline:raise RuntimeError("quality marker not reached")
        pid=read(ready)["pid"]
        driver=(root/"trial/driver").resolve()
        if (Path("/proc")/str(pid)/"exe").resolve()!=driver or read(run/"health.json")["pid"]!=pid:
            raise RuntimeError("quality PID did not match the supervised driver")
        cpus=read(run/"SPEC.json")["condition"]["process_cpu_ids"]
        cpu_list=",".join(map(str,cpus))
        record["target_pid"]=pid
        fmt=["/usr/bin/perf","stat","-a","-A","-C",cpu_list,"--no-big-num","-I","2000","-x",";","-e","msr/aperf/,msr/mperf/,msr/tsc/","-o",str(folder/"msr.csv")]
        samples=["/usr/bin/perf","record","-q","--no-buildid-cache","--no-buildid","--sample-cpu","-T","-k","mono","-F","79","-e","cpu-clock:u","-p",str(pid),"-o",str(folder/"worker.local.data")]
        for name, command in [("clocks",fmt),("workers",samples)]:
            log=(folder/(name+".stderr")).open("xb")
            files.append(log)
            record["commands"][name]=command
            a=stamp()
            procs[name]=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=log,env={"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C"})
            record["start_brackets_ns"][name]=[a,stamp()]
        time.sleep(.30)
        record["armed_ns"]=stamp()
        record["running_at_arm"]={name:p.poll() is None for name,p in procs.items()}
        (run/"quality-observer-armed.json").write_text(json.dumps({"monotonic_ns":record["armed_ns"],"perf_running":record["running_at_arm"]})+"\n")
        deadline=time.monotonic()+140
        while not (run/"quality-observer-finished.json").is_file():
            if stop.wait(.1) or time.monotonic()>deadline:raise RuntimeError("bounded quality window ended")
        record["completed_marker"]=read(run/"quality-observer-finished.json")
    except Exception as err:
        record["observation_error"]=str(err)
    finally:
        record["stop_ns"]=stamp()
        for p in procs.values():
            if p.poll() is None:p.send_signal(signal.SIGINT)
        for name,p in procs.items():
            try:p.wait(timeout=7)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            record["exit_codes"][name]=p.returncode
        for f in files:f.close()
        (folder/"OBSERVATION.json").write_text(json.dumps(record,indent=2)+"\n")
