"""Observe scheduling during the one fixed request. No second inference or perf attachment."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def stat(path):
    fields=(path/"stat").read_text().rsplit(")",1)[1].split()
    return int(fields[19]),int(fields[36])


def sample(pid,start):
    base=Path("/proc")/str(pid)
    assert stat(base)[0]==start
    a=time.perf_counter_ns()
    tasks=[]
    for folder in (base/"task").iterdir():
        tid=int(folder.name)
        try:
            st,cpu=stat(folder)
            runtime=int((folder/"schedstat").read_text().split()[0])
            allowed=sorted(os.sched_getaffinity(tid))
            after,cpu=stat(folder)
            if after==st:
                tasks.append([tid,st,cpu,runtime,allowed])
        except (FileNotFoundError,ProcessLookupError):
            pass
    return dict(begin=a,end=time.perf_counter_ns(),tasks=tasks)


def main():
    s=json.loads((ROOT/"production-before/snapshot.json").read_text())["process"]
    pid=s["pid"];start=s["stat_at_collection_end"]["starttime_ticks"]
    folder=ROOT/"normal-scheduling"
    folder.mkdir()
    bind=max(os.sched_getaffinity(0))
    original=os.sched_getaffinity(0)
    t0=time.process_time()
    intervals=[]
    proc=None
    # The capture client inherits the original allowance; only our sampling
    # process moves to CPU 47 after the launch.
    intervals.append(sample(pid,start))
    cmd=["sudo","-n","/usr/bin/python3",str(ROOT/"capture_production.py"),"stream"]
    with (folder/"client.out").open("x") as out:
        proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        try:
            os.sched_setaffinity(0,{bind})
            while proc.poll() is None:
                before=time.monotonic()
                intervals.append(sample(pid,start))
                try:
                    proc.wait(timeout=max(0,0.35-(time.monotonic()-before)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            os.sched_setaffinity(0,original)
            rc=proc.wait(timeout=350)
    (folder/"samples.json").write_text(json.dumps(intervals,separators=(",",":"))+"\n")
    record={"pid":pid,"start_ticks":start,"sampler_cpu":bind,"samples":len(intervals),"sampler_cpu_seconds":time.process_time()-t0,
            "client_return":rc,"same_pid_after":stat(Path("/proc")/str(pid))[0]==start,
            "phase_boundaries":"client perf_counter_ns. Only whole measurement intervals within received-content phase qualify."}
    (folder/"run.json").write_text(json.dumps(record,indent=2)+"\n")
    print(record)
    if rc:
        raise SystemExit(rc)


if __name__=="__main__":
    main()
