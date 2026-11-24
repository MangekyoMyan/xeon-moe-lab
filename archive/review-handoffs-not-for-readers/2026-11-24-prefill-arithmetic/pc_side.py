"""Sample the isolated worker-trace PID only; wait locally and close perf at the last sampled token."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from shared import HERE, PRIOR, fresh, read, sha

RUN=HERE/"runs/worker-trace"
SO=Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")


def main():
    assert os.geteuid()==0
    target=RUN/"worker-0-5.tsv"
    clock=time.monotonic()+3600
    while not target.is_file():
        if time.monotonic()>clock:raise TimeoutError("no worker trace was active")
        time.sleep(.2)
    pid=read(RUN/"health.json")["pid"]
    proc=Path("/proc")/str(pid)
    assert (proc/"exe").resolve()==HERE/"driver/driver"
    start_ticks=proc.joinpath("stat").read_text().rsplit(")",1)[1].split()[19]
    assert sha(SO)==read(PRIOR/"CPU_SAMPLES.json")["library_sha256"]
    mapped=[v.split() for v in (proc/"maps").read_text().splitlines() if v.endswith(str(SO)) and "r-xp" in v]
    assert len(mapped)==1
    m=mapped[0];st=SO.stat()
    assert st.st_ino==int(m[4]) and st.st_dev==os.makedev(*[int(i,16) for i in m[3].split(":")])
    fresh(HERE/"private/gomp-map.json",{"pid":pid,"start_ticks":start_ticks,"mapping":m})
    os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    data=HERE/"private/workers.perf"
    cmd=["/usr/bin/perf","record","-q","--no-buildid-cache","--no-buildid","--sample-cpu","-T","-k","mono",
         "-F","199","-e","cpu-clock:u","-p",str(pid),"-o",str(data)]
    start=time.monotonic_ns()
    with (HERE/"private/perf.stderr").open("x") as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        try:
            while p.poll() is None and (proc/"stat").exists() and time.monotonic_ns()-start<400_000_000_000:
                if (RUN/"worker-1-61.tsv").exists():break
                time.sleep(.07)
        finally:
            if p.poll() is None:p.send_signal(signal.SIGINT)
            rc=p.wait(timeout=15)
    assert rc in (0,-2) and data.stat().st_size>0
    fresh(HERE/"PC_ATTACHMENT.json",{"pid":pid,"start_ticks":start_ticks,"begin_ns":start,"end_ns":time.monotonic_ns(),
          "perf_status":rc,"command":cmd,"input_sha256":sha(data),"libgomp_sha256":sha(SO),"map_inode_checked":True})
    args=["/usr/bin/perf","script","--ns","-i",str(data),"-F","pid,tid,cpu,time,ip,sym,dso"]
    with (HERE/"private/workers.perf.txt").open("x") as out:
        subprocess.run(args,stdout=out,check=True,timeout=90)
    print("CPU clock samples captured",pid,data.stat().st_size,flush=True)


if __name__=="__main__":main()
