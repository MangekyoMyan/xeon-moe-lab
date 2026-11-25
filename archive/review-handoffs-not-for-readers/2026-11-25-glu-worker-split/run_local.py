"""Run only isolated CPU FFN test children; keep the ordinary server in place."""
import argparse
from pathlib import Path
import subprocess
import time
import os
from build_cpu import HERE, LAB, read, save, sha


def snapshot(pid):
    watch=read(Path("/var/lib/xeon-watch/latest.json"))
    mem=next(int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))*1024
    stat=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
    data={"time":time.time(), "watch_time":watch["unix"], "MemAvailable_B":mem,
          "normal_api_cpu_ticks":int(stat[11])+int(stat[12]), "normal_api_start_ticks":int(stat[19]),
          "active_alerts":watch["active_alerts"], "sensor_errors":watch["errors"]}
    if mem < 64*1024**3 or watch["active_alerts"] or watch["errors"] or abs(time.time()-watch["unix"])>90:
        raise RuntimeError(f"CPU test guard: {data}")
    return data


def ready_stage(name):
    r=read(HERE/"runs"/name/"RUN.json")
    assert r["passed"] and all(c["matched_saved_down"] for c in r["cells"])


def run_one(design, pid, name, stage, arm, cells):
    dest=HERE/"runs"/name
    dest.mkdir(parents=True,exist_ok=False)
    key="trace" if stage=="trace" else "runtime"
    lib=Path(design["arms"][arm][key])
    cpu=read(HERE/"BUILD.json")["arms"][arm]["outputs"][key]
    assert sha(lib)==cpu["sha256"]
    spec=dict(design["common"],CPU=str(lib),cells=cells,arm=arm,stage=stage)
    save(dest/"SPEC.json",spec)
    env={"PATH":"/usr/bin:/bin","LANG":"C","OMP_DYNAMIC":"FALSE","OMP_THREAD_LIMIT":"24",
         "OMP_MAX_ACTIVE_LEVELS":"1","OPENBLAS_NUM_THREADS":"1",
         "LD_LIBRARY_PATH":os.pathsep.join([str(lib.parent),str(Path(spec["base"]).parent)]),
         "CUDA_VISIBLE_DEVICES":""}
    command=["/usr/bin/python3","-B",str(LAB/"moe-implementation-20261026/release/without_thp.py"),
             "/usr/bin/numactl","--physcpubind=0-23","--interleave=0,1",
             str(HERE/"probe"),str(dest/"SPEC.json"),str(dest)]
    save(dest/"LAUNCH.json",{"command":command,"environment":env,"design_sha256":sha(HERE/"DESIGN.json"),
         "library_sha256":sha(lib),"probe_sha256":sha(HERE/"probe")})
    log=[snapshot(pid)]
    with (dest/"stdout.log").open("w") as stdout,(dest/"stderr.log").open("w") as stderr:
        with subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr) as child:
            try:
                while True:
                    try:
                        child.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    log.append(snapshot(pid))
                    if time.time()-log[0]["time"]>80: raise RuntimeError("FFN subset child exceeded 80 seconds")
                log.append(snapshot(pid))
                if child.returncode:
                    raise RuntimeError(f"{name} failed ({child.returncode}), see {dest}/stderr.log")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:child.kill()
                save(dest/"GUARD.json",log)
    if log[0]["normal_api_start_ticks"]!=log[-1]["normal_api_start_ticks"]:
        raise RuntimeError("ordinary API process changed")
    api_cpu=(log[-1]["normal_api_cpu_ticks"]-log[0]["normal_api_cpu_ticks"])/os.sysconf("SC_CLK_TCK")
    if api_cpu>max(0.2,(log[-1]["time"]-log[0]["time"])*0.5):
        raise RuntimeError("ordinary API active during diagnostic; preserve period for review")
    ready_stage(name)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("stage",choices=["quality","trace","timing"]);stage=a.parse_args().stage
    d=read(HERE/"DESIGN.json");pid=read(HERE/"LIVE_BEFORE.json")["verified_api_pid"]
    assert sha(HERE/"probe")==d["probe_sha256"] and sha(HERE/"BUILD.json")==d["CPU_build_record_sha256"]
    snap=snapshot(pid);time.sleep(0.4);now=snapshot(pid)
    assert now["normal_api_cpu_ticks"]-snap["normal_api_cpu_ticks"] < os.sysconf("SC_CLK_TCK")/5,"normal API busy"
    if stage!="quality":
        assert read(HERE/"QUALITY.json")["passed"]
    if stage=="timing":
        assert read(HERE/"TRACE.json")["passed"]
        for pair in d["pairs"]:
            for arm in pair["arms"]:
                run_one(d,pid,f"pair-{pair['pair']:02d}-{arm}","timing",arm,pair["cells"])
        print("Completed all 12 predeclared pairs.")
    else:
        for arm in d[stage+"_order"]:
            run_one(d,pid,f"{stage}-{arm}",stage,arm,[0,1])
        print("Completed",stage)


if __name__=="__main__":
    main()
