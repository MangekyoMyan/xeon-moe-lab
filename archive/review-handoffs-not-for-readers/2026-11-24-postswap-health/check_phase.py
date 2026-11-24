"""Run one individually selected single-worker phase with checked results."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from guard import Guard, memory_kib
from host import CPU, ENV, EV, NODE, ROOT, command, cpu_map, get, now, save

EXE = ROOT / "toolroot/usr/bin/stress-ng"
LIB = ROOT / "toolroot/usr/lib/x86_64-linux-gnu"


def representatives():
    seen={}
    for row in cpu_map():
        if row["online"] and row["allowed"]:
            seen.setdefault((row["socket"],row["core"]),row)
    rows=sorted(seen.values(),key=lambda x:(x["socket"],x["cpu"]))
    sockets=sorted({r["socket"] for r in rows})
    assert len(sockets)==2 and all(sum(r["socket"]==s for r in rows)==12 for s in sockets)
    return rows


def descendants(pid):
    pending=[pid]
    result=[]
    while pending:
        parent=pending.pop()
        proc=Path(f"/proc/{parent}")
        try:
            pending.extend(map(int,get(proc/"task"/str(parent)/"children").split()))
            st=get(proc/"stat").split(") ",1)[1].split()
            s=dict(l.split(":",1) for l in get(proc/"status").splitlines() if ":" in l)
            result.append(dict(pid=parent,comm=s["Name"].strip(),allowed=s["Cpus_allowed_list"].strip(),
                               last_cpu=int(st[36]),ticks=int(st[11])+int(st[12]),start_tick=int(st[19]),
                               rss_kib=int(s.get("VmRSS","0").split()[0])))
        except (FileNotFoundError,ProcessLookupError):
            continue
    return result


def numa_region(proc):
    p=Path(f"/proc/{proc['pid']}/numa_maps")
    try:
        maps=get(p)
    except PermissionError:
        maps=command(["sudo","-n","cat",str(p)])
    regions=[]
    for line in maps.splitlines():
        parts=line.split()
        nodes={int(k[1:]):int(v) for k,v in re.findall(r"\b(N\d+)=(\d+)",line)}
        if not nodes:
            continue
        page_kib=int(re.search(r"kernelpagesize_kB=(\d+)",line)[1])
        size=page_kib*1024*sum(nodes.values())
        if size >= 64*1024**2:
            regions.append(dict(pid=proc["pid"],policy=parts[1],page_kib=page_kib,nodes=nodes,
                                bytes_resident=size,raw_without_address=" ".join(parts[1:])))
    return regions


def stop_test(proc, observed):
    # timeout is our session leader. Also cover any same-owner stress-ng
    # descendants which left the process group, guarding against PID reuse.
    observed.update({p["pid"]:p["start_tick"] for p in descendants(proc.pid)})
    for sig in (signal.SIGINT,signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid,sig)
        for pid,start in observed.items():
            try:
                st=get(Path(f"/proc/{pid}/stat")).split(") ",1)[1].split()
                if int(st[19])==start and st[0]!="Z":
                    os.kill(pid,sig)
            except (FileNotFoundError,ProcessLookupError):
                pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def results(log, kind):
    text=log.read_text(errors="replace")
    m=re.search(r"\]\s+"+kind+r"\s+(\d+)\s+([0-9.]+)\s",text)
    if not m or not int(m[1]) or not re.search(r"passed:\s+1\b",text):
        raise RuntimeError(f"missing verified {kind} work in {log.name}")
    for what in ("skipped","failed","metrics untrustworthy"):
        if not re.search(re.escape(what)+r":\s+0\b",text):
            raise RuntimeError(f"{what} summary in {log.name}")
    return int(m[1]),float(m[2])


def run_one(g, row, method, seconds, size_mib=None):
    cpu=row["cpu"]
    node=row["node"]
    kind="vm" if size_mib else "cpu"
    tag=f"s{row['socket']}-cpu{cpu}-{method}"+(f"-{size_mib}MiB" if size_mib else "")
    name=g.dir/tag
    log=name.with_suffix(".txt")
    out=name.with_suffix(".json")
    if out.exists() or log.exists():
        raise RuntimeError(f"duplicate {tag}")
    if size_mib:
        if memory_kib(NODE/f"node{node}"/"meminfo")["MemFree"] < (size_mib+2048)*1024:
            raise RuntimeError("node memory headroom")
        launcher=["numactl",f"--physcpubind={cpu}",f"--membind={node}"]
        args=["--vm","1","--vm-bytes",f"{size_mib}M","--vm-method",method,
              "--vm-madvise","normal","--vm-keep","--oomable"]
    else:
        launcher=["taskset","-c",str(cpu)]
        args=["--cpu","1","--cpu-method",method]
    cmd=["timeout","--signal=INT","--kill-after=5s",f"{seconds+15}s",*launcher,str(EXE),*args,
         "--verify","--abort","--timeout",f"{seconds}s","--metrics-brief"]
    record=dict(command=cmd,target=row,seconds=seconds,method=method,size_mib=size_mib,
                snapshots=[],large_maps=[],start=now(),label=tag)
    g.tick("before-"+tag)
    child=None
    observed={}
    try:
        with log.open("x") as f:
            child=subprocess.Popen(cmd,start_new_session=True,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                   cwd=g.dir,env={**ENV,"LD_LIBRARY_PATH":str(LIB)})
            record["pid"]=child.pid
            while child.poll() is None:
                loop=time.monotonic()
                g.tick(tag)
                tree=descendants(child.pid)
                record["snapshots"].append({"t":loop,"tree":tree})
                for item in tree:
                    observed[item["pid"]]=item["start_tick"]
                if size_mib and not any(x["bytes_resident"]>=size_mib*1024**2 for x in record["large_maps"]):
                    for item in tree:
                        if item["rss_kib"]>=size_mib*1024 and item["comm"].startswith("stress-ng"):
                            record["large_maps"].extend(numa_region(item))
                try:
                    child.wait(timeout=max(0,2-(time.monotonic()-loop)))
                except subprocess.TimeoutExpired:
                    pass
        record["exit_code"]=child.returncode
        if child.returncode!=0:
            raise RuntimeError(f"{tag} child exit {child.returncode}")
        ops,elapsed=results(log,kind)
        record["bogo_ops"]=ops
        record["stressor_wall_s"]=elapsed
        if elapsed<seconds-1.0:
            raise RuntimeError("stressor finished early")
        samples=[p for snap in record["snapshots"] for p in snap["tree"] if p["comm"].startswith("stress-ng") and p["ticks"]>1]
        active=[p for p in samples if p["ticks"]>=max(5,os.sysconf("SC_CLK_TCK")*(seconds-5)*0.60)]
        if not active or any(p["allowed"]!=str(cpu) or p["last_cpu"]!=cpu for p in samples):
            raise RuntimeError("missing active worker on the single intended CPU")
        record["active_worker_pids"]=sorted({p["pid"] for p in active})
        if size_mib:
            nbytes=size_mib*1024**2
            confirmed=[x for x in record["large_maps"] if x["pid"] in record["active_worker_pids"] and
                       x["bytes_resident"]==nbytes and x["policy"]==f"bind:{node}" and set(x["nodes"])=={node}]
            if not confirmed:
                raise RuntimeError("no matching full-size worker VMA on intended node")
            # v0.19.02 increments once per verified 64 bytes and presents
            # the accumulated count shifted right by 12 at deinit.
            record["minimum_full_checkerboard_passes"]=ops*(1<<12)*64//nbytes
            if not record["minimum_full_checkerboard_passes"]:
                raise RuntimeError("no full checkerboard pass established")
        g.tick("after-"+tag)
        record["status"]="pass"
        print(tag, "pass", ops, flush=True)
    except BaseException as exc:
        if child:
            stop_test(child,observed)
            record["exit_code"]=child.poll()
        record.update(status="stopped",error=str(exc))
        raise
    finally:
        record["end"]=now()
        save(out,record)


def run(phase):
    rows=representatives()
    sockets=sorted({r["socket"] for r in rows})
    first=[next(r for r in rows if r["socket"]==s) for s in sockets]
    prior={"P1":"P0","P2":"P1","P5":"P2"}[phase]
    if json.loads((EV/prior/"end.json").read_text())["status"]!="pass":
        raise RuntimeError(f"{prior} did not pass")
    g=Guard(phase)
    try:
        if phase=="P5":
            for r in first:
                for mib,sec in ((256,30),(1024,60)):
                    run_one(g,r,"checkerboard",sec,mib)
        else:
            for r in first if phase=="P1" else rows:
                for method in ("int64","sqrt"):
                    run_one(g,r,method,30 if phase=="P1" else 10)
        g.finish("pass")
    except BaseException as exc:
        g.finish("stopped",str(exc))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["P1","P2","P5"])
    run(parser.parse_args().phase)
