"""Recalculate the observed rates from the two actual sample times."""
import collections
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
MIB=2**20


def load(p):
    return json.loads(p.read_text())


def disk(row,device):
    for s in row["system_diskstats"]["text"].splitlines():
        parts=s.split()
        if parts[2]==device:
            return list(map(int,parts[3:]))
    raise ValueError(device)


def pressure(row,k,mode):
    line=next(s for s in row["pressure"][k]["text"].splitlines() if s.split()[0]==mode)
    return int(line.split("total=")[-1])


def stats(vals):
    return {"min":min(vals),"median":statistics.median(vals),"max":max(vals)}


def kib(row,k,group="status"):
    return int(row[group]["data"][k].split()[0])


def main(root):
    repo=root/"observation"
    rows=[json.loads(line) for line in (repo/"samples.jsonl").read_text().splitlines()]
    meta=load(repo/"metadata.json")
    summary=load(repo/"summary.json")
    cfg=load(root/"CONFIG_OBSERVED.json")
    begin,end=rows[0],rows[-1]
    hz=meta["clock_ticks_per_second"]
    duration=end["monotonic_s"]-begin["monotonic_s"]
    topology={c["cpu"]:int(c["physical_package_id"]["text"]) for c in meta["topology"]["cpus"]}
    device=cfg["model_storage"]["device"]
    packages={socket:next(k for k in begin["temperatures"] if k.endswith(f"/Package id {socket}/input"))
              for socket in sorted(set(topology.values()))}
    jobs=[]
    count_d=0
    gpu=[]; dimm=[]; fan=[]; alerts=[]; warnings=[]
    keys=set(begin["counters"])
    changes=[]
    for i,r in enumerate(rows):
        ps=r["process_stat"]["data"]
        assert [ps["pid"],ps["starttime_ticks"]]==meta["pid_identity"]
        for field in ("process_stat","status","io","meminfo","system_cpu","system_diskstats","vmstat"):
            assert r[field]["status"]=="ok"
        count_d+=ps["state"]=="D"
        assert set(r["counters"])==keys and all(v["status"]=="ok" for v in r["counters"].values())
        for key in keys:
            if r["counters"][key]["data"]!=begin["counters"][key]["data"]:
                changes.append({"sample":i,"counter":key,"value":r["counters"][key]["data"]})
        w=r["existing_watch"]
        if w["status"]!="ok" or w.get("observation_warnings"):
            warnings.append(i)
        else:
            if w["active_alerts"] or w["errors"] or w["bmc"]["bad_sensor_status"]:
                alerts.append(i)
            gpu.append(w["gpu"])
            dimm.append(max(w["bmc"]["dimms_C"].values()))
            fan.append(max(w["bmc"]["fan_percent"]))
        if i==0:
            continue
        a,b=rows[i-1],r
        dt=b["monotonic_s"]-a["monotonic_s"]
        io=b["io"]["data"]["read_bytes"]-a["io"]["data"]["read_bytes"]
        assert io>=0 and io==b["deltas"]["read_bytes_delta"] and not b["deltas"]["discontinuities"]
        x,y=disk(a,device),disk(b,device)
        pkg_busy={}
        for socket in packages:
            n=0
            for cpu,s in topology.items():
                if s==socket:
                    ca=a["system_cpu"]["data"][f"cpu{cpu}"]; cb=b["system_cpu"]["data"][f"cpu{cpu}"]
                    n+=sum(cb[j]-ca[j] for j in (0,1,2,5,6,7))
            pkg_busy[str(socket)]=n/hz/dt
        jobs.append(dict(dt=dt,pid_bytes=io,pid_MiB_s=io/MIB/dt,
                         cpu_core_equivalents=b["deltas"]["process_average_logical_CPUs"],
                         disk_read_bytes=(y[2]-x[2])*512,disk_busy_ms=y[9]-x[9],
                         disk_write_bytes=(y[6]-x[6])*512,system_busy_cpus_by_socket=pkg_busy))
    def metric_total(k):return sum(j[k] for j in jobs)
    def fields(field,names):
        return {k:{"start":begin[field]["data"][k],"end":end[field]["data"][k],
                "delta":end[field]["data"][k]-begin[field]["data"][k]} for k in names}
    temp={s:[r["temperatures"][k]["data"]/1000 for r in rows] for s,k in packages.items()}
    tt=[r["thread_snapshot"] for r in rows if "thread_snapshot" in r]
    masks=collections.Counter(); lastcpu=collections.Counter(); truncated=0; missing=0
    for snapshot in tt:
        assert snapshot["status"]=="ok"
        truncated+=snapshot["truncated"]
        for t in snapshot["rows"]:
            if not t["identity_stable"] or t["status"]["status"]!="ok" or t["stat"]["status"]!="ok":
                missing+=1;continue
            masks[t["status"]["data"]["Cpus_allowed_list"]]+=1
            lastcpu[str(topology[t["stat"]["data"]["last_cpu"]])]+=1
    counter_names=lambda frag:len([k for k in keys if frag in k])
    mounts=cfg["model_storage"]
    return dict(sample_count=len(rows),interval_count=len(jobs),first_utc=begin["utc"],last_utc=end["utc"],
      span_s=duration,observer=summary,interval_seconds=stats([j["dt"] for j in jobs]),
      pid_disk_bytes=metric_total("pid_bytes"),pid_disk_MiB_per_s=metric_total("pid_bytes")/MIB/duration,
      pid_interval_MiB_per_s=stats([j["pid_MiB_s"] for j in jobs]),
      cpu_logical_equiv=sum(j["cpu_core_equivalents"]*j["dt"] for j in jobs)/duration,
      process_cpu_increment_s=sum(j["cpu_core_equivalents"]*j["dt"] for j in jobs),leader_D_samples=count_d,
      io_counters=fields("io",("read_bytes","write_bytes","rchar","wchar")),
      major_minor_faults=fields("process_stat",("majflt","minflt")),
      process_memory_kib={k:{"start":kib(begin,k),"end":kib(end,k),"delta":kib(end,k)-kib(begin,k)} for k in ("VmRSS","RssFile","RssAnon","VmLck")},
      storage=mounts,device_read_bytes=metric_total("disk_read_bytes"),device_write_bytes=metric_total("disk_write_bytes"),
      device_read_MiB_per_s=metric_total("disk_read_bytes")/MIB/duration,
      device_io_ticks_percent=metric_total("disk_busy_ms")/10/duration,
      system_PSI_stall_pct={k:{mode:(pressure(end,k,mode)-pressure(begin,k,mode))/duration/10000 for mode in ("some","full")} for k in ("memory","io")},
      system_swap=fields("vmstat",("pswpin","pswpout")),system_available_GiB=stats([kib(r,"MemAvailable","meminfo")/1024**2 for r in rows]),
      system_socket_logical_CPU_equiv={s:stats([j["system_busy_cpus_by_socket"][s] for j in jobs]) for s in sorted(jobs[0]["system_busy_cpus_by_socket"])},
      temp_C={str(s):stats(ls) for s,ls in temp.items()},paired_S1_minus_S0_C=stats([b-a for a,b in zip(temp[0],temp[1])]),
      package_or_core_crit_alarms=sum(v["data"]!=0 for r in rows for k,v in r["temperatures"].items() if k.endswith("/crit_alarm")),
      process_threads=stats([r["process_stat"]["data"]["threads"] for r in rows]),thread_snapshot_count=len(tt),
      stable_thread_observations_per_allowed_list=dict(masks),stable_thread_observations_per_last_socket=dict(lastcpu),
      truncated_thread_snapshots=truncated,unavailable_or_unstable_thread_rows=missing,
      counters={"edac_files":counter_names("/edac/"),"thermal_files":counter_names("/thermal_throttle/"),"changes":changes},
      gpu_util_pct=stats([g["util_percent"] for g in gpu]) if gpu else None,
      gpu_used_MiB=stats([g["used_MiB"] for g in gpu]) if gpu else None,
      gpu_board_W=stats([g["power_W"] for g in gpu]) if gpu else None,
      dimm_max_C=max(dimm) if dimm else None,fan_max_pct=max(fan) if fan else None,
      fresh_watch_samples=len(gpu),watch_warnings_at_sample=warnings,watch_alerts_at_sample=alerts)


if __name__=="__main__":
    root=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT
    data=main(root)
    if "--check" in sys.argv:
        assert data==load(root/"OBSERVATION_ANALYSIS.json")
        print("Reproduced",data["sample_count"],"samples and",data["interval_count"],"intervals")
    else:
        (root/"OBSERVATION_ANALYSIS.json").write_text(json.dumps(data,indent=2)+"\n")
