"""Recalculate the measurements offline from ordinary JSON and stress-ng text."""
import json
from pathlib import Path
import re
import statistics
import sys

PHASES=("P0","P1","P2","P5","P6")


def read(p):
    return json.loads(p.read_text())


def timed_stats(rows,sensors):
    def extrema(key):
        v=[row["temp_mC"][key]/1000 for row in rows]
        return {"first":v[0],"min":min(v),"median":statistics.median(v),"max":max(v),"last":v[-1]}
    return {k:extrema(k) for k in sensors}


def power(rows,ranges):
    # Intervals wrap in a single domain. These are domain estimates;
    # CPU package and DRAM stay separate. Never infer wall watts.
    out={}
    for key,r in ranges.items():
        j=0.0
        s=0.0
        peaks=[]
        for a,b in zip(rows,rows[1:]):
            dt=b["energy_t"]-a["energy_t"]
            du=b["energy_uj"][key]-a["energy_uj"][key]
            if du<0:
                du+=r["range_uj"]
            assert dt>0 and du>=0
            j+=du*1e-6
            s+=dt
            peaks.append(du*1e-6/dt)
        if s:
            out[key]={"mean_W":j/s,"max_interval_W":max(peaks),"interval_total_s":s}
    return out


def analyze(root):
    data=root/"evidence"
    hw=read(data/"hardware.json")
    before,after=[read(data/(name+".json")) for name in ("before","after")]
    assert hw["boot"]==before["boot"]==after["boot"]
    assert len(hw["processors"])==2 and len(hw["dimms"])==16
    assert not before["models"] and not after["models"]
    assert all("Id=llm-deepseek41-api.service" in d["units"] and "ActiveState=inactive" in
               next(s for s in d["units"].split("\n\n") if "Id=llm-deepseek41-api.service" in s)
               for d in (before,after))
    assert before["counters"].keys()==after["counters"].keys()
    delta={k:after["counters"][k]-v for k,v in before["counters"].items()}
    assert not any(delta.values()) and not before["journal_hardware_matches"] and not after["journal_hardware_matches"]
    assert (data/"before-sel.txt").read_bytes()==(data/"after-sel.txt").read_bytes()
    observed={}
    for r in hw["cpus"]:
        if r["online"] and r["allowed"]:
            observed.setdefault((r["socket"],r["core"]),r)
    reps=sorted(observed.values(), key=lambda r:r["cpu"])
    assert len(reps)==24
    sensors=hw["sensors"]
    alltemps={}
    phases={}
    covered=[]
    memory=[]
    command_count=0
    for phase in PHASES:
        d=data/phase
        start=read(d/"start.json")
        end=read(d/"end.json")
        assert end["status"]=="pass" and start["boot"]==hw["boot"] and start["uid"]>0
        rows=[json.loads(line) for line in (d/"samples.jsonl").read_text().splitlines()]
        assert all(not r["stop_reasons"] and not r["alerts"] and not r["monitor_errors"] and
                   set(r["temp_mC"])==set(sensors) and all(t<start["stop_mC"][k] for k,t in r["temp_mC"].items()) for r in rows)
        for k in sensors:
            alltemps.setdefault(k,[]).extend(r["temp_mC"][k] for r in rows)
        subset=sorted(d.glob("s*-cpu*.json"),key=lambda path:read(path)["start"]["monotonic_s"])
        phase_ops=[]
        for path in subset:
            v=read(path)
            assert v["status"]=="pass" and v["exit_code"]==0 and v["stressor_wall_s"]>=v["seconds"]-1
            kind="vm" if v["size_mib"] else "cpu"
            text=path.with_suffix(".txt").read_text()
            op=int(re.search(r"\]\s+"+kind+r"\s+(\d+)\s+[0-9.]+",text)[1])
            assert op==v["bogo_ops"] and op>0
            assert all(re.search(r"\b"+s+r":\s+0\b",text) for s in ("failed","skipped","metrics untrustworthy"))
            assert re.search(r"passed:\s+1\b",text) and "--verify" in v["command"]
            active=set(v["active_worker_pids"])
            cpu=v["target"]["cpu"]
            seen=[p for snap in v["snapshots"] for p in snap["tree"] if p["pid"] in active]
            assert active and {p["last_cpu"] for p in seen if p["ticks"]>1}=={cpu}
            assert all(p["allowed"]==str(cpu) for p in seen if p["ticks"]>1)
            measured=[r for r in rows if r["test"]==v["label"]]
            assert measured
            sockets=[k for k in sensors if "/Package" in k]
            by_socket=timed_stats(measured,sockets)
            prior=next(r for r in rows if r["test"]=="before-"+v["label"])
            item=dict(cpu=cpu,socket=v["target"]["socket"],core=v["target"]["core"],method=v["method"],stressor_wall_s=v["stressor_wall_s"],
                      bogo_ops=op,observed_workers=sorted(active),package_C=by_socket,
                      package_start_before_C={k:prior["temp_mC"][k]/1000 for k in sockets},result_file=str(path.relative_to(root)))
            if phase=="P2":
                covered.append((cpu,v["method"]))
            if kind=="vm":
                sz=v["size_mib"]*1024**2
                node=v["target"]["node"]
                assert any(x["pid"] in active and x["policy"]==f"bind:{node}" and set(x["nodes"])=={str(node)}
                           and x["nodes"][str(node)]*x["page_kib"]*1024==x["bytes_resident"]==sz for x in v["large_maps"])
                passes=op*(1<<12)*64//sz
                assert passes==v["minimum_full_checkerboard_passes"] and passes>=1
                memory.append(dict(node=node,cpu=cpu,bytes_verified_per_full_pass=sz,minimum_complete_passes=passes,
                                   ops=op,wall_s=v["stressor_wall_s"],result_file=item["result_file"]))
            phase_ops.append(item)
        gaps=[b["monotonic_s"]-a["monotonic_s"] for a,b in zip(rows,rows[1:])]
        assert all(g>0 for g in gaps)
        summary=dict(start=start["start"],end=end["end"],samples=len(rows),sample_interval_median_s=statistics.median(gaps),
                   sample_interval_max_s=max(gaps),span_s=rows[-1]["monotonic_s"]-rows[0]["monotonic_s"],
                   temp_C=timed_stats(rows,sensors),power=power(rows,start["rapl"]),jobs=phase_ops,
                   dimm_max_C=max(max(r["dimm"].values()) for r in rows),max_fan_pct=max(max(r["fan_duty"]) for r in rows),
                   gpu_max_util_pct=max(r["gpu"]["util_percent"] for r in rows))
        phases[phase]=summary
        command_count+=len(phase_ops)
    assert sorted(covered)==sorted((r["cpu"],m) for r in reps for m in ("int64","sqrt"))
    assert len(memory)==4 and sorted((v["node"],v["bytes_verified_per_full_pass"]) for v in memory)==[(n,sz) for n in (0,1) for sz in (2**28,2**30)]
    assert len(phases["P1"]["jobs"])==4 and command_count==56
    return dict(boot=hw["boot"],representatives=reps,physical_cores_checked=len(reps),cpu_jobs=command_count-len(memory),
                memory_jobs=memory,total_completed_stressors=command_count,phases=phases,
                counters_read=len(delta),changed_counters={k:v for k,v in delta.items() if v},
                journal_delta_text_records=after["journal_text_records"],journal_error_matches=after["journal_hardware_matches"],
                new_sel_records=[],highest_core_or_package_C=max(max(ts) for ts in alltemps.values())/1000,
                peak_C_by_sensor={k:max(ts)/1000 for k,ts in alltemps.items()},
                all_measured_phases_have_no_monitor_alerts=True)


if __name__=="__main__":
    root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
    result=analyze(root)
    if "--check-only" in sys.argv:
        assert result==read(root/"MEASUREMENTS.json")
        print("Replay passed:",result["total_completed_stressors"],"verified stressors,",result["physical_cores_checked"],"physical cores.")
    else:
        (root/"MEASUREMENTS.json").write_text(json.dumps(result,indent=2)+"\n")
