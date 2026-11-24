"""Save selected fields from the already-running guard; does not poll any sensor."""
from pathlib import Path
import json
import os
import time

ROOT=Path(__file__).resolve().parent
WATCH=Path("/var/lib/xeon-watch/latest.json")


def read(path):
    return json.loads(path.read_bytes())


def save_line(file, fields):
    file.write(json.dumps(fields,separators=(",",":"))+"\n")


def main():
    folder=ROOT/"guard-archive"
    folder.mkdir()
    hw=read(ROOT.parent/"postswap-health-20261124/evidence/hardware.json")
    packages={k.split("/")[0]:v["path"]+"_input" for k,v in hw["sensors"].items() if "/Package" in k}
    stages=[s["name"] for s in read(ROOT/"trial/DESIGN.json")["stages"]]
    initial=read(WATCH)
    expected=initial["host_counters"]
    (folder/"baseline.json").write_text(json.dumps({"boot":initial["boot_id"],"unix":initial["unix"],"counters":expected,
              "package_sensors":packages,"source":"guard health.json every ~5s + existing persistent monitor JSON, no hardware polling"},indent=2)+"\n")
    seen={name:set() for name in stages}
    last_watch=0; start=time.monotonic(); cpu_start=time.process_time()
    if len(os.sched_getaffinity(0))>1:
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    reason="deadline"
    with (folder/"guard.jsonl").open("x",buffering=1) as g,(folder/"watch.jsonl").open("x",buffering=1) as w:
        while time.monotonic()-start < 14*3600:
            a=time.monotonic_ns();wall=time.time_ns();b=time.monotonic_ns()
            state=read(ROOT/"trial/STATE.json")
            for name in stages:
                path=ROOT/"trial/runs"/name/"health.json"
                if not path.is_file():
                    continue
                guard=read(path)
                for row in guard.get("recent_samples",[]):
                    if row["unix"] in seen[name]:
                        continue
                    seen[name].add(row["unix"])
                    gpu=row["gpu"]
                    save_line(g,{"stage":name,"pid":guard.get("pid"),"boot":guard["boot_id"],"unix":row["unix"],
                          "reader_wall_ns":wall,"reader_mono_ns":(a+b)//2,"reader_bracket_ns":b-a,
                          "available_kib":row["available_kib"],"cpu_package_C":{s:row["cpu_C"][p] for s,p in packages.items()},
                          "cpu_peak_C":max(row["cpu_C"].values()),"gpu":{k:gpu[k] for k in (
                            "gpu_C","hbm_C","power_W","limit_W","ecc","gpu_util","memory_util","sm_MHz","mem_MHz") if k in gpu},
                          "fan":{k:row["fans"][k] for k in ("percent","mode")}})
            watch=read(WATCH)
            if watch["unix"]!=last_watch:
                last_watch=watch["unix"];bmc=watch["bmc"]
                cur=watch["host_counters"]
                changes={k:(expected.get(k),cur.get(k)) for k in expected.keys()|cur.keys() if expected.get(k)!=cur.get(k)}
                save_line(w,{"unix":last_watch,"bmc_unix":bmc["observed_unix"],"boot":watch["boot_id"],
                    "phase_when_read":state.get("phase"),"changes":changes,"alarms":bool(watch["active_alerts"]),
                    "sensor_error":bool(watch["errors"] or bmc["bad_sensor_status"]),"max_dimm_C":max(bmc["dimms_C"].values()),
                    "gpu":{k:watch["gpu"].get(k) for k in ("gpu_C","hbm_C","power_W","used_MiB","util_percent","ecc_single","ecc_double")},
                    "fan_percent":bmc["fan_percent"]})
            if watch["boot_id"]!=initial["boot_id"]:
                reason="boot changed";break
            if len(state["runs"])==len(stages) and all(v["status"]=="passed" for v in state["runs"].values()):
                reason="six stages passed";break
            if state.get("last_error") or state.get("completed") or any(v["status"]=="quality difference" for v in state["runs"].values()):
                reason="controller ended or quality gate differed";break
            time.sleep(8)
    (folder/"end.json").write_text(json.dumps({"reason":reason,"unique_guard_samples":{k:len(v) for k,v in seen.items()},
            "process_cpu_s":time.process_time()-cpu_start,"elapsed_s":time.monotonic()-start},indent=2)+"\n")
    print(reason)


if __name__=="__main__":
    main()
