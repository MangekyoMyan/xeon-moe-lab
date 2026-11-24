"""One status at handoff, separately timestamped from the 10-minute time series."""
import json
from pathlib import Path
import subprocess
import time
from readiness import ROOT, START, probe


def main():
    check=probe()
    health=json.loads(subprocess.check_output(["sudo","-n","cat",str(ROOT.parent/"deepseek41-bringup-20261101/health.json")]))
    markers=json.loads(subprocess.check_output(["sudo","-n","/usr/bin/python3","-I",str(ROOT/"marker_ranges.py"),"--post-window"]))
    watch=json.loads(Path("/var/lib/xeon-watch/latest.json").read_text())
    archived=[]
    with Path("/var/lib/xeon-watch/history.jsonl").open() as history:
        for line in history:
            rec=json.loads(line)
            if rec["boot_id"]==START["boot"] and markers["since_wall"]<=rec["unix"]<=markers["wall"]:
                archived.append({"unix":rec["unix"],"package_C":{k:v for k,v in rec["cpu"].items() if "Package" in k},
                    "fan_pct":rec["bmc"]["fan_percent"],"gpu_util_pct":rec["gpu"]["util_percent"],
                    "gpu_MiB":rec["gpu"]["used_MiB"],"alarmed":bool(rec["active_alerts"]),"sensor_errors":bool(rec["errors"])})
    recent=[{"unix":v["unix"],"cpu_sensor_C":v["cpu_C"],"max_CPU_C":max(v["cpu_C"].values()),"available_kib":v["available_kib"]}
            for v in health.get("recent_samples",[]) if markers["since_wall"]<=v["unix"]<=markers["wall"]]
    result={"scope":"single later check; not a performance or temperature trace","health_get":check,
            "guard":{k:health.get(k) for k in ("status","pid","boot_id","updated_unix","finished_unix")},
            "guard_error":bool(health.get("error")),"watch_age_s":time.time()-watch["unix"],"alerts":bool(watch["active_alerts"]),
            "separate_log_range":markers,"existing_postwindow_watch_records":archived,
            "existing_recent_guard_samples":recent}
    with (ROOT/"HANDOFF_CHECK.json").open("x") as f:
        json.dump(result,f,indent=2)
    print(check["utc"],check["state"],result["guard"]["status"],result["alerts"],
          [(k,v.get("bytes_added"),len(v["markers"])) for k,v in markers["files"].items()])


if __name__=="__main__":
    main()
