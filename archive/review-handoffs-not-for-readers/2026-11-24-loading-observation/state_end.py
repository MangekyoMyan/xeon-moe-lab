"""Reconcile the unit, its guard, child identity and one local end-of-window health GET."""
import datetime
import json
from pathlib import Path
import subprocess
import time
from readiness import ROOT, PROC, PID, START, probe, ident


def read(p):
    return json.loads(p.read_text())


def main():
    obs=read(ROOT/"observation/summary.json")
    ready=probe()
    with (ROOT/"READINESS_END.json").open("x") as f:
        json.dump(ready,f,indent=2)
    props=subprocess.check_output(["systemctl","show","llm-deepseek41-api.service","-p",
          "Id,ActiveState,SubState,MainPID,ControlGroup,InvocationID,ExecMainStartTimestamp,ActiveEnterTimestamp"],text=True)
    h=json.loads(subprocess.check_output(["sudo","-n","cat",str(ROOT.parent/"deepseek41-bringup-20261101/health.json")]))
    watch=read(Path("/var/lib/xeon-watch/latest.json"))
    data={"report_received_via_attachment":{"stability_checked":True,"method_or_additional_full_load_logs":"not supplied",
            "model_at_message":"starting","approximately_one_hour":"reported expectation",
            "fan_origin":"Dell according to the returned update","heat_from_socket_0_to_socket_1":"hypothesis"},
          "observed":{"snapshot_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"boot":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "original_child":START["model"],"original_supervisor":START["supervisor"],"unit_at_start":START["unit"],"unit_at_end":props,
            "original_identity_matches_now":PROC.exists() and ident(),"guard_at_start":START["health"],
            "guard_at_end":{k:h.get(k) for k in ("pid","status","boot_id","started_unix","updated_unix","finished_unix")},
            "guard_error_present":bool(h.get("error")),"early_health_read":read(ROOT/"READINESS_EARLY.json"),
            "end_health_read":ready,"window_summary":obs,"last_watch_age_s":time.time()-watch["unix"],
            "watch_has_alert":bool(watch["active_alerts"]),"watch_has_errors":bool(watch["errors"])},
          "changes":{"new_model_launches":0,"inference_requests_sent_for_this_study":0,"service_or_affinity_changes":0,
                     "cooling_or_power_changes":0}}
    with (ROOT/"CURRENT_STATE.json").open("x") as f:
        json.dump(data,f,indent=2)
    print(data["observed"]["original_identity_matches_now"],ready["state"],obs)


if __name__=="__main__":
    main()
