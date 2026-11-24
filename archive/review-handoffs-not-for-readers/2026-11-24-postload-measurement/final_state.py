"""Check the final ordinary API and record the real scope of the temporary candidate."""
import json
from pathlib import Path
import time
from capture_production import capture_snapshot

ROOT=Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_bytes())


def main():
    assert read(ROOT/"trial/STATE.json")["completed"]
    pilot=ROOT/"pilot24"
    if pilot.exists():
        assert read(ROOT/"RESTORED_PILOT24.json")["normal_API_restored"]
        assert not read(ROOT/"API_WINDOW_PILOT24.json")["restore_required"]
        assert capture_snapshot("production-final")==0
        current=read(ROOT/"production-final-live.json")
        candidate=read(pilot/"READY.json"); h=read(pilot/"health.json")
        assert h["pid"]==candidate["pid"] and h["status"]=="stopped_by_request"
        assert not (pilot/"STOP_REQUESTED").exists()
        assert not (Path("/proc")/str(candidate["pid"])).exists() or int(
            (Path("/proc")/str(candidate["pid"])/"stat").read_text().rsplit(")",1)[1].split()[19])!=candidate["start_ticks"]
        recent=h["recent_samples"]
        selected={"pid":h["pid"],"boot":h["boot_id"],"started_unix":h["started_unix"],"finished_unix":h["finished_unix"],
            "child_returncode":h.get("child_returncode"),"guard_status":h["status"],"guard_error":h.get("error"),
            "initial_and_retained_recent_samples_scope":"Maxima are for the final rolling samples in the existing native supervisor.",
            "samples":len(recent),"cpu_max_C":max(max(s["cpu_C"].values()) for s in recent),
            "gpu_max_C":max(s["gpu"]["gpu_C"] for s in recent),"hbm_max_C":max(s["gpu"]["hbm_C"] for s in recent),
            "gpu_sampled_peak_W":max(s["gpu"]["power_W"] for s in recent),"fans_max":max(max(s["fans"]["percent"]) for s in recent),
            "min_available_GiB":min(s["available_kib"] for s in recent)/1024**2}
        (pilot/"GUARD_SUMMARY.json").write_text(json.dumps(selected,indent=2)+"\n")
    else:
        current=read(ROOT/"production-after-live.json")
    old=read(ROOT/"production-before-live.json")
    assert current["API_health"]=="ok" and current["backing_inodes_verified"] and current["command_and_configured_environment_match"]
    assert current["mapped_runtime_hashes"]==old["mapped_runtime_hashes"] and current["config_sha256"]==old["config_sha256"]
    assert not current["monitor_errors"] and not current["active_alerts"]
    keys=("pid","API_health","unix","boot_id","config_sha256","services","monitor_errors","active_alerts",
          "gpu_watts_limit","gpu_ecc","backing_inodes_verified","mapped_runtime_hashes","command_and_configured_environment_match")
    final={k:current[k] for k in keys}
    final.update(normal_decode=12,normal_prefill=24,temporary_candidate_retained_in_service=False,
                 window_closed=True,initial_pid=old["pid"],first_restored_pid=read(ROOT/"RESTORED_POSTSWAP.json")["pid"])
    (ROOT/"FINAL_API.json").write_text(json.dumps(final,indent=2)+"\n")
    print("Verified original ordinary 12/24 API",final["pid"],final["API_health"])


if __name__=="__main__":
    main()
