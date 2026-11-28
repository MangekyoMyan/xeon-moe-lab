import json
from pathlib import Path
from analyze import read

R=Path(__file__).resolve().parent
def main():
    before=read(R/"BEFORE.json");windows=[];stops={}
    files=[*R.glob("WINDOW_ADAPTIVE-*.json"),*(R/"entry").glob("WINDOW_BRANCH-CLI-*.json")]
    for p in sorted(files,key=lambda p:read(p)["started_unix"]):
        w=read(p);assert not w["restore_required"]
        r=read(p.with_name(p.name.replace("WINDOW_","RESTORED_",1)))
        for k in ("config_sha256","mapped_runtime_hashes","plan_sha256","model_revision","model_download_manifest_sha256"):
            assert r[k]==before[k]
        assert r["API_health"]=="ok" and r["command_and_configured_environment_match"] and r["backing_inodes_verified"]
        assert all(v is None for v in r["experimental_environment"].values())
        windows.append({"label":p.stem,"start":w["started_unix"],"end":w["finished_unix"],"normal_pid":r["pid"]})
    assert any("ADAPTIVE-COMPARISON" in x["label"] for x in windows) and any("BRANCH-CLI" in x["label"] for x in windows)
    guards=[]
    for p in [*(R/"runs").glob("*/health.json"),*(R/"entry/runs").glob("*/health.json")]:
        h=read(p);guards.append(h);assert h["status"]=="stopped_by_request" and h["child_returncode"]==0
        stops[p.parent.name]={"status":h["status"],"child_returncode":h["child_returncode"],"gpu_W":h["initial_gpu"]["limit_W"],
                      "ecc":h["initial_gpu"]["ecc"],"fan_limit_percent":h["limits"]["fan_percent"]}
    first=min(guards,key=lambda x:x["started_unix"]);now=read(Path("/var/lib/xeon-watch/latest.json"))
    assert first["boot_id"]==now["boot_id"] and first["initial_host"].keys()==now["host_counters"].keys()
    changes={k:[v,now["host_counters"][k]] for k,v in first["initial_host"].items() if now["host_counters"][k]!=v}
    out={"normal_api":"ok","pid":windows[-1]["normal_pid"],"ordinary_thresholds":[64,64],"ordinary_threads":[24,24],
       "config_sha256":before["config_sha256"],"four_dsos":before["mapped_runtime_hashes"],"model_revision":before["model_revision"],
       "closed_windows":windows,"guard_stops":stops,"alerts":list(now["active_alerts"]),"sensor_errors":list(now["errors"]),
       "host_counter_changes":changes,"host_counters":len(first["initial_host"]),
       "gpu_corrected_ECC":[first["initial_gpu"]["ecc_single"],now["gpu"]["ecc_single"]],
       "gpu_uncorrected_ECC":[first["initial_gpu"]["ecc_double"],now["gpu"]["ecc_double"]]}
    (R/"SERVICE_RESULT.json").write_text(json.dumps(out,indent=2)+"\n");print("normal api",out["pid"],"windows",len(windows))

if __name__=="__main__":main()
