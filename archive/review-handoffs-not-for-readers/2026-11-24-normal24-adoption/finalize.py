"""Finish the ordinary 24/24 receipt from the live service and the completed stream."""
import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from adopt import ROOT,LAB,ENGINE,CONFIG,CANDIDATE,BACKUP,OLD,NEW,BOOT,read,save,digest,identity,get,snapshot_health,verify


def main():
    ack=read(ROOT/"ACCEPTANCE.json");before=read(ROOT/"BEFORE_IDENTITY.json")
    after=read(ROOT/"AFTER_IDENTITY.json");who=identity(NEW)
    assert who["pid"]==ack["pid"]==after["pid"] and who["start_ticks"]==ack["start_ticks"] and before["pid"]!=who["pid"]
    status,body=get("/health")
    assert status==200 and body.get("status")=="ok"
    final=verify("24_24","FINAL_LIVE.json")
    assert final["mapped_runtime_hashes"]==read(ROOT/"BEFORE.json")["mapped_runtime_hashes"] and final["gpu_ecc"]=="Enabled" and final["gpu_watts_limit"]==150
    verified=json.loads(subprocess.check_output([sys.executable,"-I",str(LAB/"current/check.py"),"verify","--live"],timeout=90))
    assert verified["verified_api_pid"]==who["pid"] and verified["checks_passed"]
    save("CURRENT_VERIFY_LIVE.json",verified)
    manifest=read(LAB/"current/runtime.json")
    assert (LAB/"current").resolve()==ROOT/"current" and manifest["service_config"]["sha256"]==NEW and manifest["inference"]["threads_decode"]==manifest["inference"]["threads_prefill"]==24
    cfg=CONFIG.read_bytes();old=BACKUP.read_bytes()
    assert digest(old)==OLD and digest(cfg)==digest(CANDIDATE.read_bytes())==NEW
    expected=json.loads(old);expected["command"][9]="24";assert expected==json.loads(cfg)
    perms=read(ROOT/"BACKUP.json");st=CONFIG.stat()
    assert (perms["uid"],perms["gid"],perms["file_mode"])==(st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode))
    start=read(ROOT/"HEALTH_BEFORE.json");end=snapshot_health()
    save("HEALTH_FINAL.json",end)
    g=[json.loads(l) for l in (ROOT/"GUARD_SAMPLES.jsonl").read_text().splitlines()]
    w=[json.loads(l) for l in (ROOT/"WATCH_SAMPLES.jsonl").read_text().splitlines()]
    assert g and w and all(r["pid"]==who["pid"] for r in g)
    base=start["host_counters"];cur=end["host_counters"]
    changes={k:[base.get(k),cur.get(k)] for k in cur.keys()|base.keys() if base.get(k)!=cur.get(k)}
    assert not changes and all(r["boot"]==BOOT and not any((r["counter_diff"],r["alert"],r["errors"])) for r in w)
    assert all(r["limit_W"]==150 and r["ecc"]=="Enabled" and max(r["fan_percent"])<=70 for r in g)
    health=read(ENGINE/"health.json")
    assert health["pid"]==who["pid"] and health["status"]=="running" and not health.get("error")
    journal=subprocess.check_output(["journalctl","-k","-b","--no-pager","-o","cat","--since",f"@{int(start['unix'])}"],text=True)
    error_regex=r"\[hardware error\]|machine check (?:exception|events logged)|EDAC .*\b(?:CE|UE)\b|temperature above threshold|critical temperature|cpu.*throttled|NVRM: Xid|PCIe Bus Error|oom-kill:|out of memory:|kernel panic"
    events=[s for s in journal.splitlines() if re.search(error_regex,s,re.I)]
    save("KERNEL_CHECK.json",{"since_unix":int(start["unix"]),"boot":BOOT,"time_unix":time.time(),"hardware_error_match_count":len(events),
         "scope":"Current-boot kernel journal, hardware/thermal/Xid/OOM patterns only"})
    assert not events
    phases=[json.loads(l) for l in (ROOT/"events.jsonl").read_text().splitlines()]
    original=read(ROOT/"BEFORE.json")
    record={"status":"adopted","recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "change":"normal API decode 12 to 24; prefill 24","authorized_choice":"User selected adoption as the normal profile",
      "normal_config":str(CONFIG),"original_sha256":OLD,"prepared_sha256":digest(CANDIDATE.read_bytes()),"actual_after_sha256":digest(cfg),
      "only_semantic_difference":{"field":"command[9]","flag":expected["command"][8],"before":"12","after":"24"},
      "backup":perms,"installed_owner_gid_mode_octal":[st.st_uid,st.st_gid,oct(stat.S_IMODE(st.st_mode))],
      "atomic_install":True,"boot_before":before["boot"],"boot_after":who["boot"],"old_model":before,"normal_model":who,
      "cutover_start_unix":phases[0]["unix"],"ordinary_ready_verified_unix":phases[-1]["unix"],
      "stop_to_identity_and_http_ready_verified_s":phases[-1]["mono"]-phases[0]["mono"],
      "old_DSO_hashes":original["mapped_runtime_hashes"],"actual_DSO_hashes":final["mapped_runtime_hashes"],
      "mapped_backing_inodes_verified":final["backing_inodes_verified"],"command_and_environment_match":final["command_and_configured_environment_match"],
      "monitor_services":final["services"],"approval_scope":"normal deployment and one short text acceptance on the existing endpoint",
      "request":ack,"operator_view":read(ROOT/"CURRENT_ENTRY.json"),"weight_check":"Existing download receipt and sizes, no new full GGUF read",
      "health":{"native_sample_count":len(g),"native_sample_duration_s":g[-1]["unix"]-g[0]["unix"],"watch_sample_count":len(w),
          "max_socket_C":{s:max(v["packages_C"][s] for v in g) for s in g[0]["packages_C"]},
          "max_CPU_C":max(v["cpu_max_C"] for v in g),"max_GPU_C":max(v["gpu_C"] for v in g),"max_HBM_C":max(v["hbm_C"] for v in g),
          "max_DIMM_C":max(v["dimm_max_C"] for v in w),"max_fan_percent":max(max(v["fan_percent"]) for v in g),
          "min_available_GiB":min(v["available_kib"] for v in g)/1024**2,"checked_host_counters":len(cur),"counter_changes":changes,
          "GPU_volatile_ECC_single_delta":end["gpu"]["ecc_single"]-start["gpu"]["ecc_single"],
          "GPU_volatile_ECC_double_delta":end["gpu"]["ecc_double"]-start["gpu"]["ecc_double"],
          "kernel_error_matches":len(events),"continuing_alarms":end["alerts"],"sensor_errors":end["errors"],
          "compact_reserve_bytes":manifest["inference"]["compact_packing_host_reserve_bytes"],
          "native_supervisor_MemAvailable_stop_GiB":3,"hardware_stop_marker":False},
      "scope_limits":["short text acceptance on normal 24/24","no vision, long-context, concurrent or sustained-load acceptance",
                      "no new-CPU CUPTI trace or new expert assignment in this adoption"]}
    save("ADOPTION_RECORD.json",record)
    print(record["status"],who["pid"],record["health"],sep="\n")


if __name__=="__main__":main()
