"""Verify the ordinary service and all owned maintenance exits after numerical experiments."""
import json
import subprocess
import sys
from pathlib import Path
from shared import HERE,LAB,adopt,study,read,sha,fresh
from chat import post


def main():
    windows={}
    for tag in ("CHAT","WORKERS","PRECISION","CAUSAL"):
        w=read(HERE/f"WINDOW_{tag}.json");live=read(HERE/f"RESTORED_{tag}.json")
        assert not w["restore_required"] and w["restored"]["pid"]==live["pid"] and live["config_sha256"]==adopt.NEW and live["API_health"]=="ok"
        windows[tag]={k:w[k] for k in ("started_unix","finished_unix","profile_sha256")}
        windows[tag].update(before=w["initial"]["pid"],after=live["pid"])
    assert sha(study.CHECK)==adopt.CHECKER
    dest=HERE/"FINAL_API.json"
    if not dest.exists():subprocess.run([sys.executable,"-I",str(study.CHECK),str(dest),"--expected",str(study.EXPECTED)],check=True,timeout=90)
    live=read(dest);who=adopt.identity(adopt.NEW);snapshot=adopt.snapshot_health()
    assert who["pid"]==live["pid"]==windows["CAUSAL"]["after"] and live["mapped_runtime_hashes"]==read(HERE/"BEFORE.json")["mapped_runtime_hashes"]
    assert who["decode"]==who["prefill"]==24 and live["gpu_watts_limit"]==150 and live["gpu_ecc"]=="Enabled"
    config=read(study.E/"server-config.json")
    assert all(config["environment"][k]=="64" for k in ("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"))
    first=read(HERE/"runs/chat-A1/health.json")
    assert len(first["initial_host"])==112 and first["initial_host"]==snapshot["host_counters"]
    assert all(first["initial_gpu"][k]==snapshot["gpu"][k] for k in ("ecc_single","ecc_double"))
    exits={};guard=[];watch=[]
    for d in sorted((HERE/"runs").iterdir()):
        h=read(d/"health.json");expected="stopped_by_request" if d.name.startswith("chat-") else "exited"
        assert h["boot_id"]==who["boot"] and h["status"]==expected and h["child_returncode"]==0 and not h.get("error")
        assert h["initial_host"]==first["initial_host"]
        rows=[json.loads(s) for s in (d/"guard-samples.jsonl").read_text().splitlines()]
        g=[v for v in rows if v["source"]=="guard"];w=[v for v in rows if v["source"]=="watch"]
        assert g and w and not any(v["alert"] or v["error"] for v in w)
        guard.extend(g);watch.extend(w)
        exits[d.name]={"pid":h["pid"],"status":expected,"exit":h["child_returncode"],"initial_host_counters_match":True,
                       "guard_samples":len(g),"watch_samples":len(w),"health_sha256":sha(d/"health.json")}
    ids=set()
    for filename in ("both","MX_only"):
        for r in read(HERE/"risk"/(filename+".json"))["grouped"]["original96"]["changed_positions"]:
            ids.update((r["A_top"],r["B_top"],r["target"]))
    labels={str(i):post(adopt.HOST,"/detokenize",{"tokens":[i]})["content"] for i in sorted(ids)}
    if not (HERE/"TOKEN_LABELS.json").exists():fresh(HERE/"TOKEN_LABELS.json",labels)
    assert labels==read(HERE/"TOKEN_LABELS.json")
    assert who==adopt.identity(adopt.NEW)
    fresh(HERE/"FINAL_STATE.json",{"ordinary_profile_sha256":adopt.NEW,"ordinary_pid":who["pid"],"http":live["API_health"],
        "model":who,"services":live["services"],"thresholds":[64,64],"threads":[24,24],"host_counters_unchanged":112,
        "GPU_ECC_delta":{k:snapshot["gpu"][k]-first["initial_gpu"][k] for k in ("ecc_single","ecc_double")},
        "alerts":snapshot["alerts"],"errors":snapshot["errors"],"windows":windows,"diagnostic_processes":exits,
        "guard_samples":len(guard),"watch_samples":len(watch),"peak_C":{k:max(v[k] for v in guard) for k in ("cpu_max_C","gpu_C","hbm_C")},
        "peak_DIMM_C":max(v["max_dimm_C"] for v in watch),"max_fan_pct":max(v["fan_max_pct"] for v in guard),
        "min_available_GiB":min(v["available_kib"] for v in guard)/1024**2})
    print("ordinary API verified",who["pid"],"windows",len(windows),"process exits",len(exits),"detokenized switch IDs",len(labels))


if __name__=="__main__":main()
