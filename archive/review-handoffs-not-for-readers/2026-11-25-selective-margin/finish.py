"""Freeze the final ordinary API and guard state, without changing any configuration."""
import json
from pathlib import Path
import subprocess
import sys
from prepare import ROOT,LAB,load,save,sha
sys.path.insert(0,str(LAB/"adopt24-20261124"));import adopt

def main():
    w=load(ROOT/"WINDOW_MARGIN.json");rest=load(ROOT/"RESTORED_MARGIN.json")
    assert not w["restore_required"] and w["restored"]["pid"]==rest["pid"]
    c=LAB/"results-review-20261124/tools/check_live_identity.py";exp=LAB/"results-review-20261124/reference/EXPECTED_NORMAL_24_24.json"
    assert sha(c)==adopt.CHECKER
    p=ROOT/"FINAL_API.json"
    if not p.exists():subprocess.run([sys.executable,"-I",str(c),str(p),"--expected",str(exp)],check=True,timeout=90)
    before=load(ROOT/"BEFORE.json");now=load(p);id=adopt.identity(adopt.NEW);snap=adopt.snapshot_health()
    assert now["pid"]==rest["pid"]==id["pid"] and now["API_health"]=="ok" and now["mapped_runtime_hashes"]==before["mapped_runtime_hashes"]
    assert id["decode"]==id["prefill"]==24 and now["gpu_watts_limit"]==150 and now["gpu_ecc"]=="Enabled"
    env=load(adopt.CONFIG)["environment"];assert all(env[k]=="64" for k in ("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"))
    assert not now["monitor_errors"] and not now["active_alerts"] and all(x=="active" for x in now["services"].values())
    gs=[];ws=[];exits={};h0=load(ROOT/"runs/F0/health.json")
    assert h0["initial_host"]==snap["host_counters"] and len(snap["host_counters"])==112
    assert all(h0["initial_gpu"][k]==snap["gpu"][k] for k in ("ecc_single","ecc_double"))
    spans={};tests=0
    for label in ("F0","candidate","F1"):
        d=ROOT/"runs"/label;h=load(d/"health.json");verified=load(d/"VERIFIED.json")
        assert h["status"]=="exited" and h["child_returncode"]==0 and not h.get("error") and h["pid"]==verified["pid"] and h["initial_host"]==h0["initial_host"]
        samples=[json.loads(s) for s in (d/"guard-samples.jsonl").read_text().splitlines()]
        gs.extend(v for v in samples if v["source"]=="guard");ws.extend(v for v in samples if v["source"]=="watch")
        rows=[json.loads(s) for s in (d/"measurements.jsonl").read_text().splitlines()]
        spans[label]=[min(v["start_ns"] for v in rows),max(v["ready_ns"] for v in rows)];tests+=len(rows)
        exits[label]={"pid":h["pid"],"returncode":h["child_returncode"],"measurements":len(rows),"guard_sha256":sha(d/"health.json")}
    assert tests==70 and not any(v["alert"] or v["error"] for v in ws)
    off=[load(ROOT/n) for n in ("OFFLINE_BUILD.json","OFFLINE_QUALITY_INTERVAL.json")]
    assert all(not (a<q["end_ns"] and q["start_ns"]<b) for q in off for a,b in spans.values())
    assert load(ROOT/"SAMPLER_INPUTS.json")["clock_begin_ns"]>max(z for _,z in spans.values())
    assert load(ROOT/"ROUTER_MICRO.json")["API_pid"]==id["pid"]
    save(ROOT/"FINAL_STATE.json",{"normal_config_sha256":adopt.NEW,"prior_pid":before["pid"],"ordinary_pid":id["pid"],"api":"ok",
        "ordinary_threads":[24,24],"ordinary_prefill_thresholds":[64,64],"experimental_margin_enabled_in_ordinary":False,
        "normal_libraries":now["mapped_runtime_hashes"],"guard_passed":True,"closed_window":True,"exits":exits,
        "model_prefix_measurements":tests,"services":now["services"],"host_counters_unchanged":112,"GPU_ECC_delta":[0,0],
        "min_available_GiB":min(v["available_kib"] for v in gs)/1024**2,
        "peaks":{"CPU_C":max(v["cpu_max_C"] for v in gs),"GPU_C":max(v["gpu_C"] for v in gs),"HBM_C":max(v["hbm_C"] for v in gs),
                 "DIMM_C":max(v["max_dimm_C"] for v in ws),"fan_pct":max(v["fan_max_pct"] for v in gs)},
        "guard_records":len(gs),"watch_records":len(ws),"inference_windows_exclude_offline_benchmarks":True})
    print("Normal service verified",id["pid"],"three clean exits",tests,"prefixes")

if __name__=="__main__":main()
