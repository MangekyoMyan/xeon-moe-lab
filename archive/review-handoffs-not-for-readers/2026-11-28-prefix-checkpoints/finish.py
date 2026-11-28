"""Verify all maintenance windows and the second physical-ring numerical control."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
F=R/"checkout/current/experiments/state-branches"
def js(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
def sha(p):
    with p.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def rings():
    rows={}
    for run in (R/"runs").glob("ring-*/trace/calls.jsonl"):
        for s in run.read_text().splitlines():
            v=json.loads(s)
            if "logits_file" in v:rows[v["phase"],v["pos"][-1]+1]=(v,run.parent/v["logits_file"])
    comparisons=[]
    for (name,end),(row,p) in sorted(rows.items()):
        if not (name.startswith("ring-return-") or name in ("empty-suffix","ordinary-return")):continue
        control="ordinary-control" if name=="ordinary-return" else "ring-control"
        base,bpath=rows[control,end]
        v=dict(name=name,control=control,end=end,sha256=sha(p),exact=sha(p)==sha(bpath),same_ids=row["input_ids"]==base["input_ids"],
               same_backend=row["ubatches"]==base["ubatches"],evaluated=len(row["input_ids"]))
        assert v["exact"] and v["same_ids"] and v["same_backend"],v
        comparisons.append(v)
    assert len(comparisons)==10
    fix=js(R/"ring-evidence/fixed127.json")
    n=fix["first"]
    assert [e["position"] for e in n["checkpoint"]["events"] if e["status"]=="saved"]==[127,254,381]
    empty=js(R/"ring-evidence/empty-suffix.json")["resume"]
    assert empty["from_position"]==1535 and empty["old_replayed"]==1 and empty["new_input"]==0
    return dict(comparisons=comparisons,geometry=js(R/"ring-evidence/geometry.json"),empty_suffix=empty,
                cancelled=js(R/"ring-evidence/cancel.json")["fence"],
                recovered=js(R/"ring-evidence/recover.json")["mode"],fixed127_processed=n["processed"],
                fixed127_mark=fix["mark"]["snapshot"]["processed"])
def service():
    windows=[]; guards=[]
    for root in (R,R/"layout",R/"entry"):
        for path in sorted(root.glob("WINDOW_*.json")):
            v=js(path)
            restored=js(path.with_name(path.name.replace("WINDOW_","RESTORED_")))
            assert v["restore_required"] is False and restored["API_health"]=="ok" and restored["slots_idle"]
            assert restored["config_sha256"]==js(R/"BEFORE.json")["config_sha256"]
            assert restored["mapped_runtime_hashes"]==js(R/"BEFORE.json")["mapped_runtime_hashes"]
            windows.append(dict(label=path.name[7:-5],started=v["started_unix"],ended=v["finished_unix"],
                         normal_pid=restored["pid"],monitor_errors=restored["monitor_errors"],alerts=restored["active_alerts"]))
        for path in sorted((root/"runs").glob("*/READY.json")):
            h=js(path.parent/"health.json");stop=js(path.parent/"STOP.json");ready=js(path)
            assert stop["status"]=="stopped_by_request" and stop["returncode"]==stop["child_returncode"]==0
            assert h["limits"]["fan_percent"]<=70 and ready["identity"]["threads"]==[24,24]
            gpu=h["initial_gpu"];samples=h["recent_samples"]
            guards.append(dict(run=path.parent.name,pid=ready["pid"],identity=ready["identity"],
                  observed=ready["observed"],stop=stop,samples_kept=len(samples),
                  gpu_watts_limit=gpu["limit_W"],gpu_ecc=gpu["ecc"],fan_limit=h["limits"]["fan_percent"],
                  minimum_logged_available_GiB=min(r["available_kib"] for r in samples)/1024**2,
                  gpu_ecc_delta=[samples[-1]["gpu"][k]-gpu[k] for k in ("ecc_single","ecc_double")],
                  monitor_timeout_count=h["gpu_monitor"]["timeout_count"]))
    windows.sort(key=lambda w:w["ended"])
    return dict(normal_profile=js(R/"BEFORE.json")["config_sha256"],initial_pid=js(R/"BEFORE.json")["pid"],
                final_normal_pid=windows[-1]["normal_pid"],ordinary_libraries=js(R/"BEFORE.json")["mapped_runtime_hashes"],
                windows=windows,private_guards=guards)
def code():
    base="09798ce2cc7cb41d879813962377f6d1679e76e3";timed="1823084b0590e70dea12548137656366aa7ae808"
    source={p.name:sha(p) for p in F.iterdir() if p.is_file()}
    live=js(R/"workflow-evidence/implementation.json")
    assert js(R/"build-release/BUILD.json")["patch_sha256"]==sha(F/"native-stream.patch")
    assert js(R/"build-release/BUILD.json")["prefix_capture_sha256"]==sha(F/"prefix-capture.hpp")
    assert js(R/"build-release/physical/BUILD.json")["patch_sha256"]==sha(F/"physical-layout.patch")
    return dict(base=base,timed_engine_source=timed,final_source_commit=subprocess.check_output(["git","-C",str(F),"rev-parse","HEAD"],text=True).strip(),
              file_sha256=source,as_measured=live,server_adapter_sha256=js(R/"build-release/BUILD.json")["library_sha256"],
              physical_model_library_sha256=js(R/"build-release/physical/BUILD.json")["library_sha256"],
              original_model_library_sha256=js(R/"build-release/physical/BUILD.json")["base_sha256"])
if __name__=="__main__":
    write(R/"RING.json",rings());write(R/"SERVICE.json",service());write(R/"CODE.json",code())

