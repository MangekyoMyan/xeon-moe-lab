"""Issue the index only after comparing the raw artifacts and the restored ordinary runtime."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from measure import read,sha,dump,aggregate

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"ecc9-simplification-20261119"
JST=timezone(timedelta(hours=9))

def main():
    before=read(ROOT/"production-before/snapshot.json")
    after=read(ROOT/"production-after/snapshot.json")
    b=read(ROOT/"production-before-live.json")
    a=read(ROOT/"production-after-live.json")
    health=read(ROOT/"RESTORED_P12.json")
    states=read(ROOT/"trial/STATE.json")
    assert states["completed"] and all(stage["status"]=="passed" for stage in states["runs"].values())
    assert not read(ROOT/"API_WINDOW_P12.json")["restore_required"]
    assert b["pid"]!=a["pid"]==health["pid"] and b["boot_id"]==a["boot_id"]
    assert a["API_health"]=="ok" and not a["active_alerts"] and not a["monitor_errors"]
    assert all(value=="active" for value in a["services"].values())
    for p in [before,after]:
        assert p["summary"]["all_small_reference_files_match"] and p["summary"]["all_weight_sizes_match"] and not p["summary"]["findings"]
        assert p["process"]["pid_identity_stable_across_collection"]
    assert b["config_sha256"]==a["config_sha256"] and b["mapped_runtime_hashes"]==a["mapped_runtime_hashes"]
    assert b["plan_sha256"]==a["plan_sha256"] and b["model_download_manifest_sha256"]==a["model_download_manifest_sha256"]
    create=lambda **kw:dict(**kw)
    quality=ROOT/"trial/runs/old-p12-quality-01"
    qr=read(quality/"RUN.json")
    plan=read(OLD/"PLAN_MEMBERSHIP.json")
    dump(ROOT/"PLAN_MEMBERSHIP.json",plan)
    registration=qr["registration"]
    assert {k:v for k,v in registration["experts"].items() if v}==plan
    if registration["page_location_errno"]==0:
        assert set(registration["sampled_page_nodes"])=={"0","1"} and sum(registration["sampled_page_nodes"].values())==1440
    dump(ROOT/"HISTORICAL_GOLD_SHA256.json",{k:v["full"] for k,v in read(OLD/"GOLDEN_OUTPUTS.json").items()})
    timestamps=aggregate(ROOT/"trial")
    assert timestamps==read(ROOT/"TIMINGS.json")
    checks={}
    for name,stage in states["runs"].items():
        run=Path(stage["run"]);h=read(run/"health.json")
        assert h["status"]=="exited" and h["child_returncode"]==0 and read(run/"RUN.json")["after_unload_KiB"]["VmLck"]==0
        assert sha(run/"RESULTS_SHA256.json")==stage["manifest_sha256"]
        for n,ch in read(run/"RESULTS_SHA256.json").items():assert sha(run/n)==ch,n
        if "timing" in name:
            for file,receipt in read(run/"TAIL_COMPARISON.json").items():
                f=run/file;raw=f.read_bytes()
                with (quality/file).open("rb") as gold:gold.seek(-len(raw),2);expected=gold.read()
                equal=(raw==expected)
                assert receipt["byte_equal"]==equal
                checks[name+"/"+file]=equal
    assert len(checks)==4
    ready=all(checks.values())
    host=read(ROOT/"QUALITY_HOST_EVIDENCE.json")
    index=read(ROOT/"intake/RESULTS_TEMPLATE.json")
    hist=read(ROOT/"intake/reference/HISTORICAL_CONTROL_BASELINE.json")
    index.update(status="completed" if ready else "completed_with_numeric_mismatch",real_capture_started_at=before["collection"]["start"]["utc"],real_capture_finished_at=after["collection"]["end"]["utc"],
        real_capture_started_jst=datetime.fromisoformat(before["collection"]["start"]["utc"]).astimezone(JST).isoformat(),
        real_capture_finished_jst=datetime.fromisoformat(after["collection"]["end"]["utc"]).astimezone(JST).isoformat(),
        host=b["hostname"],actual_cpu=create(model=b["CPU_model"],sockets=2,physical=12,logical=24),executor="Codex parent: real machine and API, Astra subagent: independent audit",
        source_kit_sha256=read(ROOT/"INTAKE.json")["zip_sha256"],execution_protocol_sha256=sha(ROOT/"intake/BASELINE_PROTOCOL.json"),
        resolved_design_sha256=sha(ROOT/"trial/DESIGN.json"),source_reference_commit="633e8862c9e80f87d937b4a85ba174b7122558be",
        current_baseline_identity=create(config_sha256=b["config_sha256"],normal_start_pid=b["pid"],normal_restored_pid=a["pid"],mapped_runtime_hashes=a["mapped_runtime_hashes"],
            plan_sha256=a["plan_sha256"],model_download_manifest_sha256=a["model_download_manifest_sha256"],runtime="intake/reference/current_runtime_633e.json"),
        deviations_and_reasons=["The before/after collector used its verified 633e local reference and saved only allowlisted runtime fields. The normal stream used the API's configured same-host interface.",
            "Standard Linux perf sampled only the Japanese quality case. The two primary timing processes contain no perf recorder; per-graph worker counts remain unmeasured."],
        actual_decode_team_observed=None,actual_prefill_team_observed=None,
        sampled_worker_evidence=create(decode_distinct_numeric_tids=host["phases"]["decode"]["distinct_tids_sampled_in_cpu_numeric_symbols"],
            prefill_distinct_numeric_tids=host["phases"]["prefill"]["distinct_tids_sampled_in_cpu_numeric_symbols"],scope="Japanese quality case, sampled application instruction pointers. This is not a direct team census."),
        loaded_weight_contents_rehashed_this_capture=False,unknown_fields_and_reasons=[
            {"field":"full current GGUF content equality","reason":"Nine small reference files were hashed and all eight weights' current sizes and stable stats were checked. Re-reading 347,349,300,224 bytes of weights was unnecessary."},
            {"field":"actual team membership of every individual compute graph","reason":"Context getters, process allowance, /proc thread CPU runtime and process-scoped sampled PCs were obtained. They do not enumerate every worker at each graph invocation."},
            {"field":"prefix cache state before the direct request","reason":"No prior private conversations were inspected. The server reported zero cached prompt tokens for this request."},
            {"field":"first reasoning receive time","reason":"No reasoning delta in this particular stream."},
            {"field":"new CPU performance and browser token display time","reason":"This is an old-hardware pre-upgrade capture from a direct client and an internal fixed-token harness."}],
        guard_or_invalid_run_events=[],restored_normal_api=True,baseline_comparison_ready=ready,
        historical_reference={"file":"intake/reference/HISTORICAL_CONTROL_BASELINE.json","date":"2026-11-19","conditions":"old production 12/24, separate independent existing data, not pooled"},
        quality_reference_hashes={k:v["old_p12_sha256"] for k,v in read(ROOT/"trial/QUALITY_COMPARISON.json")["cases"].items()},
        tail_equal=checks,unload_gates="Each of three explicitly released model/context registries and VmLck returned zero; exclusive resource guard before each new launch, normal restore enforced <=128 MiB GPU used before startup.",
        summary_japanese="SUMMARY_JA.txt")
    index["artifacts"]={"before_snapshot":"production-before/snapshot.json","direct_stream":"stream/summary.json","direct_stream_raw":"stream/response_body.bin",
        "topology_and_selected_physical_cores":"TOPOLOGY.json","resolved_conditions":"trial/DESIGN.json","quality_run":"trial/runs/old-p12-quality-01/RUN.json",
        "timing_runs":["trial/runs/old-p12-timing-01/RUN.json","trial/runs/old-p12-timing-02/RUN.json"],"quality_comparisons":"trial/QUALITY_COMPARISON.json",
        "timing_aggregate":"TIMINGS.json","harness_source_and_binary":{"source":"trial/driver.cpp","header":"trial/affinity_check.h","build":"trial/BUILD.json","local_binary":read(ROOT/"FROZEN_ASSETS.json")["driver"]},
        "runtime_exe_dso_model_plan_receipts":["FROZEN_ASSETS.json","production-before-live.json","intake/reference/current_runtime_633e.json","PLAN_MEMBERSHIP.json"],
        "host_observations":"QUALITY_HOST_EVIDENCE.json","restore_evidence":["RESTORED_P12.json","API_WINDOW_P12.json","production-after-live.json"],
        "after_restore_snapshot":"production-after/snapshot.json","independent_review":"audit/FINAL_REVIEW_JA.txt","sha256_manifest":"FILES_SHA256.json"}
    dump(ROOT/"RESULT_INDEX.json",index)
    print("P12 direct comparison ready:",ready)
if __name__=="__main__":main()
