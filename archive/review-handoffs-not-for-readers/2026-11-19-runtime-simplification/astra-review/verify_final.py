"""Independent offline reading of original clocks, output arrays and cache tables."""
from array import array
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

study = Path(__file__).resolve().parents[1]
trial = study / "trial"
prior = study.parent / "169a-exploration-20261119"
cpu_name = "libggml-cpu.so.0.28.0"

def load(file):
    return json.loads(file.read_text())

def sha(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def summarize(x):
    a0, b0, b1, a1 = map(Fraction, x)
    return {"four_values": list(map(float, x)),
            "paired_delta": [float(b0 - a0), float(b1 - a1)],
            "paired_percent": [float(100 * (b0 / a0 - 1)), float(100 * (b1 / a1 - 1))],
            "change_percent": float(100 * ((b0 + b1) / (a0 + a1) - 1)),
            "gap_min_B_minus_max_A": float(min(b0, b1) - max(a0, a1)),
            "A_mean": float((a0 + a1) / 2), "B_mean": float((b0 + b1) / 2)}

def main():
    cases = load(trial / "CASES.json")["cases"]
    original = load(prior / "BASELINE_REVERIFIED.json")["outputs"]
    pins = load(study / "INITIAL_RUNTIME.json")
    build = load(study / "build/BUILD.json")
    plan = study / "current/models/deepseek41-mix-256g/frequency19200.plan"
    assert sha(plan) == pins["plan_sha256"]
    lines = [r.split() for r in plan.read_text().splitlines()]
    assert lines[0] == ["XEON_IQ_CACHE_PLAN_V2", "80", "19200", "127401984000"]
    expected = {r[0]: list(map(int, r[6:])) for r in lines[1:]}
    assert len(expected) == 80 and sum(map(len, expected.values())) == 19200
    snapshots = {}
    for c in cases:
        n = c["case_id"] + ".f32"
        src = prior / "main/runs/baseline" / n
        assert sha(src) == original[n]
        snapshots[n] = src.read_bytes()
        floats = array("f", snapshots[n])
        if sys.byteorder != "little":
            floats.byteswap()
        assert len(floats) == 65 * 129280 and all(map(math.isfinite, floats))
    # Byte equality to the individually checked finite arrays transfers finiteness.
    design = trial / "DESIGN.json"
    stages = load(design)["stages"]
    assert [s["arm"] for s in stages] == ["control", "ordinary", "control", "ordinary", "ordinary", "control"]
    results = []
    clocks = {c["case_id"]: {"decode_ms": [], "prefill_ms": []} for c in cases}
    quality_nodes = {}
    phases = {"load_s": [], "vision_s": [], "unload_s": [], "loaded_RSS_GiB": [], "peak_RSS_GiB": []}
    saw_pids = set()
    reported = load(study / "ANALYSIS.json")
    token_clocks = 0
    files = 0
    state = load(trial / "STATE.json")
    for stage in stages:
        folder = trial / "runs" / stage["name"]
        rec = load(folder / "RUN.json")
        launch = load(folder / "LAUNCH.json")
        health = load(folder / "health.json")
        expected_providers = load(folder / "PROVIDER_EXPECTED.json")
        arm = stage["arm"]
        quality = stage["stage"] == "quality"
        spec = load(folder / "SPEC.json")
        manifest = folder / "RESULTS_SHA256.json"
        assert sha(manifest) == state["runs"][stage["name"]]["manifest_sha256"]
        for name, digest in load(manifest).items():
            assert sha(folder / name) == digest
            files += 1
        assert state["runs"][stage["name"]]["status"] == "passed" and rec["passed"]
        assert sha(design) == launch["design_sha256"] and sha(trial / "CASES.json") == launch["case_file_sha256"]
        assert rec["spec"] == launch["spec"] == spec == stage
        assert launch["plan_sha256"] == pins["plan_sha256"] and launch["initial_config_sha256"] == pins["config_sha256"]
        assert all(v not in launch["environment"] for v in ("LD_PRELOAD", "GGML_CPU_IQ_ROW_LOCAL", "GGML_CPU_IQ_AUDIT_DIR"))
        assert health["status"] == "exited" and health["child_returncode"] == 0 and health["pid"] not in saw_pids
        saw_pids.add(health["pid"])
        libs = launch["libraries"]
        assert libs[cpu_name]["sha256"] == build[arm]["sha256"] and libs[cpu_name]["path"] == build[arm]["file"]
        for name, v in libs.items():
            if name != cpu_name:
                assert v["sha256"] == pins["mapped_runtime_hashes"][v["path"]]
        assert rec["providers"] == expected_providers["providers"] and rec["active_backend_providers"] == expected_providers["active_backend_providers"]
        for x in rec["providers"]:
            symbol, path = x["symbol"], x["path"]
            name = "libllama.so.0.8.0" if symbol.startswith("llama_") else "libggml-base.so.0.28.0" if symbol.startswith("ggml_backend_sched_") else cpu_name
            assert path == libs[name]["path"]
        assert rec["active_backend_providers"] == [
            {"name": "CUDA0", "graph_compute_provider": libs["libggml-cuda.so.0.28.0"]["path"]},
            {"name": "CPU", "graph_compute_provider": libs[cpu_name]["path"]}]
        assert rec["vision_loaded"] and [p["routed_host"] for p in rec["placement"]] == [True] * 37 + [False] * 3
        assert len(rec["shared_inventory"]) == 37 and rec["compact_bytes"] == 127401984000
        assert rec["compact_bytes_after_unload"] == rec["after_unload_KiB"]["VmLck"] == 0
        if quality:
            r = rec["registration"]
            assert len(r["experts"]) == 120 and {k:v for k,v in r["experts"].items() if v} == expected
            assert r["selected_projections"] == 19200 and r["registered_tensors"] == 80
            assert r["sampled_pages"] == 1440 and r["sampled_vma_policies"] == {"interleave:0-1": 1440}
            assert r["page_location_errno"] == 0 and r["sampled_page_nodes"] == {"0": 480, "1": 960}
            quality_nodes[arm] = r["sampled_page_nodes"]
        else:
            assert "registration" not in rec
            for key, field in [("load_s", "text_model_context_load_ns"), ("vision_s", "vision_init_ns"), ("unload_s", "unload_ns")]:
                phases[key].append(Fraction(rec[field], 10**9))
            phases["loaded_RSS_GiB"].append(Fraction(rec["loaded_text_memory_KiB"]["VmRSS"], 1024**2))
            phases["peak_RSS_GiB"].append(Fraction(rec["before_unload_KiB"]["VmHWM"], 1024**2))
        for b, idx in zip(rec["blocks"], stage["cases"], strict=True):
            case = cases[idx]
            assert b["prefix_ids"] == case["prefix_ids"] and b["input_tokens"] == case["used_T1_ids"]
            assert b["case_id"] == case["case_id"] and b["positions_begin"] == case["prefix_count"]
            assert b["status"] == [0] * 64 and b["traced_calls"] == 0
            name = case["case_id"] + ".f32"
            assert b["file"] == name and b["rows"] == (65 if quality else 1)
            ref = snapshots[name] if quality else snapshots[name][-129280 * 4:]
            assert (folder / name).read_bytes() == ref
            if not quality:
                delta = [e - s for s,e in zip(b["start_ns"], b["output_ready_ns"], strict=True)]
                assert len(delta) == 64 and min(delta) > 0
                assert all(s > e for s,e in zip(b["start_ns"][1:], b["output_ready_ns"][:-1], strict=True))
                token_clocks += len(delta)
                clocks[case["case_id"]]["decode_ms"].append(Fraction(sum(delta), 64 * 10**6))
                clocks[case["case_id"]]["prefill_ms"].append(Fraction(b["prefill_ready_ns"] - b["prefill_begin_ns"], 10**6))
        results.append({"stage": stage["name"], "pid": health["pid"], "after_unload_RSS_KiB": rec["after_unload_KiB"]["VmRSS"]})
    summary = {name: {k: summarize(v) for k,v in clock.items()} for name, clock in clocks.items()}
    assert token_clocks == 512 and len(saw_pids) == 6
    for key, case in summary.items():
        for raw, compare in [("decode_ms","mean_ms"), ("prefill_ms","prefill_ms")]:
            expected_pct = reported["cases"][key][compare]["change_percent"]
            assert abs(case[raw]["change_percent"] - expected_pct) < 1e-11
        assert case["decode_ms"]["gap_min_B_minus_max_A"] > 0 and min(case["decode_ms"]["paired_delta"]) > 0
    decision = load(study / "DECISION.json")
    restoration = load(study / "RESTORED_ROW_CLEANUP.json")
    window = load(study / "API_WINDOW_ROW_CLEANUP.json")
    profile = load(study / "current/runtime.json")
    assert state["completed"] and not window["restore_required"] and restoration["normal_API_restored"]
    assert restoration["API_health"] == "ok" and restoration["normal_runtime_backing_files_verified"]
    assert restoration["mapped_runtime_hashes"] == pins["mapped_runtime_hashes"]
    assert restoration["config_sha256"] == pins["config_sha256"] and all(s == "active" for s in restoration["services"].values())
    assert not restoration["active_alerts"] and not restoration["monitor_errors"]
    assert decision["decision"] == "hold" and not decision["normal_config_changed"] and not decision["service_switch_to_candidate"]
    assert decision["compiled_isolated_CPU"] == build["ordinary"]["sha256"]
    assert decision["current_normal_CPU"] == build["control"]["sha256"] == profile["libraries"]["cpu"]["sha256"]
    assert profile["row_local"]["normal_build_contains_specialized_placement"] and not profile["row_local"]["normal_enabled"]
    return {"fresh_processes": results, "run_manifest_entries_hashed": files, "quality_rows_compared": 260,
            "timing_final_rows_compared": 8, "F32_columns": 129280, "finite_saved_F32_compared": 268 * 129280,
            "timing_calls": token_clocks, "cases": summary, "phases": {k: summarize(v) for k,v in phases.items()},
            "quality_page_samples": quality_nodes, "rule_result": "retain previous CPU33a",
            "restoration_receipt_pid": restoration["pid"], "deployed_sha256": decision["current_normal_CPU"],
            "isolated_candidate_sha256": decision["compiled_isolated_CPU"], "pass": True}

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
