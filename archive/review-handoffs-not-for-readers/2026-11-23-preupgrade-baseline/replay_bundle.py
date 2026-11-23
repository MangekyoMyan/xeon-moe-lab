"""Verify the portable archive and reproduce P12 timings, quality, SSE and host observations."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from measure import aggregate, floats, read, sha
from host_evidence import analyze as host_analyze
from audit.replay_stream import replay as stream_replay

def verify(root):
    manifest=read(root/"FILES_SHA256.json")
    expected=set(manifest)|{"FILES_SHA256.json"}
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert actual==expected,"archive file set"
    for name,entry in manifest.items():
        p=root/name
        assert p.stat().st_size==entry["bytes"] and sha(p)==entry["sha256"],name
    idx=read(root/"RESULT_INDEX.json")
    design=read(root/"trial/DESIGN.json")
    assert idx["execution_protocol_sha256"]==sha(root/"intake/BASELINE_PROTOCOL.json")==design["protocol_sha256"]
    for label in ["before","after"]:
        snap=read(root/f"production-{label}/snapshot.json");server=read(root/f"production-{label}-live.json")
        assert snap["process"]["pid"]==server["pid"] and snap["process"]["pid_identity_stable_across_collection"]
        assert snap["summary"]["all_small_reference_files_match"] and snap["summary"]["all_weight_sizes_match"]
        assert snap["summary"]["all_weight_contents_match"] is None
    assert read(root/"production-before-live.json")["mapped_runtime_hashes"]==read(root/"production-after-live.json")["mapped_runtime_hashes"]
    assert not read(root/"API_WINDOW_P12.json")["restore_required"]
    corpus=read(root/"trial/CASES.json")["cases"]
    history=read(root/"HISTORICAL_GOLD_SHA256.json")
    comp=read(root/"trial/QUALITY_COMPARISON.json")["cases"]
    reference=root/"trial/runs"/design["stages"][0]["name"]
    all_tails=True
    for stage in design["stages"]:
        p=root/"trial/runs"/stage["name"]
        j=read(p/"RUN.json");launch=read(p/"LAUNCH.json");condition=stage["condition"]
        assert j["spec"]==stage==read(p/"SPEC.json") and j["passed"]
        assert j["requested"]==condition and j["linux_entry"]["effective_cpu_ids"]==condition["process_cpu_ids"]
        assert j["context"]["threads"]==j["context"]["threads_batch"]==j["projector_init_threads_argument"]==12
        assert j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==j["after_unload_KiB"]["VmLck"]==0
        assert [r["routed_host"] for r in j["placement"]]==[True]*37+[False]*3
        pin=read(p/"PROVIDER_EXPECTED.json");assert j["providers"]==pin["providers"] and j["active_backend_providers"]==pin["active_backend_providers"]
        assert launch["driver_sha256"]==read(root/"trial/BUILD.json")["driver_sha256"]
        assert {v["path"]:v["sha256"] for v in launch["libraries"].values()}==read(root/"production-before-live.json")["mapped_runtime_hashes"]
        assert read(p/"guard-health.json")["child_returncode"]==0
        for key,value in read(p/"RESULTS_SHA256.json").items():
            if key=="server-config.json":
                data={"command":launch["command"],"environment":launch["environment"]}
                digest=hashlib.sha256((json.dumps(data,indent=2)+"\n").encode()).hexdigest()
            else:digest=sha(p/("guard-health.json" if key=="health.json" else key))
            assert digest==value,(p.name,key)
        for block,ci in zip(j["blocks"],stage["cases"],strict=True):
            c=corpus[ci]
            assert block["prefix_ids"]==c["prefix_ids"] and block["input_tokens"]==c["used_T1_ids"] and block["input_tokens"]==c["next_targets_prefix_and64T1"][:64]
            assert block["positions_begin"]==c["prefix_count"] and len(block["start_ns"])==64 and block["status"]==[0]*64
            f=p/block["file"];raw=f.read_bytes()
            assert len(raw)==stage["rows"]*129280*4 and all(map(math.isfinite,floats(raw)))
            if stage["stage"]=="quality":
                baseline=comp[f.name]
                assert sha(f)==baseline["old_p12_sha256"] and history[f.name]==baseline["historical_sha256"]
                assert (sha(f)==history[f.name])==baseline["byte_equal"]
                if baseline["byte_equal"]:assert baseline["argmax_changes"]==0 and baseline["max_abs_error"]==0 and baseline["nmse"]==0
                reg=j["registration"]
                assert {k:v for k,v in reg["experts"].items() if v}==read(root/"PLAN_MEMBERSHIP.json")
                assert reg["sampled_vma_policies"]=={"interleave:0-1":1440}
                if reg["page_location_errno"]==0:assert set(reg["sampled_page_nodes"])=={"0","1"} and sum(reg["sampled_page_nodes"].values())==1440
            else:
                with (reference/f.name).open("rb") as earlier:
                    earlier.seek(-len(raw),2);tail=earlier.read()
                equal=(raw==tail);record=read(p/"TAIL_COMPARISON.json")[f.name]
                assert equal==record["byte_equal"] and sha(f)==record["observed_sha256"]
                all_tails=all_tails and equal
    assert aggregate(root/"trial")==read(root/"TIMINGS.json")
    assert json.loads(json.dumps(host_analyze(root)))==read(root/"QUALITY_HOST_EVIDENCE.json")
    assert stream_replay(root/"stream")==read(root/"audit/STREAM_REPLAY.json")
    assert idx["baseline_comparison_ready"]==all_tails
    return {"passed":True,"manifest_entries":len(manifest),"model_processes":3,"timed_processes":2,"timed_T1_calls":256,"full_quality_rows":130,"tail_rows":4,"stream_requests_replayed":1,"any_model_or_network_execution":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
