"""Recalculate the short signed-off scope from the archived observations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check(root):
    manifest=read(root/"MANIFEST_SHA256.json")["files"]
    for r in manifest:
        p=root/r["path"]
        assert p.stat().st_size==r["bytes"] and sha(p)==r["sha256"],r["path"]
    rec=read(root/"ADOPTION_RECORD.json");b=read(root/"BEFORE.json");a=read(root/"AFTER.json");f=read(root/"FINAL_LIVE.json")
    pre=read(root/"reference/EXPECTED_NORMAL_12_24.json");post=read(root/"reference/EXPECTED_NORMAL_24_24.json")
    assert rec["status"]=="adopted" and rec["original_sha256"]==b["config_sha256"]==pre["config_sha256"]==read(root/"BACKUP.json")["sha256"]
    assert rec["actual_after_sha256"]==rec["prepared_sha256"]==a["config_sha256"]==f["config_sha256"]==post["config_sha256"]
    assert b["mapped_runtime_hashes"]==a["mapped_runtime_hashes"]==f["mapped_runtime_hashes"]==pre["mapped_runtime_hashes"]==post["mapped_runtime_hashes"]
    assert rec["old_model"]["decode"]==12 and rec["old_model"]["prefill"]==24 and rec["old_model"]["pid"]==b["pid"]
    now=rec["normal_model"]
    assert (now["decode"],now["prefill"])==(24,24) and now["pid"]==a["pid"]==f["pid"]!=b["pid"]
    assert now["listen_inode"] and now["ppid"]==now["supervisor"] and now["boot"]==rec["old_model"]["boot"]
    assert a["backing_inodes_verified"] and f["command_and_configured_environment_match"] and f["API_health"]=="ok"
    assert rec["only_semantic_difference"]=={"field":"command[9]","flag":"-t","before":"12","after":"24"}
    assert read(root/"CURRENT_VERIFY_LIVE.json")["verified_api_pid"]==now["pid"]
    assert sha(root/"current/runtime.json")==read(root/"CURRENT_ENTRY.json")["current_runtime_sha256"]
    current=read(root/"current/runtime.json")
    assert current["service_config"]["sha256"]==f["config_sha256"] and current["hardware"]["physical"]==24 and current["hardware"]["logical"]==48
    stream=root/"stream";r=json.loads(subprocess.check_output([sys.executable,"-I",str(root/"tools/replay_stream.py"),str(stream)]))
    assert r==read(root/"STREAM_REPLAY.json")
    ack=rec["request"]
    assert ack==read(root/"ACCEPTANCE.json") and r["first_content_receive_ms"]==ack["ttft_content_ms"]
    assert r["server_usage"]==ack["reported_usage"] and r["finish_reason"]==ack["finish_reason"]=="stop"
    assert sha(stream/"request.effective.json")==sha(root/"reference/normal_stream_ja.json")==ack["input_request_hash"]
    assert sha(stream/"content.txt")==sha(root/"reference/pilot-content.txt")==ack["content_sha256"]
    timing=[json.loads(l).get("json") for l in (stream/"events.jsonl").read_text().splitlines()]
    assert [v["timings"] for v in timing if isinstance(v,dict) and "timings" in v]==[ack["server_timings"]]
    health=rec["health"];g=[json.loads(l) for l in (root/"GUARD_SAMPLES.jsonl").read_text().splitlines()]
    w=[json.loads(l) for l in (root/"WATCH_SAMPLES.jsonl").read_text().splitlines()]
    assert len(g)==health["native_sample_count"] and len(w)==health["watch_sample_count"]
    assert all(v["pid"]==now["pid"] and v["ecc"]=="Enabled" and v["limit_W"]==150 for v in g)
    for field,column in (("max_CPU_C","cpu_max_C"),("max_GPU_C","gpu_C"),("max_HBM_C","hbm_C")):
        assert max(v[column] for v in g)==health[field]
    assert min(v["available_kib"] for v in g)/1024**2==health["min_available_GiB"]
    assert health["max_socket_C"]=={k:max(v["packages_C"][k] for v in g) for k in g[0]["packages_C"]}
    assert max(max(v["fan_percent"]) for v in g)==health["max_fan_percent"] and max(v["dimm_max_C"] for v in w)==health["max_DIMM_C"]
    assert all(not (s["counter_diff"] or s["alert"] or s["errors"]) for s in w)
    assert read(root/"HEALTH_BEFORE.json")["host_counters"]==read(root/"HEALTH_FINAL.json")["host_counters"]
    assert read(root/"KERNEL_CHECK.json")["hardware_error_match_count"]==health["kernel_error_matches"]==0
    print(f"Verified {len(manifest)} payloads, ordinary 24/24 identity, SSE response and {len(g)} native guard samples")


if __name__=="__main__":check(Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve())
