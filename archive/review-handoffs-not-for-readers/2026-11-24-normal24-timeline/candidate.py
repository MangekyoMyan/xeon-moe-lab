"""Write a portable two-setting candidate and an uninstalled local profile for the measured threshold."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KEYS = ("GGML_OP_OFFLOAD_MIN_BATCH", "GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")


def main():
    state = json.loads((ROOT / "FINAL_STATE.json").read_bytes())
    result = json.loads((ROOT / "PREFIX_QUALITY.json").read_bytes())
    config = ROOT.parent / "deepseek41-bringup-20261101/server-config.json"
    original = config.read_bytes()
    assert hashlib.sha256(original).hexdigest() == state["normal_profile"]
    assert result["ran"] and (result["normal"], result["candidate"]) == (64, 128)
    base = json.loads(original)
    changed = json.loads(original)
    patch = []
    for key in KEYS:
        assert base["environment"][key] == "64"
        path = "/environment/" + key
        patch.extend([{"op": "test", "path": path, "value": "64"},
                      {"op": "replace", "path": path, "value": "128"}])
        changed["environment"][key] = "128"
    output = (json.dumps(changed, ensure_ascii=False, indent=2) + "\n").encode()
    local = ROOT / "private/normal24-prefill128.json"
    with local.open("xb") as handle:
        local.chmod(0o600)
        handle.write(output)
    check = json.loads(local.read_bytes())
    for key in KEYS:
        assert check["environment"][key] == "128"
        check["environment"][key] = "64"
    assert check == base and config.read_bytes() == original
    record = {"state": "measured candidate, not installed", "normal_config_sha256": state["normal_profile"],
              "candidate_config_sha256": hashlib.sha256(output).hexdigest(), "patch": patch,
              "decode_threads": 24, "prefill_threads": 24, "resident_experts": "normal",
              "native_latency_raw": "PREFILL.json", "full_logits_raw": "PREFIX_QUALITY.json",
              "quality_scope": "Two input families. 96-token tests change 2/65 and 1/65 top-ranked IDs on identical histories."}
    (ROOT / "PREFILL_CANDIDATE.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
