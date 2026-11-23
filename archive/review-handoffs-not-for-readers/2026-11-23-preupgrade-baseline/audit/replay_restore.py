"""Check the saved ordinary service identities on both sides of the guarded window."""

import json
from pathlib import Path


def require(test, message):
    if not test:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def observe(root, label):
    snap = load(root / label / "snapshot.json")
    live = load(root / (label + "-live.json"))
    process = snap["process"]
    require(process["pid"] == live["pid"] and process["pid_identity_stable_across_collection"], "process identity")
    exe = process["executable"]
    require(exe["matches_expected"] and exe["stable_during_read"], "running executable")
    files = {row["reference_key"]: row for row in snap["files"]["small"]}
    require(len(files) == 9 and all(r["matches_expected"] for r in files.values()), "small reference fingerprints")
    mapped = {r["reference_library"]: r for r in process["mapped_dsos"] if "reference_library" in r}
    require(set(mapped) == {"cpu", "llama", "base", "cuda"} and
            all(row["candidates"] == 1 and row["matches"] for row in process["profile_library_comparisons"]),
            "four libraries in process maps")
    for name, dso in mapped.items():
        fp = dso["file_fingerprint"]
        require(fp["matches_mapped_device_inode"] and fp["matches_expected"] and fp["sha256"] == files["libraries." + name]["sha256"]
                and live["mapped_runtime_hashes"][dso["path"]] == fp["sha256"], "mapped backing file identity")
    weights = snap["files"]["weights"]
    require(len(weights) == 8 and all(w["size_matches_expected"] and w["stable_during_read"] and
            not w["contents_rehashed"] and w["content_matches_expected"] is None for w in weights), "eight weight stats")
    opts = process["command"]["performance_options"]
    require((opts["-t"], opts["-tb"], int(opts["--cpu-mask"], 16), int(opts["--cpu-mask-batch"], 16)) ==
            ("12", "24", 0xffffff, 0xffffff), "ordinary CPU arguments")
    require(opts["--cpu-strict"] == opts["--cpu-strict-batch"] == "1", "ordinary strict masks")
    require(set(live["services"]) == {"llm-deepseek41-api.service", "xeon-watch.service", "xeon-watch-notify.service"},
            "service inventory")
    require(live["API_health"] == "ok" and live["command_and_configured_environment_match"] and
            live["backing_inodes_verified"] and all(state == "active" for state in live["services"].values()) and
            not live["active_alerts"] and not live["monitor_errors"], "health, services and monitor")
    return {
        "pid": live["pid"], "boot_id": live["boot_id"], "process_start_ticks": process["stat_before"]["starttime_ticks"],
        "executable": exe["sha256"], "small_file_sha256": {k: f["sha256"] for k, f in files.items()},
        "weight_stats": {w["reference_file"]: w["stat_after"] for w in weights},
        "environment": process["performance_environment"]["values"], "options": opts, "services": live["services"],
    }


def replay(root):
    before, after = (observe(root, name) for name in ("production-before", "production-after"))
    require(before["boot_id"] == after["boot_id"] and before["pid"] != after["pid"] and
            before["process_start_ticks"] < after["process_start_ticks"], "new server process on same boot")
    for field in ("executable", "small_file_sha256", "weight_stats", "environment", "options", "services"):
        require(before[field] == after[field], field + " changed")
    receipt = load(root / "RESTORED_P12.json")
    require(receipt["pid"] == after["pid"] and receipt["normal_API_restored"] is True and
            receipt["config_sha256"] == after["small_file_sha256"]["service_config"], "restore receipt")
    window = load(root / "API_WINDOW_P12.json")
    require(window["initially_active"] is True and window["restore_required"] is False, "window closure")
    return {
        "before_pid": before["pid"], "after_pid": after["pid"], "same_boot": True,
        "newer_process_start_verified": True, "ordinary_threads": {"decode": 12, "prefill": 24},
        "ordinary_strict_masks": ["0xffffff", "0xffffff"],
        "four_mapped_library_device_inode_hash_checks_each_snapshot": True,
        "small_file_sha256": after["small_file_sha256"], "weight_metadata_agree_across_window": 8,
        "weight_content_rehashed": False, "selected_runtime_arguments_and_environment_agree": True,
        "services_after": after["services"], "window_restore_obligation_closed": True,
    }


if __name__ == "__main__":
    print(json.dumps(replay(Path(__file__).resolve().parents[1]), indent=2))
