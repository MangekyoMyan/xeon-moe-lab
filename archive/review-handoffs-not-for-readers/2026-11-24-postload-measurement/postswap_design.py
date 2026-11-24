"""Resolve both fixed-token conditions from this boot's allowed physical cores."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "preupgrade-baseline-20261123"
CPU_MODEL = "E5-2697 v2"
PROTOCOL_SHA = "ed608ddbb6d4de23341605ab512fcedf00346b53dd76dc6de5d7b5fdb2f47e07"
CASES_SHA = "0458b7a358d0eb3a0c289eb1a1aa9903ee0bb972158acc64abbf0952f472d746"
CONFIG_SHA = "963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43"
DRIVER_SHA = "b32c3777afd478e0a726f898795c3bcbe14005c69fd4f116083c87520f646994"
QUALITY_SHA = {
    "text_cache_budget_ja.f32": "35d028fed9fa08bbe8f304c598ced362ff466aff176ecf407133c0a1870bbd64",
    "code_stable_grouping_python.f32": "077eefd33928bfbec340162c5b5e54c2bbad9e4ac3ae72bc23f38a7a33d69d01",
}
SLUGS = {"NEW_P12": "new-p12", "NEW_D24_B12": "new-d24-b12"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def require(test, why):
    if not test:
        raise ValueError(why)


def live_topology():
    models = set()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model name":
            models.add(value.strip())
    require(len(models) == 1 and CPU_MODEL in next(iter(models)), "New CPU model is not uniform/expected")
    rows = []
    for cpu in sorted(os.sched_getaffinity(0)):
        folder = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        nodes = [int(node.name[4:]) for node in folder.glob("node[0-9]*")]
        require(len(nodes) == 1, f"Unresolved NUMA node for CPU {cpu}")
        topology = folder / "topology"
        rows.append({
            "cpu": cpu, "socket": int((topology / "physical_package_id").read_text()),
            "core": int((topology / "core_id").read_text()), "node": nodes[0],
            "siblings": (topology / "thread_siblings_list").read_text().strip(),
        })
    return {
        "cpu": next(iter(models)), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "allowed": rows,
    }


def conditions(topology):
    require(CPU_MODEL in topology["cpu"], "Expected E5-2697 v2")
    cores = defaultdict(lambda: defaultdict(list))
    socket_nodes = defaultdict(set)
    seen = set()
    for row in topology["allowed"]:
        cpu, socket, core, node = (row[k] for k in ("cpu", "socket", "core", "node"))
        require(cpu >= 0 and cpu not in seen, "Duplicate or invalid CPU")
        seen.add(cpu)
        cores[socket][core].append(cpu)
        socket_nodes[socket].add(node)
    require(len(cores) == 2 and all(len(group) == 12 for group in cores.values()),
            "Need 12 distinct available physical cores on each socket")
    require(all(len(v) == 1 for v in socket_nodes.values()) and
            set.union(*socket_nodes.values()) == {0, 1}, "Need one distinct NUMA node per socket, nodes 0 and 1")
    sockets = [[min(group[core]) for core in sorted(group)] for _, group in sorted(cores.items())]
    p12 = sorted(cpu for group in sockets for cpu in group[:6])
    p24 = sorted(cpu for group in sockets for cpu in group)
    mask = lambda ids: hex(sum(1 << cpu for cpu in ids))
    return {
        name: {
            "name": name, "decode_threads": len(decode), "prefill_threads": 12,
            "projector_threads": 12, "decode_cpu_ids": decode, "prefill_cpu_ids": p12,
            "process_cpu_ids": decode, "decode_mask": mask(decode), "prefill_mask": mask(p12),
            "strict": True, "interleave_nodes": [0, 1], "OMP_THREAD_LIMIT": 24, "packing_threads_at_load": 12,
        }
        for name, decode in (("NEW_P12", p12), ("NEW_D24_B12", p24))
    }


def design(topology):
    stage_rows = []
    for condition in conditions(topology).values():
        slug = SLUGS[condition["name"]]
        for label, stage, order in (("quality-01", "quality", [0, 1]), ("timing-01", "timing", [0, 1]),
                                    ("timing-02", "timing", [1, 0])):
            quality = stage == "quality"
            stage_rows.append({
                "name": f"{slug}-{label}", "stage": stage, "cases": order, "rows": 65 if quality else 1,
                "condition": condition, "quality_host_observation": quality,
            })
    return {
        "fixed_before_runs": True, "protocol_sha256": PROTOCOL_SHA, "cases_sha256": CASES_SHA,
        "old_baseline_release": "preupgrade-baseline-20261123",
        "driver_sha256": DRIVER_SHA, "stages": stage_rows,
        "quality_reference": {"NEW_P12": "OLD_P12 quality", "NEW_D24_B12": "NEW_P12 quality"},
        "timing_reference": "the same condition's quality row 64, byte for byte",
        "timing_clock": "llama_decode to llama_get_logits_ith(-1) return",
    }


def verify_snapshot(snapshot, live):
    summary = snapshot["summary"]
    require(CPU_MODEL in live["CPU_model"] and live["config_sha256"] == CONFIG_SHA, "Unexpected API profile")
    require(live["API_health"] == "ok" and live["backing_inodes_verified"] is True, "API identity/readiness")
    require(snapshot["reference"]["sha256"] == "dc834d2a4cbdf824f7e8702a87629e4fe4af5b19f0dadeedceb8865bbdcda6ee",
            "Unexpected reference profile")
    require(summary["all_small_reference_files_match"] and summary["all_weight_sizes_match"],
            "Runtime, asset sizes or small reference files differ")
    require(all(item["kind"] == "hardware_difference" for item in summary["findings"]),
            "Snapshot reported a change outside host hardware")
    proc = snapshot["process"]
    require(proc["pid"] == live["pid"] and proc["pid_identity_stable_across_collection"] is True,
            "Inconsistent server identity in snapshot")


def verify_live_server_identity():
    snapshot = read(ROOT / "production-before/snapshot.json")
    live = read(ROOT / "production-before-live.json")
    verify_snapshot(snapshot, live)
    guard = read(ROOT.parent / "deepseek41-bringup-20261101/health.json")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(guard["status"] == "running" and guard["pid"] == live["pid"] and
            guard["boot_id"] == boot == live["boot_id"], "API guard/boot changed after capture")
    tail = (Path("/proc") / str(live["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(tail[19]) == snapshot["process"]["stat_at_collection_end"]["starttime_ticks"],
            "API PID start time changed after capture")


def verify_design():
    topology = read(ROOT / "TOPOLOGY.json")
    require(topology == live_topology(), "Boot, visible cores or available siblings changed")
    require(read(ROOT / "trial/DESIGN.json") == design(topology), "Design differs from expected six runs")


def verify_installation():
    record = read(ROOT / "STUDY_INPUTS.json")
    for relative, expected in record["installed"].items():
        require(sha(ROOT / relative) == expected, f"Study input changed: {relative}")
    for relative, expected in read(ROOT / "baseline_files.json").items():
        require(sha(BASELINE / relative) == expected, f"Old study source or reference changed: {relative}")
    frozen = read(BASELINE / "FROZEN_ASSETS.json")
    driver = ROOT / "trial/driver"
    require(driver.resolve() == Path(frozen["driver"]["path"]) and
            sha(driver) == frozen["driver"]["sha256"] == DRIVER_SHA, "Frozen driver changed")
    for item in frozen["mapped_elf_files"]:
        if "retained_on_host" in item:
            require(sha(item["path"]) == item["sha256"], f"Project runtime ELF changed: {item['path']}")


def initialize():
    verify_installation()
    verify_live_server_identity()
    topology = live_topology()
    trial = ROOT / "trial"
    require(sha(BASELINE / "intake/BASELINE_PROTOCOL.json") == PROTOCOL_SHA, "Protocol digest")
    require(sha(trial / "CASES.json") == CASES_SHA, "Cases digest")
    for dest in (ROOT / "TOPOLOGY.json", trial / "DESIGN.json", trial / "STATE.json"):
        require(not dest.exists(), f"Existing design or result at {dest}")
    resolved = design(topology)
    write_new(ROOT / "TOPOLOGY.json", topology)
    write_new(trial / "DESIGN.json", resolved)
    write_new(trial / "STATE.json", {"phase": "prepared", "runs": {}, "completed": False})
    for cond in conditions(topology).values():
        print(cond["name"], cond["decode_mask"], cond["prefill_mask"], cond["process_cpu_ids"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize"], help="freeze a fresh six-run design; no model start")
    parser.parse_args()
    initialize()
