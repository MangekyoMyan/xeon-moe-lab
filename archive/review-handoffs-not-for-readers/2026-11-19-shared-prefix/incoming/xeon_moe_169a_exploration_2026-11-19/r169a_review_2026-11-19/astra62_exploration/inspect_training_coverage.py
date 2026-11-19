"""Count real training routes and a diagnostic equal-byte CPU compact substitution."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess


COMMIT = "169a2f7eab8d52071661f0da6859d38df4206f8b"
ARCHIVE = "archive/review-handoffs-not-for-readers"
CAPTURE = f"{ARCHIVE}/2026-11-17-gpu-workspace/pilot-capture/runs/20261117T035402Z-U1024-d22b3e04"
CPU = f"{ARCHIVE}/2026-11-16-current/snapshot/llm-lab/exploration-20261112"
INVENTORY = f"{ARCHIVE}/2026-11-17-parallel-pilot/replay/runs/20261117T053052Z-U1024-1563d936/RUN.json"
CASES = ("text_cache_budget_ja", "text_cache_budget_en", "mixed_build_log_ja_en_log",
         "code_stable_grouping_python", "code_stable_grouping_cpp")
TARGETS = [(3, 147), (7, 270)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Bundled reference_source directory or Git checkout with local blobs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hashes = {}

    def read(path):
        if (args.source / ".git").exists():
            raw = subprocess.check_output(["git", "--no-lazy-fetch", "-C", str(args.source),
                                           "show", f"{COMMIT}:{path}"])
        else:
            raw = (args.source / path).read_bytes()
        hashes[path] = sha256(raw).hexdigest()
        return raw

    plan_bytes = read(f"{CPU}/plans/frequency19200.plan")
    head, *plan_rows = plan_bytes.decode().splitlines()
    assert head == "XEON_IQ_CACHE_PLAN_V2 80 19200 127401984000"
    tensors = {}
    for line in plan_rows:
        name, qtype, k, rows, experts, count, *ids = line.split()
        members = set(map(int, ids))
        assert len(members) == int(count) and int(experts) == 384
        tensors[name] = {"type": qtype, "members": members}
    inventory = json.loads(read(INVENTORY))["native_expert_inventory"]
    projections = {v["layer"]: v["projections"] for v in inventory if v["host"]}
    assert set(projections) == set(range(37))
    assert set(tensors) == {p["name"] for projs in projections.values() for p in projs if p["type"] != "mxfp4"}
    old = json.loads(read(f"{CPU}/training/20261112-075449/TRAINING_COUNTS.json"))
    historical = {record["name"]: record for record in old["records"]}
    original_bank = read(f"{ARCHIVE}/2026-11-17-parallel-pilot/integrated/capacity.plan").decode().splitlines()[1:]
    gpu_ids = {}
    for row in original_bank:
        layer, _, _, _, count, *ids = row.split()
        assert len(ids) == int(count)
        gpu_ids[int(layer)] = set(map(int, ids))
    route_counts = {}
    focal = {f"L{layer}/E{expert}": {} for layer, expert in TARGETS}
    by_case = {}
    for case in CASES:
        doc = json.loads(read(f"{CAPTURE}/{case}.json"))
        records = doc["T1_trace"]["records"]
        assert len(records) == 37 * 64
        assert len({(r["decode_call"], r["layer"]) for r in records}) == 37 * 64
        counts = Counter()
        info = Counter()
        for r in records:
            layer, ids = r["layer"], r["ids"]
            assert len(ids) == len(set(ids)) == 6 and all(0 <= e < 384 for e in ids)
            for expert in ids:
                counts[layer, expert] += 1
                for p in projections[layer]:
                    if p["type"] == "mxfp4":
                        info["mxfp4_projection_slots"] += 1
                        continue
                    packed = expert in tensors[p["name"]]["members"]
                    info["packed_iq_projection_slots" if packed else "native_iq_projection_slots"] += 1
            for l, e in TARGETS:
                if l != layer or e not in ids:
                    continue
                f = focal[f"L{l}/E{e}"].setdefault(case, {"appearances": 0, "same_route_original35_gpu_hit": 0})
                f["appearances"] += 1
                f["same_route_original35_gpu_hit"] += bool(gpu_ids[l].intersection(ids))
        for l, e in TARGETS:
            focal[f"L{l}/E{e}"].setdefault(case, {"appearances": 0, "same_route_original35_gpu_hit": 0})
        assert sum(info.values()) == 37 * 64 * 6 * 3
        route_counts[case] = counts
        by_case[case] = dict(info)
    changes = []
    old_sum = 0
    nets, hits, losses = Counter(), Counter(), Counter()
    for layer, expert in TARGETS:
        for projection in projections[layer]:
            name = projection["name"]
            enrolled = tensors[name]["members"]
            hist = historical[name]["counts"]
            assert expert not in enrolled and len(hist) == 384
            donor = min(enrolled, key=lambda member: (hist[member], member))
            old_sum += hist[expert] - hist[donor]
            row = {"name": name, "new_member": expert, "removed_member": donor,
                   "old_census_new_member": hist[expert], "old_census_removed_member": hist[donor], "training5": {}}
            for c in CASES:
                plus = route_counts[c][layer, expert]
                minus = route_counts[c][layer, donor]
                nets[c] += plus - minus
                hits[c] += plus
                losses[c] += minus
                row["training5"][c] = {"new_packed": plus, "lost_packed": minus}
            changes.append(row)
    assert len(changes) == 6
    result = {
        "commit": COMMIT, "input_sha256": hashes, "case_count": len(CASES), "T1_routes": 37 * 64 * len(CASES),
        "ordered_route_expert_slots": 37 * 6 * 64 * len(CASES), "baseline_projection_slots_by_case": by_case,
        "focal_experts": focal, "diagnostic_cpu_option": {
            "scope": "Membership counts only on frozen training data, no time or model predictions. No plan file, new measured result or validation data.",
            "method": "Start from frequency19200; for the three projections of each of the two pre-existing focal experts, substitute the least common currently packed member in that same tensor under original Nov 12 census (tie break lowest id). Old census alone determines the six donors, independent of the five recent counts. Keep 80 IQ tensors, row-wise sizes, 19,200 matrix slots and 127401984000 packed bytes. No GPU bank.",
            "changes": changes, "total_old_census_net_iq_projections": old_sum,
            "training5_projection_count_changes": {case: {"added": hits[case], "removed": losses[case],
                                                         "net": nets[case]} for case in CASES}},
        "cautions": ["Original census was aggregated and may include prefill tails. No shared denominators between old census and training5.",
                     "CPU sparse lookup membership is inferred from the frozen plan, not a fresh runtime pointer inspection. Native vs compact arithmetic and time require a live matched comparison.",
                     "Two focal identities were originally drawn from these training cases. No validation or generalization claim. A GPU allocation changes which CPU experts execute. This CPU-only alternative carries no GPU bank."],
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
