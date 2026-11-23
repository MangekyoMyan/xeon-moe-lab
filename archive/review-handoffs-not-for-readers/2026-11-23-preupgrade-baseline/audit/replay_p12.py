"""Independently recalculate fixed-input and raw-logit evidence from the three OLD_P12 processes."""

import argparse
import array
import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path


VOCAB = 129280
QUALITY_ROWS = 65
ROW_BYTES = VOCAB * 4
NOV19_GOLD = {
    "text_cache_budget_ja": "35d028fed9fa08bbe8f304c598ced362ff466aff176ecf407133c0a1870bbd64",
    "code_stable_grouping_python": "077eefd33928bfbec340162c5b5e54c2bbad9e4ac3ae72bc23f38a7a33d69d01",
}
CASE_FILE = "0458b7a358d0eb3a0c289eb1a1aa9903ee0bb972158acc64abbf0952f472d746"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tokens_digest(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def stats(ns):
    values = sorted(ns)
    count = len(values)
    total = sum(values)
    return {
        "N": count,
        "sum_ns": total,
        "mean_ms": total / count / 1e6,
        "median_ms": statistics.median(values) / 1e6,
        "p95_ms_nearest_rank": values[(count * 95 + 99) // 100 - 1] / 1e6,
        "max_ms": values[-1] / 1e6,
        "T1_per_second": count * 1e9 / total,
    }


def floats(row):
    require(len(row) == ROW_BYTES, "short F32 row")
    values = array.array("f")
    require(values.itemsize == 4, "unexpected native float representation")
    values.frombytes(row)
    if sys.byteorder == "big":
        values.byteswap()
    require(all(map(math.isfinite, values)), "nonfinite logit")
    return values


def compare(observed, reference, rows):
    require(observed.stat().st_size == rows * ROW_BYTES, "observed logits shape")
    require(reference.stat().st_size == QUALITY_ROWS * ROW_BYTES, "reference logits shape")
    max_abs, energy_rows, error_rows = 0.0, [], []
    bytes_equal, changed_argmax = True, []
    with observed.open("rb") as obs, reference.open("rb") as ref:
        ref.seek((QUALITY_ROWS - rows) * ROW_BYTES)
        for i in range(rows):
            ref_bytes, obs_bytes = ref.read(ROW_BYTES), obs.read(ROW_BYTES)
            expected = floats(ref_bytes)
            energy_rows.append(math.fsum(x * x for x in expected))
            if ref_bytes == obs_bytes:
                continue
            bytes_equal = False
            actual = floats(obs_bytes)
            diff = [x - y for x, y in zip(actual, expected, strict=True)]
            max_abs = max(max_abs, max(map(abs, diff)))
            error_rows.append(math.fsum(x * x for x in diff))
            if expected.index(max(expected)) != actual.index(max(actual)):
                changed_argmax.append(QUALITY_ROWS - rows + i)
    denom = math.fsum(energy_rows)
    return {
        "observed_sha256": digest(observed), "reference_file_sha256": digest(reference),
        "rows": rows, "vocab": VOCAB, "reference_first_row": QUALITY_ROWS - rows,
        "all_finite": True, "byte_equal": bytes_equal, "max_abs_error": max_abs,
        "NMSE": math.fsum(error_rows) / denom if denom else None,
        "argmax_changes": len(changed_argmax), "argmax_changed_rows": changed_argmax,
    }


def check_blocks(folder, stage, case_order, cases):
    raw = load(folder / "RUN.json")
    spec = load(folder / "SPEC.json")
    require(raw["passed"] and raw["vocab_size"] == VOCAB, "run or vocabulary")
    require(raw["spec"] == spec and spec["cases"] == case_order and spec["stage"] == stage, "stage spec")
    blocks = raw["blocks"]
    require([b["case_index"] for b in blocks] == case_order, "case order")
    require(raw["vision_loaded"] and raw["compact_bytes"] == 127401984000 and
            raw["compact_bytes_after_unload"] == 0 and raw["after_unload_KiB"]["VmLck"] == 0,
            "projector, compact bytes or post-unload locked memory")
    require([x["routed_host"] for x in raw["placement"]] == [True] * 37 + [False] * 3, "placement")
    pin = load(folder / "PROVIDER_EXPECTED.json")
    require(raw["providers"] == pin["providers"] and raw["active_backend_providers"] == pin["active_backend_providers"],
            "active providers")
    prep = raw["preparation"]
    require(prep["JA_first64"] is True and prep["JA_full_prefix"] == 272 and prep["JA_reference_T1"] == 4,
            "identical per-process warmup")
    require(0 < prep["warmup_start_ns"] < prep["warmup_end_ns"] <= blocks[0]["prefill_begin_ns"], "warmup clock")
    require(min(raw[k] for k in ("text_model_context_load_ns", "vision_init_ns", "unload_ns")) > 0,
            "load/projector/unload clocks")
    require(("registration" in raw) == (stage == "quality"), "registration observer scope")
    if stage == "quality":
        reg = raw["registration"]
        require((reg["registered_tensors"], reg["selected_projections"], reg["sampled_pages"]) == (80, 19200, 1440)
                and reg["sampled_vma_policies"] == {"interleave:0-1": 1440}, "compact selection and VMA sample")
        if reg["page_location_errno"] == 0:
            nodes = reg["sampled_page_nodes"]
            require(set(nodes) == {"0", "1"} and all(n > 0 for n in nodes.values()) and sum(nodes.values()) == 1440,
                    "physical placement of sampled compact pages")
    require(("ja_quality_task_observations" in raw) == spec.get("quality_host_observation", False),
            "phase-observation scope")
    rev = {}
    last_end = 0
    for block, index in zip(blocks, case_order, strict=True):
        case = cases[index]
        name = case["case_id"]
        require(block["case_id"] == name and case["split"] == block["split"] == "train", "case identity")
        for key, ref_key, hash_key in (("prefix_ids", "prefix_ids", "prefix_ids_i32_sha256"),
                                      ("input_tokens", "used_T1_ids", "used_T1_ids_i32_sha256")):
            ids = block[key]
            require(ids == case[ref_key] and tokens_digest(ids) == case[hash_key], "input token digest")
        require(block["input_tokens"] == case["next_targets_prefix_and64T1"][:64], "teacher IDs")
        require(block["positions_begin"] == case["first_T1_position"] == len(case["prefix_ids"]), "position")
        require(block["status"] == [0] * 64 and block["traced_calls"] == 0, "decode status and observer")
        require(block["rows"] == (QUALITY_ROWS if stage == "quality" else 1) and block["file"] == name + ".f32",
                "saved file/rows")
        starts, ends = block["start_ns"], block["output_ready_ns"]
        require(len(starts) == len(ends) == 64, "T1 call count")
        require(last_end < block["prefill_begin_ns"] < block["prefill_ready_ns"] <= starts[0],
                "prefill ready or inter-case clock order")
        require(all(a < b for a, b in zip(starts, ends)) and
                all(b <= a for b, a in zip(ends[:-1], starts[1:])), "T1 clocks")
        last_end = ends[-1]
        rev[name] = {
            "ns": [b - a for a, b in zip(starts, ends)],
            "prefill_seconds": (block["prefill_ready_ns"] - block["prefill_begin_ns"]) / 1e9,
        }
    return rev


def cpuset(value):
    cpus = set()
    for part in value.split(","):
        numbers = list(map(int, part.split("-")))
        lo, hi = numbers[0], numbers[-1]
        require(1 <= len(numbers) <= 2 and 0 <= lo <= hi, "CPU range syntax")
        cpus.update(range(lo, hi + 1))
    return cpus


def option(argv, flag):
    matches = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]
    matches.extend(arg[len(flag) + 1:] for arg in argv if arg.startswith(flag + "="))
    require(len(matches) == 1, "expected one " + flag)
    return matches[0]


def launch_facts(folder, snapshot, condition, historical, design_stage, driver_sha):
    health = load(folder / "health.json")
    launch = load(folder / "LAUNCH.json")
    report = load(folder / "RUN.json")
    argv = launch["command"]
    selected = cpuset(option(argv, "--physcpubind"))
    mask = sum(1 << cpu for cpu in selected)
    expected_args = historical["args"].copy()
    for flag, value in (("-tb", "12"), ("--cpu-mask", hex(mask)), ("--cpu-mask-batch", hex(mask))):
        expected_args[expected_args.index(flag) + 1] = value
    require(launch["args"] == expected_args and argv[-len(expected_args):] == expected_args, "full model arguments")
    require(launch["spec"] == report["spec"] == design_stage and launch["driver_sha256"] == driver_sha,
            "frozen process design and same measurement binary")
    live_env = snapshot["process"]["performance_environment"]["values"]
    require(launch["environment"] == {**live_env, "LD_BIND_NOW": "1"}, "live performance environment")
    dsos = snapshot["process"]["profile_library_comparisons"]
    require(len(dsos) == 4 and all(v["matches"] for v in dsos), "four observed process libraries")
    require(len(launch["libraries"]) == 4 and
            {v["sha256"] for v in launch["libraries"].values()} == {v["expected_sha256"] for v in dsos},
            "launched library hashes vs production backing files")
    require(launch["reserve_GiB"] == 64, "RAM reserve")
    require(int(option(argv, "--cpu-mask"), 16) == int(option(argv, "--cpu-mask-batch"), 16) == mask,
            "launcher and both graph masks")
    expected_flags = {
        "-t": "12", "-tb": "12", "--cpu-strict": "1", "--cpu-strict-batch": "1",
        "-c": "131072", "-b": "1024", "-ub": "1024", "--parallel": "1",
        "--cache-type-k": "f16", "--cache-type-v": "f16", "--interleave": "0,1",
        "--load-mode": "mmap+mlock", "--lazy-mode": "off", "--n-cpu-moe": "37",
    }
    for flag, expected in expected_flags.items():
        require(option(argv, flag) == expected, "launch " + flag)
    require("--op-offload" in argv and "--mmproj-offload" in argv and
            any(arg.endswith("/without_thp.py") for arg in argv), "offload/THP wrapper")
    require(health["status"] == "exited" and health["child_returncode"] == 0 and
            health["started_unix"] <= health["finished_unix"], "guard lifecycle")
    available = snapshot["hardware"]["cpu"]
    require(selected <= cpuset(available["global"]["online"]["value"]), "online CPU allowance")
    require(selected <= set(available["collector_affinity"]["cpus"]), "collector CPU allowance")
    physical = {}
    per_socket = {}
    for entry in available["sysfs_topology"]:
        cpu = entry["cpu"]
        if cpu in selected:
            t = entry["topology"]
            pair = (int(t["physical_package_id"]["value"]), int(t["core_id"]["value"]))
            require(pair not in physical and selected & cpuset(t["thread_siblings_list"]["value"]) == {cpu},
                    "more than one sibling selected")
            physical[pair] = cpu
            per_socket[pair[0]] = per_socket.get(pair[0], 0) + 1
    require(len(selected) == len(physical) == 12 and sorted(per_socket.values()) == [6, 6], "P12 physical cores")
    entry, parsed, context = report["linux_entry"], report["parsed_cpu"], report["context"]
    require(report["spec"]["condition"] == report["requested"] == condition, "recorded request and frozen condition")
    require(condition["name"] == "OLD_P12" and condition["strict"] is True and condition["interleave_nodes"] == [0, 1],
            "OLD_P12 condition")
    require(condition["decode_threads"] == condition["prefill_threads"] == condition["projector_threads"] == 12 and
            condition["OMP_THREAD_LIMIT"] == 24 and condition["packing_threads_at_load"] == 12, "requested counts")
    require(sorted(selected) == entry["effective_cpu_ids"] == condition["process_cpu_ids"] ==
            condition["decode_cpu_ids"] == condition["prefill_cpu_ids"], "entry and requested CPU IDs")
    require(int(condition["decode_mask"], 16) == int(condition["prefill_mask"], 16) == mask and
            parsed["decode_ids"] == parsed["prefill_ids"] == sorted(selected), "parsed mask")
    require(parsed["decode_count"] == parsed["prefill_count"] == 12 and parsed["decode_strict"] and
            parsed["prefill_strict"], "parsed counts and strictness")
    require(entry["main_tid"] == entry["pid"] == health["pid"] and entry["mempolicy_mode"] == 3 and
            entry["mempolicy_nodes_mask"] == 3 and entry["thp_disabled"] is True, "Linux entry observation")
    require(context["threads"] == context["threads_batch"] == 12 and context["n_ctx"] == 131072 and
            context["n_batch"] == context["n_ubatch"] == 1024 and context["n_outputs_max"] == 1 and
            context["n_rs_seq"] == 0 and report["projector_init_threads_argument"] == 12, "context and projector")
    return {
        "pid": health["pid"], "boot_id": health["boot_id"], "guard_returncode": health["child_returncode"],
        "guard_start_unix": health["started_unix"], "guard_end_unix": health["finished_unix"],
        "numactl_cpu_ids": sorted(selected), "cores_per_socket": per_socket, "strict_mask": hex(mask),
        "entry_and_parsed_masks_agree": True, "context_thread_getters": [context["threads"], context["threads_batch"]],
        "numa_interleave_node_mask": entry["mempolicy_nodes_mask"], "entry_THP_disabled": entry["thp_disabled"],
        "projector_init_thread_argument": report["projector_init_threads_argument"],
    }


def quality_tasks(report):
    observation = report["ja_quality_task_observations"]
    block = report["blocks"][0]
    require(block["case_id"] == "text_cache_budget_ja", "Japanese quality observation")
    clocks = {
        "prefill": (block["prefill_begin_ns"], block["prefill_ready_ns"]),
        "decode": (block["start_ns"][0], block["output_ready_ns"][-1]),
    }
    summary = {}
    for name, (begin, end) in clocks.items():
        before, after = observation["before_" + name], observation["after_" + name]
        require(before["end_ns"] <= begin < end <= after["begin_ns"], "task inventory phase bracket")
        old = {(row["tid"], row["start_ticks"]): row for row in before["tasks"]}
        active = []
        for row in after["tasks"]:
            earlier = old.get((row["tid"], row["start_ticks"]))
            if earlier is None:
                continue
            delta = row["runtime_ns"] - earlier["runtime_ns"]
            require(delta >= 0, "per-TID runtime reversal")
            if delta > 0:
                active.append({"tid": row["tid"], "runtime_delta_ns": delta, "affinity_before": earlier["affinity"],
                               "affinity_after": row["affinity"]})
        singleton = [task for task in active if len(task["affinity_before"]) == 1 and
                     task["affinity_before"] == task["affinity_after"]]
        summary[name] = {
            "common_TIDs_with_positive_runtime": sorted(active, key=lambda row: row["tid"]),
            "TIDs_active_with_same_singleton_affinity": len(singleton),
            "distinct_cpus_of_these_TIDs": sorted({task["affinity_before"][0] for task in singleton}),
        }
    summary["scope"] = ("One Japanese quality case. The decode bracket also contains quality logit checks and copies. "
                        "Per-TID runtime and endpoint affinity alone do not identify participants in an individual graph.")
    return summary


def replay(study, runs, old_gold):
    protocol = load(study / "intake" / "BASELINE_PROTOCOL.json")
    reference_cases = study / "intake" / "reference" / "CASES_633e.json"
    require(digest(reference_cases) == CASE_FILE == protocol["fixed_cases_sha256"], "immutable cases")
    cases = load(reference_cases)["cases"]
    stages = protocol["runs"]
    require([p["stage"] for p in stages] == ["quality", "timing", "timing"], "three planned processes")
    require(runs and len(runs) == 3 and len(set(runs)) == 3, "three distinct output directories")
    quality = runs[0]
    snapshot = load(study / "production-before" / "snapshot.json")
    condition = load(study / "TOPOLOGY.json")["measurement"]
    historical = load(study / "intake" / "reference" / "timing-0-control.LAUNCH.json")
    design = load(study / "trial" / "DESIGN.json")
    driver_sha = digest(study / "trial" / "driver")
    require(driver_sha == load(study / "trial" / "BUILD.json")["driver_sha256"], "built driver digest")
    output = {
        "source": "RUN.json start/ready nanoseconds, F32-LE logits, archived fixed case IDs",
        "fixed_case_sha256": CASE_FILE,
        "quality_vs_Nov19_normal_12_24": {}, "timing_tails_vs_OLD_P12_quality": {},
        "timing_per_process": {}, "timing_combined_by_case": {},
        "process_evidence": {},
    }
    samples = {c["case_id"]: [] for c in cases}
    for folder, plan, design_stage in zip(runs, stages, design["stages"], strict=True):
        stage = plan["stage"]
        output["process_evidence"][folder.name] = launch_facts(folder, snapshot, condition, historical, design_stage,
                                                               driver_sha)
        per_case = check_blocks(folder, stage, plan["case_indices"], cases)
        if stage == "quality":
            output["quality_task_observations"] = quality_tasks(load(folder / "RUN.json"))
        else:
            output["timing_per_process"][folder.name] = {}
            output["timing_tails_vs_OLD_P12_quality"][folder.name] = {}
        for name, measurements in per_case.items():
            observed = folder / (name + ".f32")
            if stage == "quality":
                gold = old_gold / (name + ".f32")
                require(digest(gold) == NOV19_GOLD[name], "historical gold digest from November 19")
                output["quality_vs_Nov19_normal_12_24"][name] = compare(observed, gold, QUALITY_ROWS)
            else:
                tails = output["timing_tails_vs_OLD_P12_quality"][folder.name]
                tails[name] = compare(observed, quality / (name + ".f32"), 1)
                summary = stats(measurements["ns"])
                summary["prefill_seconds"] = measurements["prefill_seconds"]
                output["timing_per_process"][folder.name][name] = summary
                samples[name].extend(measurements["ns"])
    for name, ns in samples.items():
        require(len(ns) == 128, "per case timing sample total")
        summary = stats(ns)
        means = [p[name]["mean_ms"] for p in output["timing_per_process"].values()]
        summary["process_mean_range_ms"] = [min(means), max(means)]
        output["timing_combined_by_case"][name] = summary
    process = list(output["process_evidence"].values())
    require(len({p["pid"] for p in process}) == 3 and len({p["boot_id"] for p in process}) == 1,
            "distinct model processes on this boot")
    require(all(first["guard_end_unix"] <= second["guard_start_unix"] for first, second in zip(process, process[1:])),
            "overlapping process windows")
    output["timing_process_count"] = len(output["timing_per_process"])
    output["timing_T1_count"] = sum(map(len, samples.values()))
    output["all_timing_tails_byte_equal_to_P12"] = all(
        entry["byte_equal"] for process in output["timing_tails_vs_OLD_P12_quality"].values() for entry in process.values()
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-gold", required=True, type=Path)
    parser.add_argument("runs", nargs=3, type=Path, help="quality, first timing, second timing")
    args = parser.parse_args()
    print(json.dumps(replay(args.study, args.runs, args.old_gold), indent=2, allow_nan=False))
