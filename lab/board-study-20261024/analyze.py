#!/usr/bin/env python3
import collections
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name


def watts(a, b):
    dt = b["monotonic"] - a["monotonic"]
    result = {"package": 0.0, "dram": 0.0}
    for key, start in a["energy"].items():
        end = b["energy"][key]
        delta = (end["uj"] - start["uj"]) % start["range"]
        kind = "package" if start["name"].startswith("package-") else "dram"
        result[kind] += delta / 1e6 / dt
    result["cpu_plus_dram"] = result["package"] + result["dram"]
    return result


def main():
    reports = {}
    sensor_values = collections.defaultdict(list)
    for folder in sorted(OUT.glob("policy-*")) + [OUT / "memory-smoke", OUT / "memory-96g"]:
        if not folder.is_dir():
            continue
        run = json.loads((folder / "run.json").read_text())
        assert run["returncode"] == 0 and "error" not in run and "ECC" not in run["sel_after"]
        assert run["before"]["counters"] == run["after"]["counters"]
        samples = [run["before"], *run["samples"], run["after"]]
        intervals = [watts(a, b) for a, b in zip(samples, samples[1:]) if b["monotonic"] - a["monotonic"] >= 0.5]
        entry = {"elapsed_s": run["elapsed_s"], "average_power_W": watts(samples[0], samples[-1]),
                 "maximum_observed_interval_power_W": max(x["cpu_plus_dram"] for x in intervals),
                 "max_cpu_C": max(v for s in samples for v in s["temperatures_C"].values()),
                 "max_disk_read_bytes": max((s.get("process") or {}).get("disk_read_bytes", 0) for s in samples)}
        if folder.name.startswith("policy-"):
            policy = json.loads((folder / "policy.json").read_text())
            entry["variant"] = policy["label"]
            entry["idle_power_W"] = watts(policy["idle_samples"][0], policy["idle_samples"][-1])
            entry["bench"] = json.loads((folder / "stdout.txt").read_text())
        for snapshot in run["bmc"]:
            for line in snapshot["sensors"].splitlines():
                fields = [x.strip() for x in line.split("|")]
                if len(fields) < 4:
                    continue
                try:
                    value = float(fields[1])
                except ValueError:
                    continue
                sensor_values[fields[0]].append({"value": value, "unit": fields[2], "state": fields[3]})
        reports[folder.name] = entry
    policies = {}
    for variant in "ABCD":
        entries = [r for r in reports.values() if r.get("variant") == variant]
        assert len(entries) == 2
        result = {"commands": 2, "timed_samples_per_test": 4, "tests": {},
                  "mean_whole_command_power_W": statistics.mean(r["average_power_W"]["cpu_plus_dram"] for r in entries),
                  "mean_idle_power_W": statistics.mean(r["idle_power_W"]["cpu_plus_dram"] for r in entries)}
        for prompt, generated in ((128, 0), (512, 0), (0, 128)):
            rows = [b for r in entries for b in r["bench"] if b["n_prompt"] == prompt and b["n_gen"] == generated]
            times = [t for row in rows for t in row["samples_ns"]]
            total = prompt + generated
            result["tests"][f"pp{prompt}" if prompt else f"tg{generated}"] = {
                "pooled_tokens_s": len(times) * total * 1e9 / sum(times),
                "sample_tokens_s": [total * 1e9 / t for t in times]}
        policies[variant] = result
    for variant, policy in policies.items():
        for name, test in policy["tests"].items():
            test["change_from_A_percent"] = (test["pooled_tokens_s"] / policies["A"]["tests"][name]["pooled_tokens_s"] - 1) * 100
    sensors = {name: {"min": min(x["value"] for x in rows), "max": max(x["value"] for x in rows),
                      "unit": rows[0]["unit"], "states": sorted(set(x["state"] for x in rows))} for name, rows in sensor_values.items()}
    maps = (OUT / "memory-96g-numa-maps.txt").read_text().splitlines()
    largest = max(maps, key=lambda x: int((re.search(r"anon=(\d+)", x) or ["", "0"])[1]))
    placement = {f"node{node}_GiB_approx": int(pages) * 4096 / 2**30 for node, pages in re.findall(r"N(\d+)=(\d+)", largest)}
    summary = {"power_policies": policies, "runs": reports, "bmc_sensor_ranges": sensors,
               "memory_allocation_by_node": placement, "cpu_policy_selected": "A: original schedutil / EPB 7",
               "power_scope": "CPU packages plus DRAM from RAPL; not wall or PSU output power. Averages include loading and all command phases."}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in policies.items():
        print(key, {name: round(test["pooled_tokens_s"], 3) for name, test in value["tests"].items()},
              "active_W", round(value["mean_whole_command_power_W"], 2), "idle_W", round(value["mean_idle_power_W"], 2))
    print("Memory", reports["memory-96g"], placement)
    print("Sensors", {k: sensors[k] for k in ("PCH Temp", "12V", "5V", "3.3V", "VBAT", "P1-DIMMB1 TEMP", "P2-DIMMG1 TEMP")})


if __name__ == "__main__":
    main()
