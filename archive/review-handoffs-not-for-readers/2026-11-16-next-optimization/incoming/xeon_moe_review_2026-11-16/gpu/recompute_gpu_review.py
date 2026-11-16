#!/usr/bin/env python3
"""Read frozen records and source; no model, service, CUDA, or packaged ELF execution.

Usage: python3 recompute_gpu_review.py --snapshot /path/to/2026-11-16-current
Optional --host-sort compiles ONLY the supplied C++ index unit test in a temp dir.
All findings are local arithmetic / record reanalysis, not new inference benchmarks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def c_div(x, y):
    return (abs(x) // y) * (-1 if x < 0 else 1)


def table(text, name, width, expected):
    body = re.search(r"GGML_TABLE_BEGIN\([^,]+,\s*" + name + r",\s*\d+\)(.*?)GGML_TABLE_END", text, re.S).group(1)
    words = [int(s, 16) for s in re.findall(r"0x[0-9a-fA-F]+", body)]
    assert len(words) == expected, (name, len(words))
    return [[(word >> (8 * i)) & 255 for i in range(width)] for word in words]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="New result file; never overwrite a saved review")
    ap.add_argument("--host-sort", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output must be a new file")
    s = args.snapshot.resolve()
    x = s / "snapshot/llm-lab/exploration-20261112"
    source = s / "snapshot/llm-lab/deepseek41-bringup-20261101/source/ggml/src"
    read = lambda p: json.loads(p.read_text())
    capacity = read(x / "hot-ffn/CAPACITY.json")
    pilot = read(x / "hot-ffn/MODEL_PILOT_RESULTS.json")
    train = read(x / "training/20261112-075449/TRAINING_COUNTS.json")
    records = {r["name"]: r for r in train["records"]}
    hist = [sum(row["H_histogram"][h] for row in pilot["dispatch_layers"]) for h in range(7)]
    calls = sum(hist)
    hit = sum(h * n for h, n in enumerate(hist))
    assert hit == pilot["actual_hot_routes"]
    assert calls * 6 == pilot["actual_total_routes"]
    assert all(sum(row["H_histogram"]) == row["calls"] == 1331 for row in pilot["dispatch_layers"])
    assert all(row["H_histogram"] == [1331, 0, 0, 0, 0, 0, 0] for row in pilot["dispatch_layers"][37:])

    # Exact rational fractional-knapsack relaxation bounds the value of any
    # integral solver that only improves the EXISTING count-per-byte objective.
    candidates = []
    for layer in capacity["layers"][:37]:
        r = records[f'blk.{layer["layer"]}.ffn_gate_exps.weight']
        for e, count in enumerate(r["counts"]):
            candidates.append({"layer": layer["layer"], "expert": e, "count": count, "bytes": layer["expert_bytes"]})
    candidates.sort(key=lambda c: (-Fraction(c["count"], c["bytes"]), -c["count"], c["bytes"], c["layer"], c["expert"]))
    bounds = []
    for row in capacity["budgets"]:
        budget = round(row["budget_GiB"] * 1024 ** 3)
        used = val = 0
        for c in candidates:
            if c["count"] and used + c["bytes"] <= budget:
                used += c["bytes"]
                val += c["count"]
        assert used == row["native_payload_bytes"] and val == row["training_assignments_covered"]
        left = budget
        upper = Fraction(0)
        for c in candidates:
            take = min(left, c["bytes"])
            upper += Fraction(take * c["count"], c["bytes"])
            left -= take
            if not left:
                break
        assert upper >= val
        gap = upper - val
        bounds.append({"budget_GiB": row["budget_GiB"], "greedy_value": val,
                       "fractional_upper_bound": float(upper), "upper_bound_rational": str(upper),
                       "max_improvable_training_assignments": float(gap),
                       "max_relative_improvement_percent": float(gap / val * 100) if val else None,
                       "max_coverage_percentage_points": float(gap / capacity["denominator_route_assignments"] * 100)})

    # Concrete reachable arithmetic cases, using the ACTUAL source tables.
    # q8 activation [0,..,+/-127,..,0] has amax=127,d=1 and quantizes exactly.
    # gguf weight base scale=1, subscale bits=0 and positive signs are valid.
    common = (source / "ggml-common.h").read_text()
    iq2 = table(common, "iq2s_grid", 8, 1024)
    iq3 = table(common, "iq3xxs_grid", 4, 256)
    vals2 = sorted({a for row in iq2 for a in row})
    vals3 = sorted({a for row in iq3 for a in row})
    assert vals2 == [8, 25, 43] and vals3 == [4, 12, 20, 28, 36, 44, 52, 62]
    cases = []
    for typ, grid, value in [("IQ2_S", iq2, 25), ("IQ3_XXS", iq3, 62)]:
        idx, position = next((i, row.index(value)) for i, row in enumerate(grid) if value in row)
        for q in [127, -127]:
            dot = value * q
            if typ == "IQ2_S":
                actual = c_div(c_div(dot, 2), 4)
                exact = Fraction(dot, 8)
            else:
                actual = c_div(c_div(dot, 2), 2)
                exact = Fraction(dot, 4)
            cases.append({"type": typ, "grid_index": idx, "byte_index": position, "grid_value": value,
                          "only_nonzero_q8": q, "weight_base_scale": 1, "subscale_bits": 0,
                          "q8_scale": 1, "integer_expression": actual, "exact_rational_dot": str(exact),
                          "exact_dot": float(exact), "difference": float(Fraction(actual) - exact)})
    assert max(abs(c["difference"]) for c in cases if c["type"] == "IQ2_S") == .875
    assert max(abs(c["difference"]) for c in cases if c["type"] == "IQ3_XXS") == .5
    # Exhaustive residue/subscale check. IQ3_XXS dot is always even; IQ2_S need not be.
    max2 = max(abs(Fraction(c_div(a * sa + b * sb + c_div(a + b, 2), 4)) - Fraction((2 * sa + 1) * a + (2 * sb + 1) * b, 8))
               for a in range(-8, 9) for b in range(-8, 9) for sa in range(16) for sb in range(16))
    max3 = max(abs(Fraction(c_div(scale * dot + c_div(dot, 2), 2)) - Fraction((2 * scale + 1) * dot, 4))
               for dot in range(-32, 33, 2) for scale in range(16))
    assert max2 == Fraction(7, 8) and max3 == Fraction(1, 2)

    small = next(r for r in capacity["budgets"] if r["budget_GiB"] == .5)
    hot14 = next(r for r in capacity["budgets"] if r["budget_GiB"] == 14)
    cal = read(x / "gpu-fit/CALIBRATED_CAPACITY.json")
    memory = {"CUDA_total_MiB_recorded": cal["device_total_MiB"], "final_higher_recorded_used_MiB": 30628,
              "hot_small_native_MiB": small["native_payload_bytes"] / 1024 ** 2,
              "headroom_before_extra_private_graph_workspace_MiB": cal["device_total_MiB"] - 30628 - small["native_payload_bytes"] / 1024 ** 2,
              "headroom_after_prior_768MiB_margin_MiB": cal["device_total_MiB"] - 30628 - small["native_payload_bytes"] / 1024 ** 2 - 768,
              "GPU_layers37_39_native_GiB": capacity["GPU_layers37_39_native_bytes"] / 1024 ** 3,
              "hot14_native_GiB": hot14["native_payload_bytes"] / 1024 ** 3,
              "preinit_guard_required_extra_MiB": 3072,
              "postinit_guard_free_MiB": 2560,
              "fit_proven": False,
              "limits": "Historic capacity arithmetic only. Current capture omits used/free. Constructor guards are before vision/pool growth; whole-128k workload peak is not recorded. Do not relax guards."}
    files = [x / n for n in ["hot-ffn/hot-backend.cpp", "hot-ffn/test-backend.cpp", "hot-ffn/llama-graph.cpp", "hot-ffn/llama-context.cpp",
             "hot-ffn/CAPACITY.json", "hot-ffn/MODEL_PILOT_RESULTS.json", "hot-ffn/FIXTURE_RESULTS.json", "hot-ffn/MISS_ABBA.json",
             "cuda-sort/release-candidate-a/ggml-cuda.cu", "cuda-sort/release-candidate-a/sort-ids.inc", "cuda-sort/test-sort.cpp",
             "cuda-sort/LIVE_ABBA.json", "cuda-sort/gpu-20261112-115341/RESULT.json", "prepare_hot_profiles.py", "prepare_hot_plans.py",
             "plans/VALUE_PRIOR_MANIFEST.json", "HELDOUT_VALUE_PRIOR_R1.json", "gpu-fit/CALIBRATED_CAPACITY.json",
             "checkpoint-1600/FINAL_STATUS.json", "EVENING_FINAL_STATUS.json", "training/20261112-075449/TRAINING_COUNTS.json",
             "cpu-cuda-arithmetic-audit-20261112/TERRA_EVIDENCE.txt", "cpu-cuda-arithmetic-audit-20261112/WEB_NEXT_STEPS_JA.txt"]]
    files += [source / n for n in ["ggml-common.h", "ggml-quants.c", "ggml-cuda/quantize.cu", "ggml-cuda/vecdotq.cuh", "ggml-cuda/mmvq.cu"]]
    files += [s / n for n in ["review/BINARY_SOURCE_MAP.json", "review/capture/CURRENT_STATE.json", "review/adopted-overlay-diff/INDEX.json",
                              "snapshot/llm-lab/gpu-stream-20261103/ggml-backend.cpp"]]
    result = {"scope": "Saved evidence reanalysis, source arithmetic proof, and optional host index unit test. No CUDA/model/service/packaged ELF execution.",
              "source_commit": "7d8b077afcf1258602366245f1567e820255defa",
              "source_sha256": {str(p.relative_to(s)): sha(p) for p in files},
              "pilot_H_distribution": {"all_layers": hist, "total_layer_calls": calls, "hit_assignments": hit,
                  "all40_hit_fraction": hit / (calls * 6), "layers0_36_hit_fraction": hit / (1331 * 37 * 6),
                  "H2_H4_H5_calls": sum(hist[h] for h in [2, 4, 5]), "H2_H4_H5_fraction": sum(hist[h] for h in [2, 4, 5]) / calls,
                  "H6_fraction": hist[6] / calls, "scope": pilot["routing_scope"]},
              "potential_transfer_reduction": {
                  "interpretation": "A hypothetical GPU output backend uploads CPU miss slots only, preserving original slot ordering. Byte totals are not a speed prediction.",
                  "output_roundtrip_saved_bytes_per_40_layer_T1_pass": 2 * hit / 1331 * 5120 * 4,
                  "private_input_upload_removable_bytes_per_40_layer_T1_pass_upper_bound": (calls - hist[0]) / 1331 * 5120 * 4,
                  "not_necessarily_generated_tokens": True},
              "existing_frequency_objective_knapsack_bounds": bounds,
              "arithmetic_cases": {"IQ2_S_grid_values": vals2, "IQ3_XXS_grid_values": vals3,
                 "reachable_cases": cases, "IQ2_S_max_unscaled_truncation_abs": float(max2), "IQ3_XXS_max_unscaled_truncation_abs": float(max3),
                 "IQ2_S_exact_numerator_abs_bound": 31 * 32 * 43 * 127,
                 "IQ3_XXS_exact_numerator_abs_bound": 31 * 32 * 62 * 127,
                 "bounds_scope": "Per 32-activation dot before scales. Constructive proof from frozen tables; not measured frequency, actual CUDA execution, or model-quality impact."},
              "conditional_small_cache_capacity": memory,
              "host_sort_test": {"run": False}}
    if args.host_sort:
        compiler = shutil.which("g++")
        if not compiler:
            raise RuntimeError("g++ required for --host-sort")
        test_src = x / "cuda-sort/test-sort.cpp"
        helper = x / "cuda-sort/release-candidate-a/sort-ids.inc"
        runs = []
        with tempfile.TemporaryDirectory(prefix="xeon_host_sort_") as td:
            td = Path(td)
            shutil.copy2(test_src, td / "test-sort.cpp")
            shutil.copy2(helper, td / "sort-ids.inc")
            for mode, flags in [("release_ndebug", ["-O2", "-DNDEBUG"]),
                                ("ubsan_asan", ["-O1", "-g", "-DNDEBUG", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])]:
                command = [compiler, "-std=c++17", *flags, str(td / "test-sort.cpp"), "-o", str(td / "test-sort")]
                cp = subprocess.run(command, capture_output=True, text=True, timeout=60)
                if cp.returncode:
                    raise RuntimeError(cp.stderr)
                cp = subprocess.run([str(td / "test-sort")], capture_output=True, text=True, timeout=60)
                runs.append({"mode": mode, "compiler_flags": flags, "returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr})
                if cp.returncode and "LeakSanitizer does not work under ptrace" in cp.stderr:
                    runs[-1]["initial_runtime_failure"] = {"returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}
                    env = os.environ.copy()
                    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
                    cp = subprocess.run([str(td / "test-sort")], capture_output=True, text=True, timeout=60, env=env)
                    runs[-1].update(returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr,
                                    leak_check_disabled_reason="LeakSanitizer refuses this ptrace environment; ASAN bounds and UBSAN remain enabled.")
                if cp.returncode:
                    raise RuntimeError(json.dumps(runs[-1]))
                runs[-1]["result"] = json.loads(cp.stdout)
        result["host_sort_test"] = {"run": True, "compiler": subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
                                     "runs": runs, "scope": "Existing exact sort test, adopted helper, host index equivalence only; no kernels or inference executed. Temporary binaries deleted."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "H": hist, "arithmetic_cases": cases,
                      "hot14_count_objective_bound": next(r for r in bounds if r["budget_GiB"] == 14),
                      "host_sort_run": result["host_sort_test"]["run"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
