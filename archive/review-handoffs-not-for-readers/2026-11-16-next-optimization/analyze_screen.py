"""Analyze saved single-thread CPU screens; no benchmark or service operation."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
NAMES = {1: "IQ3_XXS", 2: "IQ2_S", 3: "IQ3_S"}


def key(row):
    return row["type"], row["variant"], row["n"], row["rows"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    arms = []
    sources = []
    expected = {(t, "experimental_input_abs", n, rows) for t in NAMES for n in (2304, 5120) for rows in (256, 8192)}
    expected |= {(1, "stored_iq_rhs_bias62", n, rows) for n in (2304, 5120) for rows in (256, 8192)}
    for label in ("abs-original-xeon", "abs-socket1-xeon"):
        path = ROOT / label / "RESULT.json"
        data = json.loads(path.read_text())
        require(data.get("test_passed") is True and data.get("benchmark_passed") is True, "Test/benchmark failure")
        rows = data["benchmark_rows"]
        require(len(rows) == 16 and {key(x) for x in rows} == expected, "Expected case set differs")
        require(all(x["exit_code"] == 0 for x in data["commands"]), "Failed command in receipt")
        for row in rows:
            for field, count in (("paired_speedups", 4), ("thread_cpu_paired_speedups", 4), ("current_ns_per_dot", 8), ("candidate_ns_per_dot", 8), ("current_thread_cpu_ns_per_dot", 8), ("candidate_thread_cpu_ns_per_dot", 8)):
                require(len(row[field]) == count and all(math.isfinite(v) and v > 0 for v in row[field]), "Incomplete or nonfinite timing series")
            require(row["preparation_once_per_matrix_included"] is True, "Preparation excluded")
        arms.append({key(x): x for x in rows})
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "binary_sha256": data["probe_sha256"]})
    require(sources[0]["binary_sha256"] == sources[1]["binary_sha256"], "Different binaries")
    results = []
    for k in sorted(expected):
        pair = [a[k] for a in arms]
        # The prospective 3% rule uses wall timing; thread CPU time corroborates,
        # and must not rescue a wall result just below the boundary.
        survives = all(x["paired_median_speedup"] >= 1.03 and min(x["paired_speedups"]) > 1 and x["thread_cpu_paired_median_speedup"] >= 1.03 and min(x["thread_cpu_paired_speedups"]) > 1 for x in pair)
        results.append({"type": NAMES[k[0]], "variant": k[1], "K": k[2], "rows": k[3],
                        "wall_speedup_percent": [100 * (x["paired_median_speedup"] - 1) for x in pair],
                        "thread_cpu_speedup_percent": [100 * (x["thread_cpu_paired_median_speedup"] - 1) for x in pair],
                        "raw_wall_pairs": [x["paired_speedups"] for x in pair], "raw_thread_cpu_pairs": [x["thread_cpu_paired_speedups"] for x in pair],
                        "passes_practical_screen_both_sockets": survives,
                        "engine_adoption": False})
    output = {"scope": "Two physical cores on different sockets, sequential single-thread synthetic screens, memory local to each socket; not 12-worker or whole-model performance.",
              "sources": sources, "correctness_unique_family": {"bit_exact_outputs": 61440, "unsafe_rejections": 4800, "saved_bias_checks": 20480},
              "abs_cases": 12, "abs_cases_screen_pass_both": sum(x["passes_practical_screen_both_sockets"] for x in results if x["variant"] == "experimental_input_abs"),
              "rows": results, "limits": ["Single input; no six-input down path", "Rows256/8192, not actual M2304/5120", "No actual GGUF weights", "OS not isolated; API resident and only checked idle before/after", "No new generation rate or quality measurement", "Bias62 comparison has separate paired blocks, not a direct randomized abs-vs-bias comparison", "Four retained pair ratios per socket; no statistical confidence interval claimed"],
              "next": "Discuss an actual-shape, multiworker measurement and narrow integration proposal with GPT Pro before engine changes."}
    (ROOT / "CPU_SCREEN_RESULTS.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    lines = ["type\tvariant\tK\trows\tsocket0_wall_gain_pct\tsocket1_wall_gain_pct\tscreen_both\n"]
    for row in results:
        lines.append(f"{row['type']}\t{row['variant']}\t{row['K']}\t{row['rows']}\t{row['wall_speedup_percent'][0]:.4f}\t{row['wall_speedup_percent'][1]:.4f}\t{row['passes_practical_screen_both_sockets']}\n")
    (ROOT / "CPU_SCREEN_RESULTS.tsv").write_text("".join(lines))
    print(json.dumps({"abs_cases_pass": output["abs_cases_screen_pass_both"], "abs_cases": 12, "engine_changed": False}))


if __name__ == "__main__":
    main()
