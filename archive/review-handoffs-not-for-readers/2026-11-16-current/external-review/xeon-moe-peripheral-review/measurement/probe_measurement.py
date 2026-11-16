#!/usr/bin/env python3
"""Reproduce known behavior of one fixed archived revision with synthetic inputs.

No model, GPU, service, perf command, network request, or tracked repo mutation.
The C++ compilation below includes only a copied standard-library median helper.
This is an old-behavior reproducer, NOT a post-fix acceptance or regression gate.
Both --repo and a NEW --output directory are required. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

EXPECTED_COMMIT = "4ff68eba2d8c480692511dc6d219d27a6a102866"
EXPECTED_SOURCE_SHA256 = {
    "decode_analyzer": "9b1d5485c3b0bbdc537d1a0a92bb50ab16b075d03a306c53bb4991fef4c9458d",
    "postcompact_analyzer": "9b1d5485c3b0bbdc537d1a0a92bb50ab16b075d03a306c53bb4991fef4c9458d",
    "prefill_summary": "d82801a05c56ff08cd6f7bf94057f1c1a893bd41cd952b2ddd5300863270696a",
    "operator_probe": "7495c5818feef42ea99d30b6b3fda1718a09ad5bdeb17c6dc1de20626c7fa8ad",
    "prefill_live_analyzer": "c2e2336dbf528a6474a206c1c5772e6ccf5dc3ec23a576ea30513b0fd2b93144",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def line(ns: int, name: str, args: str = "") -> str:
    return f"100/100 {ns // 10**9}.{ns % 10**9:09d}: synthetic:{name}: {args}"


def decode_trace(omit: str = "none") -> str:
    records = []
    for step in range(3):
        base = 1_000_000_000 + step * 100_000_000
        records.append((base, "decode", "n_tokens=1"))
        records.append((base + 10_000_000, "cpu", 'first="ffn_moe_gate-0"'))
        if omit != "all" and not (omit == "one" and step == 1):
            records.append((base + 50_000_000, "cpu_ret", ""))
        records.append((base + 80_000_000, "decode_ret", ""))
    return "\n".join(line(*item) for item in sorted(records)) + "\n"


def trace_probe(source: Path, root: Path) -> dict:
    spec = importlib.util.spec_from_file_location("synthetic_trace_analyzer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for name, omission in [("baseline", "none"), ("all_cpu_returns_missing", "all"),
                           ("one_cpu_return_missing_control", "one")]:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "trace.txt").write_text(decode_trace(omission))
        try:
            analysis = module.analyze(folder)
            result[name] = {"accepted": True, "steps": analysis["steps"],
                            "CPU_wall_percent": analysis["exclusive_host_time"]["CPU_graph"]["wall_percent"],
                            "other_wall_percent": analysis["exclusive_host_time"]["other"]["wall_percent"],
                            "unmatched_inside_ROI": analysis["unmatched_inside_ROI"],
                            "event_counts": analysis["event_counts"]}
        except Exception as exc:
            result[name] = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
    assert result["baseline"]["CPU_wall_percent"] == 40
    assert result["all_cpu_returns_missing"]["accepted"]
    assert result["all_cpu_returns_missing"]["CPU_wall_percent"] == 0
    assert result["all_cpu_returns_missing"]["unmatched_inside_ROI"] == 0
    assert not result["one_cpu_return_missing_control"]["accepted"]
    return result


def run_script(path: Path, *args: str) -> dict:
    process = subprocess.run([sys.executable, "-B", str(path), *args],
                             cwd=path.parent, capture_output=True, text=True, timeout=10)
    return {"returncode": process.returncode, "stdout": process.stdout,
            "stderr": process.stderr}


def summary_probe(source: Path, root: Path) -> dict:
    result = {}
    specifications = {
        "empty_suite": [],
        "validation_failure_before_timing": [("failed", 1, "numerical check failed", False, False)],
        "one_good_one_failed_before_timing": [("good", 0, None, True, True),
                                              ("failed", 1, "numerical check failed", False, False)],
        "failed_with_timing_control": [("failed", 1, "synthetic post-timing failure", False, True)],
    }
    for name, cases in specifications.items():
        folder = root / name
        operators = folder / "operators"
        operators.mkdir(parents=True)
        shutil.copyfile(source, folder / "summarize.py")
        for case, code, error, passed, timing in cases:
            target = operators / case
            target.mkdir()
            write_json(target / "RESULT.json", {"exit_code": code, "error": error})
            text = "VALIDATION " + json.dumps({"pass": passed}) + "\n"
            if timing:
                text += "TIMING " + json.dumps({"stock_ms": 1.1, "candidate_ms": 1.0,
                                               "speedup": 1.1, "samples_each": 10}) + "\n"
            (target / "stdout.txt").write_text(text)
        process = run_script(folder / "summarize.py")
        process["attempted_cases"] = len(cases)
        process["failed_receipts"] = sum(c[1] != 0 or bool(c[2]) for c in cases)
        if (folder / "SUMMARY.json").exists():
            summary = json.loads((folder / "SUMMARY.json").read_text())
            process["timing_cases"] = summary["timing_cases"]
            process["all_numeric_checks_passed"] = summary["all_numeric_checks_passed"]
        result[name] = process
    for name in list(specifications)[:3]:
        assert result[name]["returncode"] == 0 and result[name]["all_numeric_checks_passed"]
    assert result["failed_with_timing_control"]["returncode"] != 0
    return result


def median_probe(source: Path, root: Path) -> dict:
    helper = next(s for s in source.read_text().splitlines()
                  if s.startswith("static double median(std::vector<double> x)"))
    root.mkdir(parents=True)
    program = "\n".join([
        "#include <algorithm>", "#include <vector>", "#include <cstdio>", helper,
        "int main() {",
        "  std::vector<double> a={10,11,12,13,14,20,21,22,23,24};",
        "  std::vector<double> b(10,18);",
        '  std::printf("{\\\"stock_reported\\\":%.17g,\\\"candidate_reported\\\":%.17g,\\\"reported_ratio\\\":%.17g}\\n", median(a),median(b),median(a)/median(b));',
        "}", ""])
    cpp = root / "median_helper_only.cpp"
    cpp.write_text(program)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("A standard C++ compiler is required for the helper-only probe")
    binary = root / "median_helper_only"
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    if build.returncode:
        raise RuntimeError(build.stderr)
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10, check=True)
    result = json.loads(run.stdout)
    stock = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    candidate = [18] * 10
    result.update({"copied_helper": helper, "stock_synthetic_samples": stock,
                   "candidate_synthetic_samples": candidate,
                   "stock_standard_median": statistics.median(stock),
                   "candidate_standard_median": statistics.median(candidate),
                   "standard_ratio": statistics.median(stock) / statistics.median(candidate),
                   "scope": "Artificial sample distribution; not a correction of any archived benchmark.",
                   "compiler": compiler, "compile_returncode": build.returncode})
    assert result["stock_reported"] == 20 and result["stock_standard_median"] == 17
    return result


def prefill_trace(cpu_ms: int) -> str:
    return "\n".join([
        line(1_000_000_000, "ubatch", "n_tokens=2048"),
        line(1_010_000_000, "cpu"),
        line(1_010_000_000 + cpu_ms * 1_000_000, "cpu_ret"),
        line(1_100_000_000, "ubatch_ret")]) + "\n"


def live_selection_probe(source: Path, root: Path) -> dict:
    result = {}
    specifications = {
        "only_deep_run_default_command": [("100-trace-deep-n2048", 80)],
        "normal_then_deep": [("100-trace-n2048", 40), ("200-trace-deep-n2048", 80)],
        "deep_then_normal": [("200-trace-deep-n2048", 80), ("100-trace-n2048", 40)],
    }
    for name, runs in specifications.items():
        folder = root / name
        live = folder / "live"
        live.mkdir(parents=True)
        shutil.copyfile(source, folder / "analyze_live.py")
        for run_name, cpu_ms in runs:
            target = live / run_name
            target.mkdir()
            (target / "trace.txt").write_text(prefill_trace(cpu_ms))
        process = run_script(folder / "analyze_live.py")
        process["directory_iteration_order"] = [p.name for p in live.iterdir()]
        if (folder / "LIVE_SUMMARY.json").exists():
            analysis = json.loads((folder / "LIVE_SUMMARY.json").read_text())
            process["wall_ms"] = analysis["wall_ms"]
            process["cpu_ms"] = analysis["exclusive_host_time"]["cpu"]["ms"]
            process["summary_keys"] = list(analysis)
        result[name] = process
    assert result["only_deep_run_default_command"]["returncode"] == 0
    assert result["only_deep_run_default_command"]["cpu_ms"] == 80
    return result


def archived_inventory(study: Path) -> dict:
    receipts = []
    for folder in sorted((study / "operators").iterdir()):
        receipt = json.loads((folder / "RESULT.json").read_text())
        receipts.append({"folder": folder.name, "exit_code": receipt.get("exit_code"),
                         "error": receipt.get("error"), "stdout_exists": (folder / "stdout.txt").exists()})
    summary = json.loads((study / "SUMMARY.json").read_text())
    return {"operator_receipts": len(receipts),
            "failed_receipts": [r for r in receipts if r["exit_code"] != 0 or r["error"]],
            "stdout_files_available": sum(r["stdout_exists"] for r in receipts),
            "summary_timing_cases": summary["timing_cases"],
            "samples_each_values": sorted({r["samples_each"] for r in summary["timing_results"]}),
            "live_raw_directory_available": (study / "live").exists(),
            "receipts": receipts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True,
                    help=f"Repository checkout at HEAD {EXPECTED_COMMIT}; source hashes must also match.")
    ap.add_argument("--output", type=Path, required=True,
                    help="New output directory; must not exist, even if empty.")
    args = ap.parse_args()
    repo = args.repo.resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"],
                                         cwd=repo, text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        ap.error(f"Cannot inspect --repo HEAD: {exc}")
    if commit != EXPECTED_COMMIT:
        ap.error(f"Unsupported HEAD {commit}; this old-behavior reproducer requires {EXPECTED_COMMIT}.")
    lab = repo / "archive/development-history-not-for-readers/lab"
    study = lab / "gpu-prefill-study-20261104"
    sources = {"decode_analyzer": lab / "decode-observation-20261103/analyze_trace.py",
               "postcompact_analyzer": lab / "postcompact-20261103/analyze_trace.py",
               "prefill_summary": study / "summarize.py",
               "operator_probe": study / "operator_probe.cpp",
               "prefill_live_analyzer": study / "analyze_live.py"}
    source_hashes = {}
    for name, path in sources.items():
        try:
            source_hashes[name] = digest(path)
        except OSError as exc:
            ap.error(f"Cannot read required source {path}: {exc}")
        if source_hashes[name] != EXPECTED_SOURCE_SHA256[name]:
            ap.error(f"Source hash mismatch for {path}; modified analyzers must not be run by this reproducer.")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        ap.error(f"--output already exists: {output}. Choose a new directory; existing results are never overwritten.")
    except OSError as exc:
        ap.error(f"Cannot create --output {output}: {exc}")
    root = Path(tempfile.mkdtemp(prefix="synthetic-fixtures-", dir=output))
    result = {"scope": "Synthetic parser/aggregation inputs only; no engine benchmark.",
              "purpose": "Reproduce known old behavior; not a post-fix acceptance or regression gate.",
              "repo_commit": commit,
              "fixture_root": str(root),
              "source_files": {k: {"path": str(v), "sha256": source_hashes[k]} for k, v in sources.items()},
              "trace_parser": trace_probe(sources["decode_analyzer"], root / "decode"),
              "postcompact_trace_parser": trace_probe(sources["postcompact_analyzer"], root / "postcompact"),
              "prefill_summary": summary_probe(sources["prefill_summary"], root / "summary"),
              "median_convention": median_probe(sources["operator_probe"], root / "median"),
              "prefill_live_selection": live_selection_probe(sources["prefill_live_analyzer"], root / "live_selection"),
              "archived_inventory": archived_inventory(study)}
    destination = output / "PROBE_RESULTS.json"
    write_json(destination, result)
    print(json.dumps({"results": str(destination), "fixtures": str(root),
                      "probe_checks_completed": True, "model_or_service_started": False,
                      "is_post_fix_acceptance_gate": False}, indent=2))


if __name__ == "__main__":
    main()
