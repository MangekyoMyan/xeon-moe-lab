"""Independent read-only checks of a pinned validation3 handoff.

This script never imports or executes repository code, generated answers, or
recorded launch commands. Output is written only to the requested audit folder.
"""
import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    root = args.archive.resolve()
    result = {"scope": "Parent independent offline recomputation; no engine or generated Python executed", "timing": [], "quality": [], "answers": {}}
    read = lambda p: json.loads(p.read_text())
    plan = read(root / "caller/FROZEN_PLAN.json")
    cases = {c["case_id"]: c for c in plan["cases"]}
    quality_paths = defaultdict(dict)
    answer_data = defaultdict(dict)
    for series, base in (("initial", root), ("python_repeat", root / "python-repeat")):
        blocks = defaultdict(list)
        for runpath in sorted((base / "caller/runs").glob("*/RUN.json")):
            run = read(runpath)
            for b in run["blocks"]:
                if run["stage"] == "quality":
                    quality_paths[b["case_id"]][b["mode"]] = runpath.parent / b["file"]
                if run["stage"] != "timing":
                    continue
                durations = [int(e) - int(s) for s, e in zip(b["start_ns"], b["output_ready_ns"])]
                assert len(durations) == 64 and all(n > 0 for n in durations)
                assert b["positions"] == list(range(b["prefix_count"], b["prefix_count"] + 64))
                blocks[b["case_id"]].append({"mode": b["mode"], "process_index": run["spec"]["process_index"], "duration_ns": durations, "positions": b["positions"], "raw_run": str(runpath.relative_to(root)), "mean_ms": sum(durations) / 64e6})
            for a in run["answers"]:
                answer_data[a["case_id"]][a["mode"]] = a
        for case_id, rows in blocks.items():
            rows.sort(key=lambda r: r["process_index"])
            assert [r["mode"] for r in rows] == list("FPPF")
            total = {m: sum(sum(r["duration_ns"]) for r in rows if r["mode"] == m) for m in "FP"}
            result["timing"].append({"series": series, "case_id": case_id, "calls_per_mode": 128, "F_mean_ms": total["F"] / 128e6, "P_mean_ms": total["P"] / 128e6, "F_T1_s": 128e9 / total["F"], "P_T1_s": 128e9 / total["P"], "gain_pct": (total["F"] / total["P"] - 1) * 100, "time_reduction_pct": (1 - total["P"] / total["F"]) * 100, "saved_ms": (total["F"] - total["P"]) / 128e6, "blocks": rows})
    for case_id, paths in quality_paths.items():
        f, s, p = [np.memmap(paths[m], mode="r", dtype="<f4", shape=(65, 129280)) for m in "FSP"]
        assert all(np.isfinite(x).all() for x in (f, s, p))
        assert s.tobytes() == p.tobytes()
        targets = cases[case_id]["execution_input"]["teacher_forced_next_targets65"]
        row_results = []
        for n, target in enumerate(targets):
            x, y = f[n].astype(np.float64), p[n].astype(np.float64)
            x0, y0 = x - x.max(), y - y.max()
            lx = x0 - np.log(np.exp(x0).sum())
            ly = y0 - np.log(np.exp(y0).sum())
            px, py = np.exp(lx), np.exp(ly)
            d = y - x
            tx, ty = np.argsort(x)[-2:][::-1], np.argsort(y)[-2:][::-1]
            worst = int(np.argmax(np.abs(d)))
            row_results.append({"row": n, "target": target, "F_top1": int(tx[0]), "P_top1": int(ty[0]), "F_top1_top2_margin": float(x[tx[0]] - x[tx[1]]), "P_top1_top2_margin": float(y[ty[0]] - y[ty[1]]), "F_top1_probability": float(px[tx[0]]), "P_top1_probability": float(py[ty[0]]), "target_NLL_P_minus_F": float(lx[target] - ly[target]), "KL_F_to_P": float(px @ (lx - ly)), "total_variation": float(np.abs(px - py).sum() / 2), "mean_logit_offset_P_minus_F": float(d.mean()), "centered_RMSE": float(np.sqrt(np.mean((d - d.mean()) ** 2))), "max_abs": float(np.abs(d).max()), "max_abs_vocab_id": worst, "F_probability_at_max_abs": float(px[worst]), "P_probability_at_max_abs": float(py[worst])})
        result["quality"].append({"case_id": case_id, "S_P_all_bytes_equal": True, "F_P_byte_equal_rows": [i for i in range(65) if f[i].tobytes() == p[i].tobytes()], "F_P_top1_different_rows": [r["row"] for r in row_results if r["F_top1"] != r["P_top1"]], "mean_NLL_difference": float(np.mean([r["target_NLL_P_minus_F"] for r in row_results])), "mean_KL_F_to_P": float(np.mean([r["KL_F_to_P"] for r in row_results])), "max_KL_F_to_P": float(max(r["KL_F_to_P"] for r in row_results)), "mean_total_variation": float(np.mean([r["total_variation"] for r in row_results])), "max_total_variation": float(max(r["total_variation"] for r in row_results)), "rows": row_results})
    for case_id, modes in answer_data.items():
        ids_f, ids_p = [modes[m]["generated_ids"] for m in "FP"]
        common = next((i for i, (f, p) in enumerate(zip(ids_f, ids_p)) if f != p), min(len(ids_f), len(ids_p)))
        result["answers"][case_id] = {"first_difference_zero_based": common, "modes": {m: {k: modes[m][k] for k in ("generated_count", "decoded_T1_calls", "stop_reason", "reference_supplied")} for m in "FP"}}
    code_dir = root / "answers/code_bounded_queue_python"
    trees = {m: ast.parse((code_dir / f"{m}.py").read_text()) for m in "FP"}
    methods = {m: {n.name: n for n in trees[m].body[0].body if isinstance(n, ast.FunctionDef)} for m in "FP"}
    result["python_static_review"] = {
        "source_sha256": {m: hashlib.sha256((code_dir / f"{m}.py").read_bytes()).hexdigest() for m in "FP"},
        "same_non_init_AST": {name: ast.dump(methods["F"][name]) == ast.dump(methods["P"][name]) for name in methods["F"] if name != "__init__"},
        "generated_code_executed_by_this_review": False,
        "observed_init_difference": "Only _buf and _capacity assignment order; inspected manually under prompt integer/capacity preconditions.",
        "sequential_correctness_argument": "For 0<=count<=capacity, live queue slots are (head+i)%capacity for 0<=i<count. A successful push writes at i=count; successful pop reads i=0 then advances head and decrements count. Full/empty rejection precedes mutation. This establishes the requested sequential FIFO invariant, not concurrent atomicity.",
        "minor_F_explanation_caveat": "F.txt line57 overstates never partially updated if read as a concurrency/atomicity guarantee. Concurrency was not requested, and the queue code need not be changed for this task."
    }
    result["language_review"] = {"frozen_template_no_system_message": plan["frozen_token_global_metadata_exact"]["no_system_message"], "JA_frozen_prompt_explicit_answer_language": False, "JA_answers": "Both observed Chinese from the opening. Full response read manually; this is a local DeepSeek answer observation, not an Astra language test.", "UI_rendering_tested": False}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "PARENT_INDEPENDENT_RECOMPUTATION.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target), "timing": [{k: v for k, v in r.items() if k != "blocks"} for r in result["timing"]], "quality": [{k: v for k, v in r.items() if k != "rows"} for r in result["quality"]], "answers": result["answers"], "static": result["python_static_review"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
