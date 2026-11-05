#!/usr/bin/env python3
"""Matched serial confirmation of the guarded AVX Q8 four-column kernel."""
import fcntl
import json
import re
import statistics

from glm_q8_experiment import LAB, OUT, run
from run_glm_q8_observations import QUESTION

CANDIDATE = str(LAB / "review-glm-q8-columns4-20261020/build")


def timing(text, name):
    match = re.search(
        rf"(?:common_perf_print|llama_perf_context_print):\s+{name} time\s*=\s*"
        r"([\d.]+) ms /\s*(\d+)\s+(?:tokens|runs).*?\(\s*[\d.]+ ms per token,\s*"
        r"([\d.]+) tokens per second",
        text,
    )
    if not match:
        raise RuntimeError(f"Timing missing: {name}")
    ms, count, rate = match.groups()
    return {"ms": float(ms), "count": int(count), "tok_s_printed": float(rate),
            "tok_s": int(count) * 1000 / float(ms)}


def main():
    state = {"complete": False, "prefill": [], "responses": []}
    destination = OUT / "column-confirmation.json"

    def save():
        destination.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists() and json.loads(destination.read_text()).get("complete"):
            raise RuntimeError("Confirmation is already complete")
        check = json.loads((OUT / "validation-review-glm-q8-columns4-20261020/comparison.json").read_text())
        assert check["complete"] and check["bit_identical"]
        save()
        for index, variant in enumerate(("B", "C", "C", "B")):
            label = f"columns-prefill-{index}-{variant}"
            prior = OUT / label / "run.json"
            if prior.exists():
                result = json.loads(prior.read_text())
                assert result["exit_code"] == 0 and result["benchmarks"]
            else:
                result = run(label, build="B" if variant == "B" else CANDIDATE,
                             threads="24", prompt=128, tokens=0, repetitions=3)
            state["prefill"].append({"label": label, "variant": variant,
                                     "benchmark": result["benchmarks"][0]})
            save()
        reference = None
        for index, variant in enumerate(("B", "C", "C", "B")):
            label = f"columns-response-{index}-{variant}"
            folder = OUT / label
            if (folder / "run.json").exists():
                assert json.loads((folder / "run.json").read_text())["exit_code"] == 0
            else:
                run(label, build="B" if variant == "B" else CANDIDATE, threads="24",
                    tokens=192, response=QUESTION)
            answer = (folder / "stdout.txt").read_bytes()
            if reference is None:
                reference = answer
            if answer != reference:
                raise RuntimeError("Deterministic Japanese response differs")
            stderr = (folder / "stderr.txt").read_text()
            state["responses"].append({
                "label": label, "variant": variant, "identical": True,
                "prompt": timing(stderr, "prompt eval"),
                "generation": timing(stderr, "eval"),
            })
            save()
        summary = {}
        for variant in ("B", "C"):
            pp = [r["benchmark"]["avg_ts"] for r in state["prefill"] if r["variant"] == variant]
            responses = [r for r in state["responses"] if r["variant"] == variant]
            summary[variant] = {
                "pp128_tok_s_median": statistics.median(pp),
                "response_prompt_ms_median": statistics.median(r["prompt"]["ms"] for r in responses),
                "response_generation_ms_median": statistics.median(r["generation"]["ms"] for r in responses),
                "response_prompt_tok_s_median": statistics.median(r["prompt"]["tok_s"] for r in responses),
                "response_generation_tok_s_median": statistics.median(r["generation"]["tok_s"] for r in responses),
                "response_eval_ms_median": statistics.median(
                    r["prompt"]["ms"] + r["generation"]["ms"] for r in responses),
            }
        state["summary"] = summary
        state["complete"] = True
        save()
        print(json.dumps({"complete": True, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
