#!/usr/bin/env python3
"""Check short and long Japanese answers with the baseline and rebuilt candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=Path, required=True)
args = parser.parse_args()
lab = Path(__file__).resolve().parent
output = lab / "results/deep-review-functional.json"
if output.exists():
    raise SystemExit("Preserve the existing acceptance record")
baseline = lab / "review-deep-pools-20261019/baseline/build/bin/llama-completion"
env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1",
           GGML_CPU_NUMA_REPLICA="0", GGML_CPU_MOE_PAIR_TRACE="0", GGML_CPU_NUMA_REPLICA_TRACE="0")
runs = []
for size, repetitions, variant in (("short59", 0, "baseline"), ("short59", 0, "candidate"),
                                   ("long971", 48, "candidate"), ("long971", 48, "baseline")):
    label = "deep-review-" + size + "-" + variant
    command = [sys.executable, str(lab / "verify_deepseek.py"), "--binary",
        str(baseline if variant == "baseline" else args.candidate.resolve()), "--label", label,
        "--threads", "12", "--threads-batch", "24", "--batch", "512", "--ubatch", "512",
        "--note-repetitions", str(repetitions)]
    print("starting " + label, flush=True)
    with (lab / "results" / (label + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    matches = sorted((lab / "results").glob("deepseek-functional-" + label + "-*/summary.json"))
    if len(matches) != 1:
        raise RuntimeError("Expected one fresh functional result for " + label)
    result = json.loads(matches[0].read_text())
    response = result["response"]
    token_match = re.search(r"prompt eval time\s*=.*?/\s*(\d+) tokens", "\n".join(result["timing_lines"]))
    runs.append({"variant": variant, "size": size, "summary": str(matches[0]),
                 "exit_code": result["exit_code"], "response": response,
                 "content_checks_passed": result["semantic_checks_passed"],
                 "prompt_tokens": int(token_match[1]) if token_match else None,
                 "timings_ms_single_run_only": result["timings_ms"],
                 "response_sha256": hashlib.sha256(response.encode()).hexdigest()})
    output.write_text(json.dumps({"runs": runs, "complete": False}, ensure_ascii=False, indent=2) + "\n")
checks = {}
for size in ("short59", "long971"):
    pair = [row for row in runs if row["size"] == size]
    checks[size] = len({r["response_sha256"] for r in pair}) == 1
report = {"runs": runs, "complete": True, "responses_identical_by_size": checks,
          "all_passed": all(checks.values()) and all(r["content_checks_passed"] for r in runs),
          "scope": "Two fixed Japanese prompts, greedy decoding, reasoning off. Timings are single-run observations, not statistically established speed gains. Content assertions do not enforce one-line formatting."}
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if not report["all_passed"]:
    raise SystemExit("Review response differences before adoption")
