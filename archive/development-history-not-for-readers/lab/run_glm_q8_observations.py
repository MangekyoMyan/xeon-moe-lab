#!/usr/bin/env python3
"""Observe GLM Q8 only after the matched baseline comparisons complete."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from glm_q8_experiment import LAB, OUT, MODEL, BUILDS, environment, run

QUESTION = (
    "日本語で、古い2ソケットXeonサーバーで大規模言語モデルを動かす場合に、"
    "メモリ容量とメモリ帯域がそれぞれ何に影響するかを説明してください。"
    "容量を増やすだけで必ず生成が速くなるわけではない理由も含め、"
    "初学者に分かるように200文字程度で答えてください。/nothink"
)


def main():
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((OUT / "stage1.json").read_text())
        assert state["complete"]
        threads = state["selected_threads"]
        run("response-B", threads=str(threads), tokens=192, response=QUESTION)
        run("prefill-B", threads="12,24", tokens=0, prompt="128,512", repetitions=2,
            timeout=3600)
        target = OUT / "observation-B"
        target.mkdir(exist_ok=False)
        common = ["--bench", str(BUILDS["B"] / "bin/llama-bench"),
                  "--model", str(MODEL), "--threads", str(threads), "--tokens", "128"]
        commands = [
            [sys.executable, str(LAB / "observe_glm_counters.py"), "decode", "stalls",
             *common, "--window", "18", "--output", str(target / "stalls")],
            *[[sys.executable, str(LAB / "observe_glm_phase.py"),
               *common, "--phase", "decode", "--mode", mode, "--cpus", "0-23",
               "--nodes", "0,1", "--seconds", "18", "--output", str(target / mode)]
              for mode in ("record", "memory")],
        ]
        report = {"complete": False, "commands": []}
        for command in commands:
            print("Diagnostic:", command[-1], flush=True)
            result = subprocess.run(command, env=environment(), check=False)
            report["commands"].append({"argv": command, "exit_code": result.returncode})
            (target / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError("Diagnostic failed")
        report["complete"] = True
        (target / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
