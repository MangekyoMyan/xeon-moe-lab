#!/usr/bin/env python3
"""Run a named, sequential experiment group through the thermal/PWM guard."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
lab = root.parent
base = lab / "v100-prep-20261024"
binary = base / "build-cuda/bin"
models = json.loads((root / "download-state.json").read_text())["models"]
assert all(m["status"] == "verified" for m in models)
qwen, glm = [m["path"] for m in models]
runtime = ["env", "LD_LIBRARY_PATH=" + ":".join(map(str, [root / "runtime", binary, base / "cuda-12.9.2/lib"]))]
local = ["python3", str(lab / "without_thp.py"), "numactl", "--physcpubind=6-11,18-23", "--membind=1"]
interleave = ["python3", str(lab / "without_thp.py"), "numactl", "--physcpubind=0-23", "--interleave=0,1"]
cases = []

def add(label, command, timeout=900, env=()):
    cases.append({"label": label, "command": list(map(str, command)), "timeout": timeout, "env": list(env)})

def bench(label, model, gpu=99, numa_local=True, p="512", n="128", repeats="3", extra=()):
    cpus, nth, mask = (local, "6", "0xfc0fc0") if numa_local else (interleave, "12", "0xffffff")
    add(label, runtime + cpus + [binary / "llama-bench", "-m", model, "-t", nth,
        "-b", "512", "-ub", "512", "--cpu-mask", mask, "--cpu-strict", "1", "--poll", "0",
        "--offline", "-lm", "mmap", "-lzm", "off", "--repack", "0", "-p", p, "-n", n,
        "-d", "0", "-r", repeats, "--progress", "-o", "json", "-ngl", str(gpu), "-fa", "auto",
        "-nopo", "1" if gpu == 0 else "0", *extra])

stage = sys.argv[1]
if stage == "loader-regression":
    for label, patched, mtp in [("qwen-original-mtp-loaded", False, 1),
                                 ("qwen-fixed-mtp-skipped", True, 0),
                                 ("qwen-fixed-mtp-loaded", True, 1)]:
        add(label, (runtime if patched else []) + local + [root / "model-probe", qwen, str(mtp), "99", "6", root / (label + ".f32")])
elif stage == "gpu-matrix":
    add("gpu-roofline-warm", [lab / "v100-performance-20261025/gpu-roofline", "3"], 600)
    for name, model in [("glm", glm), ("qwen", qwen)]:
        bench(name + "-gpu-local", model)
        bench(name + "-gpu-interleave", model, numa_local=False)
    bench("glm-cpu-only", glm, gpu=0, numa_local=False, p="32", n="32", repeats="2")
elif stage == "dispatch":
    for name, model in [("glm", glm), ("qwen", qwen)]:
        for variant, env in [("no-graphs", "GGML_CUDA_DISABLE_GRAPHS=1"),
                             ("graph-opt", "GGML_CUDA_GRAPH_OPT=1")]:
            bench(name + "-" + variant, model)
            cases[-1]["env"] = [env]
    add("quant-kernel-perf", local + [binary / "test-backend-ops", "perf", "-b", "CUDA0", "-o", "MUL_MAT",
        "-p", "type_a=(nvfp4|mxfp4|q8_0|q4_K),type_b=f32,m=4096,n=(1|8|512),k=14336,", "-j", "1"], 600)
    add("qwen-nvfp4-text", runtime + local + [binary / "llama-completion", "-m", qwen,
        "-t", "6", "-tb", "6", "-b", "128", "-ub", "128", "--offline", "-lm", "mmap", "--no-repack",
        "-c", "2048", "-n", "160", "-ngl", "99", "-fa", "auto", "--jinja", "-cnv", "-st",
        "-rea", "off", "--temp", "0", "--seed", "42", "--simple-io", "--no-display-prompt",
        "-p", "日本語で簡潔に答えてください。1) 17×23はいくつですか。2) RAMとVRAMの違いを二文で説明してください。3) Pythonで整数のリストの合計を返す関数を一つ書いてください。"])
elif stage == "sustained":
    bench("glm-gpu-matched32", glm, numa_local=False, p="32", n="32", repeats="3")
    for label, gpu, placement, nth in [("glm-logits-cpu", 0, interleave, 12),
                                       ("glm-logits-gpu", 99, local, 6)]:
        add(label, runtime + placement + [root / "model-probe", glm, "0", str(gpu), str(nth), root / (label + ".f32")])
    bench("qwen-context2048", qwen, p="0", n="0", repeats="2", extra=["-pg", "2048,128"])
    bench("glm-context4096", glm, p="0", n="0", repeats="3", extra=["-pg", "4096,128"])
    bench("glm-sustained512", glm, p="0", n="512", repeats="20")
else:
    raise ValueError(stage)

(root / (stage + "-commands.json")).write_text(json.dumps(cases, indent=2) + "\n")
for case in cases:
    from watch import gpu_sample
    cooldown_start = time.monotonic()
    while True:
        gpu = gpu_sample()
        if max(gpu["gpu_C"], gpu["hbm_C"]) <= 40:
            break
        assert time.monotonic() - cooldown_start < 300, "Cooldown timeout; leave workloads stopped"
        print("Cooling at Standard fan mode", gpu["gpu_C"], gpu["hbm_C"], flush=True)
        time.sleep(20)
    print("Starting", case["label"], flush=True)
    command = [sys.executable, str(root / "watch.py"), "--label", case["label"],
               "--timeout", str(case["timeout"]), "--env", "CUDA_VISIBLE_DEVICES=GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3"]
    for entry in case["env"]:
        command += ["--env", entry]
    command += ["--", *case["command"]]
    subprocess.run(command, check=True)
    with (root / "experiment-history.jsonl").open("a") as log:
        log.write(json.dumps({"unix": time.time(), "event": "experiment_complete", "label": case["label"]}) + "\n")
owner = lab.stat()
for path in root.iterdir():
    if path.is_file(): os.chown(path, owner.st_uid, owner.st_gid)
