#!/usr/bin/env python3
"""Read-only extraction of small evidence slices. Never imports project code."""
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
pin = "d98a17c56fadc92cf79e21ecbe80732dedeed899"
N = "archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace"
R = "archive/review-handoffs-not-for-readers/2026-11-17-route-integration"
U = "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/deepseek41-bringup-20261101/source"
H = "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab"
spec = [
    ("input_is_ffn_norm", f"{R}/provenance/model-source/src/models/deepseek41.cpp", 861, 909),
    ("weights_are_normed_scaled_in_original_order", f"{R}/provenance/model-source/src/llama-graph.cpp", 2247, 2304),
    ("gate_input_and_clamp", f"{R}/provenance/model-source/src/llama-graph.cpp", 2336, 2400),
    ("down_weights_sum", f"{R}/provenance/model-source/src/llama-graph.cpp", 2451, 2528),
    ("callback_name", f"{R}/route37/llama-context.cpp", 2886, 2893),
    ("callback_contract", f"{U}/ggml/include/ggml-backend.h", 314, 322),
    ("callback_sync", f"{H}/gpu-stream-20261103/ggml-backend.cpp", 2170, 2211),
    ("cpu_original_input_quantization", f"{R}/route37/ggml-cpu.c", 1780, 1860),
    ("cpu_fusion_scope", f"{R}/route37/ggml-cpu.c", 3296, 3334),
    ("graph_reuse_callback_install", f"{R}/route37/llama-context.cpp", 1528, 1607),
    ("generation_and_gate_identity", f"{R}/route37/route37.cpp", 40, 88),
    ("raw_mask_dimension_reuse", f"{R}/provenance/model-source/src/llama-graph.cpp", 869, 887),
    ("all_dsv4_input_reuse", f"{R}/provenance/model-source/src/llama-graph.cpp", 1051, 1078),
    ("kv_padding", f"{U}/src/llama-kv-cache.cpp", 1311, 1325),
    ("raw_context_kv_update", f"{U}/src/llama-kv-cache-dsv4.cpp", 1996, 2025),
    ("compressed_plan_padding", f"{U}/src/llama-kv-cache-dsv4.cpp", 674, 683),
    ("capacity_helper_clamp", f"{N}/bank-capacity/hot-backend.cpp", 27, 44),
    ("baseline_call_geometry", f"{N}/pilot-capture/driver.cpp", 144, 178),
]
evidence = []
for title, name, first, last in spec:
    data = (repo / name).read_bytes()
    lines = data.decode().splitlines()
    evidence.append(dict(title=title, path=name, file_sha256=hashlib.sha256(data).hexdigest(),
                         first_line=first, last_line=last,
                         url=f"https://github.com/MangekyoMyan/xeon-moe-lab/blob/{pin}/{name}#L{first}-L{last}",
                         excerpt="\n".join(f"{i}: {lines[i-1]}" for i in range(first, min(last,len(lines))+1))))

rundir = repo / N / "pilot-capture/runs/20261117T035402Z-U1024-d22b3e04"
cases = ["text_cache_budget_ja", "text_cache_budget_en", "mixed_build_log_ja_en_log",
         "code_stable_grouping_python", "code_stable_grouping_cpp"]
generation = []
for case in cases:
    p = rundir / (case + ".json")
    data = p.read_bytes()
    d = json.loads(data)
    a = d["T1_trace"]["attempts"]
    transitions = [dict(index=i, position=x["position"], generation=x["generation"])
                   for i, x in enumerate(a) if i == 0 or x["generation"] != a[i-1]["generation"]]
    generation.append(dict(case_id=case, prefix=d["prefix_tokens"],
                           path=str(p.relative_to(repo)), sha256=hashlib.sha256(data).hexdigest(),
                           transitions=transitions, count=len(a)))

long_generation = []
for sub in ("runs/20261117T031214Z-U1024-93e7fe85",
            "bank-capacity/runs/20261117T033327Z-U1024-0b6d5d01"):
    p = repo / N / sub / "long1024.json"
    d = json.loads(p.read_text())
    long_generation.append(dict(path=str(p.relative_to(repo)),
                                generation=[x["generation"] for x in d["T1_trace"]["attempts"]],
                                positions=[x["position"] for x in d["T1_trace"]["attempts"]]))
result = dict(pin=pin, method="stdlib read-only; no supplied scripts, model or engine execution",
              evidence=evidence, pilot_generation_transitions=generation,
              counterexample_to_general_256_rule=long_generation)
out.mkdir(parents=True, exist_ok=True)
(out / "activation_source_evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"slices":len(evidence), "pilot_generation":generation,
                  "long1024_generation":long_generation}, ensure_ascii=False, indent=2))
