#!/usr/bin/env python3
"""Offline source/receipt audit only; never import or run repository code."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess
import sys

PIN = "d4d50b651ff7754ee2189228762468fc02971bfa"
PREFIX = "archive/review-handoffs-not-for-readers/"
N = PREFIX + "2026-11-17-parallel-pilot"
I = N + "/integrated"
R = PREFIX + "2026-11-17-route-integration"
H = PREFIX + "2026-11-16-current/snapshot/llm-lab"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo, out = (Path(x).resolve() for x in sys.argv[1:3])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != PIN:
        raise SystemExit("Unexpected source commit")
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    receipts = {}
    for receipt in ["BUILD.json", "ENGINE_BUILD.json"]:
        rel = I + "/" + receipt
        value = json.loads((repo/rel).read_text())
        receipts[receipt] = value
        for name, expected in value["source_sha256"].items():
            path = repo/I/name
            actual = digest(path)
            checks.append({"receipt": rel, "source": I+"/"+name,
                           "expected": expected, "actual": actual,
                           "matches": actual == expected})
    excerpts = {
        I+"/hot-backend.cpp": [(44,63),(74,95),(122,164),(166,207),(230,263)],
        I+"/driver.cpp": [(81,89),(128,159)],
        I+"/llama-graph.cpp": [(2300,2332),(2504,2520)],
        I+"/route37/llama-context.cpp": [(4317,4330)],
        N+"/capture/bounded_moe_capture_proposal.h": [(109,190)],
        R+"/route37/route37.cpp": [(7,34),(65,99)],
        R+"/route37/ggml-cpu.c": [(1826,1851)],
        H+"/gpu-stream-20261103/ggml-backend.cpp": [(1417,1455),(2095,2134),(2140,2215),(2230,2238)],
        H+"/exploration-20261112/cuda-sort/release-candidate-a/ggml-cuda.cu": [(3494,3513),(3566,3581),(3617,3636),(3959,3969)],
    }
    evidence = []
    hashes = []
    for rel, spans in excerpts.items():
        data = (repo/rel).read_text().splitlines()
        sha = digest(repo/rel)
        hashes.append({"source": rel, "sha256": sha, "lines": len(data)})
        for first, last in spans:
            if last > len(data):
                raise SystemExit("Excerpt exceeds file: " + rel)
            evidence.append({"source": rel, "sha256": sha, "first": first, "last": last,
                             "lines": [{"n": n, "text": data[n-1]} for n in range(first,last+1)]})
    diff_pairs = [
        (R+"/route37/llama-context.cpp", I+"/route37/llama-context.cpp", "context_integration.diff"),
        (R+"/provenance/model-source/src/llama-graph.cpp", I+"/llama-graph.cpp", "graph_integration.diff"),
        (N+"/replay/hot-backend.cpp", I+"/hot-backend.cpp", "replay_to_integration.diff"),
        (I+"/checkpoint-engine-before-C-stride/hot-backend.cpp", I+"/hot-backend.cpp", "c_stride_fix.diff"),
        (N+"/capture/checkpoint-build2/bounded_moe_capture_proposal.h", N+"/capture/bounded_moe_capture_proposal.h", "capture_build2_to_final.diff"),
    ]
    diffs = []
    for a,b,name in diff_pairs:
        text = ''.join(difflib.unified_diff((repo/a).read_text().splitlines(True),
                                          (repo/b).read_text().splitlines(True),fromfile=a,tofile=b))
        (out/name).write_text(text)
        diffs.append({"old": a, "new": b, "old_sha256": digest(repo/a),
                      "new_sha256": digest(repo/b), "artifact": name,
                      "artifact_sha256": hashlib.sha256(text.encode()).hexdigest()})
    reserve = []
    pattern = re.compile(r"nodes = (\d+) / (\d+), splits = (\d+) / (\d+), input objects = (\d+) / (\d+), input tensors = (\d+) / (\d+)")
    keys = ["pp_nodes", "t1_nodes", "pp_splits", "t1_splits", "pp_input_objects", "t1_input_objects", "pp_input_tensors", "t1_input_tensors"]
    for path in sorted((repo/I/"runs").glob("*/server.stderr.log")):
        # Unrelated model vocabulary log bytes need not be valid UTF-8.
        # Select the ASCII reservation line from raw LF-delimited bytes.
        for n,raw in enumerate(path.read_bytes().split(b"\n"),1):
            if b"sched_reserve: graph (pp bs=1024, tg bs=1)" in raw:
                line = raw.rstrip(b"\r").decode("ascii")
                match = pattern.search(line)
                if not match:
                    raise SystemExit("Unexpected reserve log format")
                reserve.append({"source": str(path.relative_to(repo)), "line": n,
                                "sha256": digest(path), "text": line,
                                **dict(zip(keys,map(int,match.groups())))})
    if len(reserve) != 3:
        raise SystemExit("Expected exactly three integrated reserve records")
    result = {
        "commit": head, "scope": "offline supplied-source/receipt/log extraction only",
        "executed_supplied_code": False, "build_or_engine_run": False,
        "receipt_source_checks": checks, "all_receipt_sources_match": all(x["matches"] for x in checks),
        "source_hashes": hashes, "reserve_graph_records": reserve,
        "reserve_scope_caveat": "Reservation geometry, not a runtime split/copy trace for each measured decode",
        "diffs": diffs,
        "reported_library_hashes": receipts["BUILD.json"]["engine_libraries"],
        "reported_libllama_sha256": receipts["ENGINE_BUILD.json"]["libllama_sha256"],
    }
    (out/"SOURCE_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (out/"SOURCE_EXCERPTS.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"commit": head, "source_checks": len(checks),
                      "all_match": result["all_receipt_sources_match"],
                      "reserve": reserve, "diffs_written": len(diffs)},ensure_ascii=False,indent=2))
    if not result["all_receipt_sources_match"]:
        raise SystemExit("Receipt source mismatch; investigate before using report")


if __name__ == "__main__":
    main()
