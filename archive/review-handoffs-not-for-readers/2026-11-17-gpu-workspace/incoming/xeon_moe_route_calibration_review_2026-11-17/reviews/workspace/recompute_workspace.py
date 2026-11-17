#!/usr/bin/env python3
"""Read-only source/log audit plus arithmetic; never imports or runs project code.

Usage: python3 recompute_workspace.py REPO OUTDIR
Writes only its own OUTDIR. No model/runtime/library execution and no network I/O.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PIN = "fa6671f7b5281bad28947fe5324e2495e792a86f"
MIB = 1 << 20
H = "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/"
U = H + "deepseek41-bringup-20261101/source/"
R = "archive/review-handoffs-not-for-readers/2026-11-17-route-integration/"
C = "archive/review-handoffs-not-for-readers/2026-11-17-route-calibration/"

SOURCES = {
    "consult": (C + "CONSULT_PRO_JA.txt", [(1, 51)]),
    "results": (C + "RESULTS_JA.txt", [(1, 150)]),
    "build": (C + "BUILD.json", []),
    "launch": (C + "runs/20261117T012730Z-F-before-a30e72f7/LAUNCH_MANIFEST.json", []),
    "api_receipt": (C + "BASELINE_API_RECEIPT.json", []),
    "log": (C + "runs/20261117T012730Z-F-before-a30e72f7/server.stderr.log", [(1, 2), (26, 76), (3332, 3349), (3395, 3401), (3437, 3447), (3482, 3492), (3527, 3533), (3582, 3597), (5381, 5384), (5713, 5723)]),
    "driver": (C + "driver.cpp", [(26, 32), (111, 180), (185, 214)]),
    "old_hot_context_initialization": (H + "exploration-20261112/hot-ffn/llama-context.cpp", [(568, 580)]),
    "old_hot_bank_guards": (H + "exploration-20261112/hot-ffn/hot-backend.cpp", [(115, 155)]),
    "base_build": (H + "gpu-stream-20261103/SCHEDULER.json", [(1, 68)]),
    "scheduler": (H + "gpu-stream-20261103/ggml-backend.cpp", [(1329, 1339), (1381, 1399), (1744, 1760), (1784, 1816), (1879, 1899), (1927, 1976), (2089, 2175)]),
    "allocator": (U + "ggml/src/ggml-alloc.c", [(211, 250), (623, 711), (718, 823), (904, 963)]),
    "cuda_build": (H + "exploration-20261112/cuda-sort/release-candidate-a/BUILD.json", [(148, 158), (441, 485)]),
    "cuda": (H + "exploration-20261112/cuda-sort/release-candidate-a/ggml-cuda.cu", [(442, 574), (2201, 2268), (5080, 5128), (6215, 6234)]),
    "context": (R + "route37/llama-context.cpp", [(739, 852)]),
    "model": (R + "provenance/model-source/src/models/deepseek41.cpp", [(204, 216), (418, 454), (525, 630), (641, 743), (780, 828), (889, 932)]),
    "graph": (R + "provenance/model-source/src/llama-graph.cpp", [(848, 866), (997, 1024), (2158, 2168), (2302, 2305), (2336, 2378), (3811, 3829)]),
    "kv_iswa": (U + "src/llama-kv-cache-iswa.cpp", [(69, 105)]),
    "kv_dsv4": (U + "src/llama-kv-cache-dsv4.cpp", [(881, 903), (1245, 1307), (1390, 1427), (1501, 1577), (2153, 2171)]),
    "batch": (U + "src/llama-batch.cpp", [(619, 620), (682, 718)]),
    "ggml": (U + "ggml/src/ggml.c", [(5521, 5533), (6485, 6516)]),
    "types": (U + "ggml/src/ggml-common.h", [(214, 219), (395, 422)]),
    "topk_cuda": (U + "ggml/src/ggml-cuda/top-k.cu", [(212, 225), (253, 278), (281, 336)]),
    "argsort_cuda": (U + "ggml/src/ggml-cuda/argsort.cu", [(31, 42)]),
}


def sized(n: int) -> dict:
    return {"bytes": n, "MiB": n / MIB}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: recompute_workspace.py REPO OUTDIR")
    repo = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    if out == repo or repo in out.parents:
        raise SystemExit("OUTDIR must be outside the reviewed repository")
    out.mkdir(parents=True, exist_ok=True)
    receipts = {}
    extracts = []
    for key, (rel, spans) in SOURCES.items():
        payload = (repo / rel).read_bytes()
        try:
            decoded = payload.decode("utf-8")
            decode_note = "UTF-8 strict"
        except UnicodeDecodeError:
            decoded = payload.decode("utf-8", errors="replace")
            decode_note = "Non-UTF-8 bytes replaced in human-readable excerpts only; hash covers the exact original bytes"
        lines = decoded.splitlines()
        receipts[key] = {
            "repo_path": rel,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "line_count": len(lines),
            "excerpt_decode": decode_note,
            "github_url": f"https://github.com/MangekyoMyan/xeon-moe-lab/blob/{PIN}/{rel}",
            "excerpt_ranges": spans,
        }
        for start, end in spans:
            extracts.append(f"\n[{key}] {rel}:{start}-{min(end, len(lines))}\n")
            extracts.extend(f"{i + 1:5d} {lines[i]}\n" for i in range(start - 1, min(end, len(lines))))

    native = {}
    for typ, block_elems, block_bytes in [
        ("MXFP4", 32, 17), ("IQ2_S", 256, 82),
        ("IQ3_XXS", 256, 98), ("IQ3_S", 256, 110)
    ]:
        single = 5120 * 2304 // block_elems * block_bytes
        native[typ] = {
            "block_elems": block_elems, "block_bytes": block_bytes,
            "single_expert_projection": sized(single),
            "full_384_expert_projection": sized(single * 384),
        }

    shapes = []
    for b in (1024, 512, 256):
        raw_rows = ((min(131072, 128 + b) + 255) // 256) * 256
        tensors = {
            "SWA_raw_KV_all_40_layers": ("40 * 512 * raw_rows * sizeof(F16)", 40 * 512 * raw_rows * 2),
            "CSA_persistent_KV": ("3 source layers * 640 * 65536 * sizeof(F16)", 3 * 640 * 65536 * 2),
            "HCA_persistent_KV": ("1 source layer * 640 * 131072 * sizeof(F16)", 640 * 131072 * 2),
            "HCA_mask": ("F16[131072,B]", 131072 * b * 2),
            "CSA_mask": ("F16[65536,B]", 65536 * b * 2),
            "SWA_mask": ("F16[raw_rows,B]", raw_rows * b * 2),
            "HCA_indexer_score": ("F32[131072,B]", 131072 * b * 4),
            "CSA_indexer_score": ("F32[65536,B]", 65536 * b * 4),
            "HCA_candidate_pin": ("F32[ceil(131072/8),B]", 16384 * b * 4),
            "HCA_candidate_expanded_mask": ("F32[131072,B]", 131072 * b * 4),
            "top_k_indices": ("I32[512,B]", 512 * b * 4),
            "candidate_top_blocks": ("I32[2048,B]", 2048 * b * 4),
            "query_or_attention_output": ("F32[512,64,B]", 512 * 64 * b * 4),
            "hyper_connection_state": ("F32[5120,4,B]", 5120 * 4 * b * 4),
            "expert_gate_or_up_output": ("F32[2304,6,B]", 2304 * 6 * b * 4),
            "expert_down_output": ("F32[5120,6,B]", 5120 * 6 * b * 4),
            "HCA_dense_comp_K_contiguous": ("F16[512,1,131072]", 512 * 131072 * 2),
            "CSA_dense_comp_K_contiguous": ("F16[512,1,65536]", 512 * 65536 * 2),
            "top_k_radix_histograms_full_HCA": ("min(B,1024)*64*256*sizeof(I32); runtime pool, not graph arena", min(b, 1024) * 64 * 256 * 4),
        }
        shapes.append({
            "n_ubatch": b, "raw_KV_rows": raw_rows,
            "tensors": {k: {"formula": formula, **sized(n)} for k, (formula, n) in tensors.items()},
            "warning": "Individual tensor payloads, not additive arena peaks or measured free-memory improvements. Full-context reserve shapes; actual short-prefix shapes can be much smaller.",
        })

    baseline_free = 2225340416
    guard_pre = 3072 * MIB
    guard_post = 2560 * MIB
    min_triplet = 2 * native["IQ2_S"]["single_expert_projection"]["bytes"] + native["IQ3_XXS"]["single_expert_projection"]["bytes"]
    occupancy = []
    for b in (1024, 512, 256):
        unique = 384 * (1 - (378 / 384) ** b)
        occupancy.append({"B": b, "expected_unique_experts_per_chunk": unique,
                          "total_expected_projection_transfers_for_1024_tokens_in_expert_units": (1024 // b) * unique})

    audit = {
        "pin": PIN,
        "method": "Read source/logs and own standard-library Python arithmetic only. No project script, model, GPU, API, library, or build execution.",
        "sources": receipts,
        "native_weight_payloads": native,
        "shape_predictions": shapes,
        "guard_arithmetic": {
            "observed_M2_CUDA_free_after_vision_and_64token_text_warmup": sized(baseline_free),
            "old_guard_phase": "Constructor sched_reserve -> xeon_hot_backend_finish; both pre-bank and post-bank/private-graph-warmup checks are BEFORE vision initialization.",
            "old_pre_bank_guard_margin": sized(guard_pre),
            "old_post_bank_guard_margin": sized(guard_post),
            "fresh_comparable_M0_CUDA_free_before_bank_and_vision": None,
            "old_phase_required_improvement": "unknown until fresh comparable M0 is measured",
            "old_guard_failure_observed_in_calibration": False,
            "hypothetical_M2_extra_free_if_3072_margin_reimposed_before_payload": sized(guard_pre - baseline_free),
            "example_small_native_triplet": sized(min_triplet),
            "hypothetical_M2_extra_free_if_3072_margin_reimposed_with_small_triplet": sized(guard_pre - baseline_free + min_triplet),
            "warning": "949.75 MiB + payload is conditional arithmetic for reimposing the 3072 MiB margin AFTER vision/text warmup, not the improvement required by the old guard and not an observed admission failure. Do not move the old guard later or charge its future-vision reserve twice. No additional mandatory threshold is proposed.",
            "decision_order": "Measure fresh ubatch1024 at comparable M0 first. If a bounded small bank passes the existing pre-vision pre/post checks and later full warmup has adequate actual headroom, ubatch512 is not mandatory. M1-M3 are additional actual-headroom observations, not old guard thresholds moved to later phases.",
        },
        "uniform_route_occupancy_illustration": {
            "assumption": "Each token independently chooses a uniformly distributed six-element subset out of 384 experts. Not fitted to real routes and not a performance prediction.",
            "formula": "384 * (1 - (378/384)^B)",
            "rows": occupancy,
        },
        "unknowns": [
            "Fresh ubatch1024 M0 free comparable to the old pre-bank/pre-vision guard; whether a bounded positive bank already fits without ubatch reduction",
            "Exact allocation owner and peak live tensor set of the observed 4669.26 MiB CUDA graph arena",
            "How much of that arena is constant full-weight staging, fragmented high-water, or variable masks/activations",
            "Runtime CUDA legacy pool high-water after larger real prefills and maximum supported context",
            "Measured graph arena and same-phase CUDA free at ubatch512/256",
            "Real route-union transfer duplication, prefill and T1 timing impact at smaller ubatch",
            "Numerical and generated-answer differences after changing prefill chunking/backend selection",
        ],
    }
    (out / "WORKSPACE_EVIDENCE.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (out / "SOURCE_EXCERPTS.txt").write_text("Pinned source/log excerpts. No runtime execution.\n" + "".join(extracts))
    (out / "SOURCE_SHA256.txt").write_text("".join(f"{r['sha256']}  {r['repo_path']}\n" for r in receipts.values()))
    print(json.dumps({"source_count": len(receipts), "shape_cases": len(shapes), "baseline_free_MiB": baseline_free/MIB, "small_triplet_MiB": min_triplet/MIB, "outdir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
