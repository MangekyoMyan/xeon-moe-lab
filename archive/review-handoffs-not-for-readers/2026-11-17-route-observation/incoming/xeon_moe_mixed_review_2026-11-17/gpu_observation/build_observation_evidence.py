#!/usr/bin/env python3
"""Offline source receipt and arithmetic only. Does not load any project ELF/model.

The repository is opened read-only. Output goes to a new directory outside it.
The proposed measurement contract is deliberately not an executable engine patch.
"""
from __future__ import annotations

import hashlib
import json
import argparse
import shutil
from pathlib import Path

ROOT: Path
OUT: Path
PACKAGE = Path(__file__).resolve().parent
COMMIT = 'ff6168094858141aeeebf45abc5c309882fe5216'
H = 'archive/review-handoffs-not-for-readers/2026-11-16-current'
E = H + '/snapshot/llm-lab/exploration-20261112'
U = H + '/snapshot/llm-lab/deepseek41-bringup-20261101/source'
A = 'archive/review-handoffs-not-for-readers/2026-11-16-abs-operator'
N = 'archive/review-handoffs-not-for-readers/2026-11-17-mixed-operator'

SOURCES = {
    'N_CONSULT': (N + '/CONSULT_PRO_JA.txt', 'Current questions and evidence limits'),
    'N_RESULTS': (N + '/RESULTS_JA.txt', 'New measured CPU mixed results'),
    'N_POLICY': (N + '/POLICY_AND_PLAN_JA.txt', 'Current user constraints; old proposals are not new user instructions'),
    'N_README': (N + '/README.txt', 'Current handoff scope'),
    'N_INITIAL': (N + '/INITIAL_RUNTIME.json', 'Initial current API identity'),
    'N_FINAL': (N + '/FINAL_RUNTIME.json', 'Final API and library identity; no GPU memory field'),
    'N_SCREEN': (N + '/SCREEN_DECISION.json', 'CPU prospective screen decision'),
    'A_FINAL': (A + '/FINAL_RUNTIME.json', 'Earlier recorded 29920 MiB GPU usage, not a fresh free-memory measurement'),
    'H_STATE': (H + '/review/capture/CURRENT_STATE.json', 'Recorded 32768 MiB GPU total memory'),
    'PROFILE': (E + '/checkpoint-1600/profile.json', 'Final GPU37-39, CPU0-36, 128k KV and compact settings'),
    'CAPACITY': (E + '/hot-ffn/CAPACITY.json', 'Native triplet bytes per layer, old marginal-training scope'),
    'HOT_CPP': (E + '/hot-ffn/hot-backend.cpp', 'Actual private CPU/GPU dispatcher and capacity guards'),
    'HOT_CONTEXT': (E + '/hot-ffn/llama-context.cpp', 'Hot backend creation, scheduler integration, initialization order'),
    'HOT_GRAPH': (E + '/hot-ffn/llama-graph.cpp', 'Hot graph substitution and GPU tail placement'),
    'HOT_BUILD': (E + '/hot-ffn/ENGINE_BUILD.json', 'Historical hot source and library provenance'),
    'HOT_PROFILES': (E + '/prepare_hot_profiles.py', 'Old hot14 changes n_cpu_moe to 40 and CPU plan'),
    'HOT_TEST': (E + '/hot-ffn/test-backend.cpp', 'Old exact-dispatcher correctness fixture; reserve0 test cache'),
    'HOT_VALIDATION': (E + '/hot-ffn/BACKEND_VALIDATION.json', 'Old tests performed with API stopped'),
    'HOT_FIXTURE_RESULTS': (E + '/hot-ffn/FIXTURE_RESULTS.json', 'Old isolated fixture scope, not current final latency'),
    'HOT_PROBE': (E + '/hot_ffn_probe.cpp', 'Old separate overlap probe with weaker 700 MiB preflight, not a current-policy runner'),
    'CPU_C': (E + '/mx-prepack/release-candidate-a/sources/ggml-cpu.c', 'Actual final CPU MUL_MAT_ID host IDs, no census hook'),
    'CPU_BUILD': (E + '/mx-prepack/release-candidate-a/BUILD.json', 'Final CPU source provenance'),
    'CENSUS_C': (E + '/census/ggml-cpu.c', 'Old marginal census insertion site'),
    'CENSUS_CPP': (E + '/census/census.cpp', 'Old counts-only mmap collector, no joint route order'),
    'CENSUS_TRAIN': (E + '/train_census.py', 'Old training calls and marginal-only scope'),
    'NORMAL_GRAPH': (U + '/src/llama-graph.cpp', 'Final normal graph top-k and FFN nodes'),
    'NORMAL_CONTEXT': (U + '/src/llama-context.cpp', 'Per-ubatch graph execution and callback setup'),
    'NORMAL_LOADER': (U + '/src/llama-model-loader.cpp', 'CPU buffer override selection'),
    'SPARSE_MODEL': (E + '/sparse-cache/llama-model.cpp', 'Actual final sparse-cache loader overlay'),
    'SPARSE_CACHE': (E + '/sparse-cache/release-v2-a/sources/model-cache.h', '64 GiB host reserve construction'),
    'GGML_C': (U + '/ggml/src/ggml.c', 'Top-k is a view of argsort and preserves strides'),
    'GGML_CPU_CPP': (U + '/ggml/src/ggml-cpu/ggml-cpu.cpp', 'CPU graph_compute synchronous backend'),
    'SCHEDULER': (H + '/snapshot/llm-lab/gpu-stream-20261103/ggml-backend.cpp', 'Actual loaded base scheduler overlay; copies, waits, callback splits'),
    'CUDA_CU': (E + '/cuda-sort/release-candidate-a/ggml-cuda.cu', 'Actual CUDA overlay, stream copies/sync, graph replay, events'),
    'CUDA_COMMON': (U + '/ggml/src/ggml-cuda/common.cuh', 'Private nonblocking streams and internal fork/join events'),
    'CUDA_BUILD': (E + '/cuda-sort/release-candidate-a/BUILD.json', 'Actual CUDA source/build provenance'),
    'LLAMA_H': (U + '/include/llama.h', 'cb_eval constructor parameters'),
    'COMMON_COMMON': (U + '/common/common.cpp', 'Common parameters to llama context callback'),
    'COMMON_DEBUG': (U + '/common/debug.cpp', 'Existing debug callback asks true before applying regex'),
    'ARG_CPP': (U + '/common/arg.cpp', 'n_cpu_moe CPU tensor override'),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    ROOT = args.repository.resolve()
    OUT = args.output_dir.resolve()
    if OUT == ROOT or ROOT in OUT.parents:
        parser.error('Output directory must be outside the source repository')
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('build_observation_evidence.py', 'GPU_OBSERVATION_REVIEW_JA.txt', 'OBSERVATION_CONTRACT.json'):
        shutil.copyfile(PACKAGE / name, OUT / name)
    entries = {}
    for key, (rel, role) in SOURCES.items():
        data = (ROOT / rel).read_bytes()
        entries[key] = {
            'path': rel,
            'sha256': digest(data),
            'bytes': len(data),
            'line_count': len(data.splitlines()),
            'role': role,
        }
    expected_source_hashes = {
        'HOT_CPP': '919d933fcf05ef0241022f756b5fb7c83a7084b413987bf1620b98b11a06dab6',
        'CPU_C': 'e8fdf3d424463f9852fa774d9de6057dce35855a872936bda87d12d3d756e3a8',
        'CUDA_CU': '2cdc72370dcb884fcbac85cf2757ef1259265c5181558b1e1eb933157a72d3d0',
        'SPARSE_MODEL': '66b4b0dc3814722f2490922555c0b7db1e385789a64abdcb0b284e2bcfe24df1',
    }
    for key, expected in expected_source_hashes.items():
        if entries[key]['sha256'] != expected:
            raise ValueError(f'Source hash mismatch for {key}')

    current = json.loads((ROOT / (N + '/FINAL_RUNTIME.json')).read_text())
    prior = json.loads((ROOT / (A + '/FINAL_RUNTIME.json')).read_text())
    state = json.loads((ROOT / (H + '/review/capture/CURRENT_STATE.json')).read_text())
    capacities = json.loads((ROOT / (E + '/hot-ffn/CAPACITY.json')).read_text())
    gpu_line = state['hardware']['gpu']['stdout'].splitlines()[0]
    total_field = gpu_line.split(',')[3].strip()
    if total_field != '32768 MiB':
        raise ValueError('Unexpected recorded GPU total memory')
    total_mib = 32768
    used_mib = prior['monitor']['gpu']['used_MiB']
    inferred_free_mib = total_mib - used_mib
    eligible = [r for r in capacities['layers'] if 0 <= r['layer'] <= 36]
    if len(eligible) != 37:
        raise ValueError('Expected 37 hypothetical eligible layers')
    min_payload = min(r['expert_bytes'] for r in eligible)
    min_layers = [r['layer'] for r in eligible if r['expert_bytes'] == min_payload]
    prior_guard_margin = inferred_free_mib - 3072 - min_payload / (1024 * 1024)
    capacity_analysis = {
        'status': 'offline_arithmetic_from_recorded_evidence_only',
        'commit': COMMIT,
        'prior_usage_source': 'A_FINAL',
        'prior_usage_unix': prior['monitor']['unix'],
        'prior_api_pid': prior['api_pid'],
        'recorded_total_source': 'H_STATE',
        'recorded_total_MiB': total_mib,
        'recorded_used_MiB': used_mib,
        'inferred_total_minus_used_MiB': inferred_free_mib,
        'not_cudaMemGetInfo': True,
        'current_mixed_final_source': 'N_FINAL',
        'current_mixed_final_unix': current['unix'],
        'current_mixed_final_api_pid': current['api_pid'],
        'current_mixed_final_has_gpu_memory': 'gpu' in current['monitor'],
        'current_actual_free_memory': None,
        'positive_payload_preflight': 'cudaMemGetInfo.free > native_payload_bytes + 3072 * 2**20',
        'postallocation_guard': 'cudaMemGetInfo.free > 2560 * 2**20',
        'minimum_one_native_triplet_bytes_cpu_layers_0_36': min_payload,
        'layers_with_minimum_native_triplet': min_layers,
        'minimum_one_native_triplet_MiB': min_payload / (1024 * 1024),
        'necessary_free_for_minimum_payload_strictly_greater_MiB': 3072 + min_payload / (1024 * 1024),
        'prior_snapshot_margin_vs_minimum_preflight_MiB': prior_guard_margin,
        'prior_snapshot_compatible_with_positive_private_guard': False,
        'caveats': [
            'The total and used values are historical recorded measurements, not a same-call CUDA free-memory result.',
            'N_FINAL retains the same API pid/config but omits GPU memory. It cannot establish current free memory.',
            'A fresh NVML free value is a necessary screening observation; it does not replace the unchanged actual CUDA guard.',
            'CUDA context, padded banks, workspaces, activation allocation and later vision/pool growth remain material.',
            'A second full-model route-only context would still duplicate GPU37-39/KV and is not memory-free.',
            'The zero-payload all-miss dispatcher is not a placement-neutral route collector.',
        ],
    }
    write_json('CAPACITY_READONLY_ANALYSIS.json', capacity_analysis)

    manifest = {
        'schema_version': 1,
        'repository': 'https://github.com/MangekyoMyan/xeon-moe-lab',
        'commit': COMMIT,
        'local_read_only_root': str(ROOT),
        'source_ids': entries,
        'selected_build_source_hashes_independently_matched': expected_source_hashes,
        'web_primary_sources': [{
            'url': 'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html',
            'role': 'cudaEventRecord/Query/ElapsedTime flags, completeness and timing semantics',
            'document_version': 'CUDA Runtime API 12.9.1',
            'local_source_sha256': None,
            'note': 'Versioned external documentation; no invented byte hash. Engine source is separately pinned above.',
        }],
        'execution_scope': {
            'model_loaded': False, 'gpu_api_called': False, 'service_accessed': False,
            'project_elf_executed': False, 'repository_modified': False,
            'offline_python_hash_and_arithmetic_only': True,
        },
    }
    write_json('SOURCE_MANIFEST.json', manifest)
    receipt_files = {}
    for name in ('build_observation_evidence.py', 'GPU_OBSERVATION_REVIEW_JA.txt',
                 'OBSERVATION_CONTRACT.json', 'CAPACITY_READONLY_ANALYSIS.json', 'SOURCE_MANIFEST.json'):
        path = OUT / name
        if path.is_file():
            data = path.read_bytes()
            receipt_files[name] = {'bytes': len(data), 'sha256': digest(data)}
    write_json('OFFLINE_RECEIPT.json', {
        'scope': 'source hashing and historical capacity arithmetic only; no performance measurement',
        'repository_commit': COMMIT,
        'sources_hashed': len(entries),
        'files': receipt_files,
    })
    print(json.dumps({'sources_hashed': len(entries), 'capacity_margin_MiB': prior_guard_margin,
                      'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
