#!/usr/bin/env python3
"""Record the selected build, matched benchmarks and functional checks."""
import hashlib
import json
from pathlib import Path
import re

LAB = Path(__file__).resolve().parent
RESULTS = LAB / 'results'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def find_run(label):
    matches = []
    for path in RESULTS.glob('deepseek-IQ2_XXS-XL-pinned_interleave-*/summary.json'):
        result = load(path)
        if result.get('arguments', {}).get('label') == label:
            matches.append((path, result))
    assert len(matches) == 1, (label, matches)
    return matches[0]


def test(result, prompt, gen):
    return next(b for b in result['benchmarks'] if
                b['n_threads'] == 12 and b['n_prompt'] == prompt and b['n_gen'] == gen)


def main():
    before_path, before = find_run('engine-omp')
    after_path, after = find_run('engine-optimized-confirm')
    db_path, db = find_run('engine-omp-recheck')
    da_path, da = find_run('engine-optimized-matched-decode')
    pb, pa = test(before, 64, 0), test(after, 64, 0)
    gb, ga = test(db, 0, 32), test(da, 0, 32)
    functional = {}
    for label in ('engine-ub128', 'engine-ub256'):
        path = next(RESULTS.glob(f'deepseek-functional-{label}-*/summary.json'))
        result = load(path)
        assert result['semantic_checks_passed'] and result['exit_code'] == 0
        functional[label] = {
            'source': str(path), 'timings_ms': result['timings_ms'],
            'response': result['response'], 'semantic_checks_passed': True,
            'disk_read_bytes': max(s['disk_read_bytes'] for s in result['observations']),
        }
    response = (RESULTS / 'optimized-launcher-response.txt').read_text()
    assert '東京' in response and '42' in response
    historical = load(RESULTS / 'deepseek-findings.json')
    old_pp = historical['functional_verification']['prompt_processing']['seconds']
    new_pp = functional['engine-ub256']['timings_ms']['prompt eval'] / 1000
    build = LAB / 'build-native-noblas'
    flags = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if re.match(r'^(GGML_NATIVE|GGML_OPENMP|GGML_BLAS|GGML_CPU_REPACK|GGML_LLAMAFILE|CMAKE_BUILD_TYPE):', line):
            key, value = line.split('=', 1)
            flags[key.split(':')[0]] = value
    hashes = {}
    for name in ('llama-completion', 'libllama-completion-impl.so', 'libllama-common.so',
                 'libllama.so', 'libggml-cpu.so', 'libggml.so'):
        path = build / 'bin' / name
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        'status': 'Applied and verified',
        'model': 'Existing DeepSeek V4 Flash IQ2_XXS-XL, 73.126 GiB, CPU only; same GGUF weights',
        'source_commit': '436f6f89e1e581249900b37a5b8a12a36a6d0912',
        'private_source': str(LAB / 'llama-private'),
        'launcher': str(LAB / 'run_deepseek_flash.sh'),
        'previous_launcher': str(LAB / 'run_deepseek_flash_previous.sh'),
        'rebuild_script': str(LAB / 'rebuild_deepseek.sh'),
        'build_flags': flags,
        'runtime': {
            'threads': 12, 'threads_batch': 12, 'batch': 256, 'ubatch': 256,
            'context': 4096, 'load_mode': 'mmap',
            'cpu_binding': 'Strict pins on physical CPUs 0-11; process cpuset 0-23',
            'memory_policy': 'Interleave NUMA nodes 0 and 1',
            'reasoning_default': 'auto, unchanged; --reasoning off now reaches the template',
        },
        'prefill_64_tokens': {
            'before_tok_s': pb['avg_ts'], 'after_tok_s': pa['avg_ts'],
            'before_seconds': pb['avg_ns'] / 1e9, 'after_seconds': pa['avg_ns'] / 1e9,
            'speed_ratio': pa['avg_ts'] / pb['avg_ts'],
            'before_repetitions': len(pb['samples_ns']), 'after_repetitions': len(pa['samples_ns']),
            'before_source': str(before_path), 'after_source': str(after_path),
            'conditions': '12 threads, mmap/interleave, same cached model pages, ubatch 128. First two timed prompt sequences match; the candidate also has a third repetition.',
        },
        'matched_decode': {
            'before_tok_s': gb['avg_ts'], 'after_tok_s': ga['avg_ts'],
            'change_percent': 100 * (ga['avg_ts'] / gb['avg_ts'] - 1),
            'before_samples': gb['samples_ts'], 'after_samples': ga['samples_ts'],
            'before_source': str(db_path), 'after_source': str(da_path),
            'conditions': 'Fresh processes, identical -p 0 -n 32 -r 3 -t 12, ubatch 128, strict pins, mmap/interleave. Mixed-suite runs consume different std::rand token streams before decode and are not isolated engine comparisons.',
        },
        'natural_211_token_prompt': {
            'previous_seconds': old_pp, 'optimized_seconds': new_pp,
            'speed_ratio': old_pp / new_pp,
            'latency_reduction_percent': 100 * (1 - new_pp / old_pp),
            'previous_source': str(RESULTS / 'deepseek-functional-20261019T042554/summary.json'),
            'optimized_source': functional['engine-ub256']['source'],
            'note': 'One run per profile. Previous: 24 threads, BLAS, ubatch 128 and reasoning option bug. Optimized: 12 threads, no BLAS, ubatch 256, corrected template kwargs. Both prompts contain 211 tokens.',
        },
        'functional_checks': functional,
        'launcher_smoke_test': {
            'response': response, 'checks_passed': True,
            'stderr': str(RESULTS / 'optimized-launcher-stderr.log'),
            'prompt_tokens': 22, 'generation_tok_s': 2.41,
        },
        'startup': {
            'cached_model_load_seconds': functional['engine-ub256']['timings_ms']['load'] / 1000,
            'note': 'Model pages were already cached. Cold HDD loading still takes minutes. mmap avoids the extra anonymous weight copy and retains file-cache pages for later launches.',
        },
        'patch': str(LAB / 'patches/completion-template-kwargs.patch'),
        'source_changes': [
            'Forward CLI template kwargs into initial completion formatting and later chat turns',
            'Check enable_thinking true/false with empty and nonempty chat history',
        ],
        'validation': [
            'Full chat test suite passed', 'Quantization tests: 0 failures',
            'DeepSeek CPU operations: 38/38 passed',
            'Native-pool barrier tests passed; that pool was not selected',
            'Final profile benchmarked three times',
            'Both 211-token runs returned the code, capital and arithmetic result correctly',
            'Updated launcher returned the correct Japanese capital and arithmetic answer',
            'No HDD reads in final benchmark and functional observation windows',
        ],
        'limits': [
            'Only short contexts benchmarked; configured 4096-token context is not a long-context benchmark',
            'No perplexity or broad quality evaluation; BLAS removal changes the supported computation path, not GGUF weights',
            'No new downloads, Q2_K model tests or GPU tests',
            'Startup timing requires cached model pages',
            'Ubatch 256 beat 128 by about 4% on one 211-token prompt, not a broad batch-size optimum',
        ],
        'experiment_manifest': str(RESULTS / 'engine-optimization-manifest.json'),
        'binary_sha256': hashes,
    }
    destination = RESULTS / 'deepseek-engine-optimization.json'
    save(destination, report)
    prefs = load(LAB / 'experiment-preferences.json')
    prefs['current_work'] = ('Applied private llama.cpp build: OpenMP ON, BLAS OFF, 12 physical-core workers, '
                            'interleaved mmap weights, ubatch 256. Synthetic prefill about 2.93x faster; '
                            'matched decode 2.1593 -> 2.1491 tok/s. Natural 211-token prefill 82.57 -> 35.15 s. '
                            'Fixed template kwargs and verified Japanese answers. System /opt build and previous launcher retained.')
    prefs['deepseek_optimization_results'] = str(destination)
    prefs['private_llama_source'] = str(LAB / 'llama-private')
    save(LAB / 'experiment-preferences.json', prefs)
    historical['engine_optimization_followup'] = str(destination)
    historical['launcher'] = str(LAB / 'run_deepseek_flash_previous.sh')
    historical['historical_note'] = 'Earlier settings-only investigation. Current run_deepseek_flash.sh uses the optimized build; see engine_optimization_followup.'
    save(RESULTS / 'deepseek-findings.json', historical)
    manifest = load(RESULTS / 'engine-optimization-manifest.json')
    manifest['final_result'] = str(destination)
    manifest['status'] = 'Complete; optimized launcher applied and Japanese responses verified.'
    manifest['final_candidate']['ubatch'] = 256
    save(RESULTS / 'engine-optimization-manifest.json', manifest)
    print(json.dumps({k: report[k] for k in ('status', 'prefill_64_tokens', 'matched_decode', 'natural_211_token_prompt', 'startup')}, indent=2))


if __name__ == '__main__':
    main()
