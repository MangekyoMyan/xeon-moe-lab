#!/usr/bin/env python3
"""Verify unchanged normal runtimes and save final source/build provenance."""
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parent
LAB = WORK.parent
OUT = LAB / 'results/mxfp4-opt-20261023'
BASE = LAB / 'review-glm-q8-columns4-20261020'
FINAL = LAB / 'review-mxfp4-final-20261023'
sys.path.insert(0, str(LAB))
from run_mxfp4_study import audit


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = json.loads((OUT / 'before.json').read_text())
        after = audit()
        assert before == after
        (OUT / 'after-audit.json').write_text(json.dumps(after, indent=2) + '\n')
        required = ['accepted-final24-complete.json', 'accepted-series128-complete.json', 'model-profiles-complete.json', 'launcher-validation.json', 'final-runtime-verify-reuse-complete.json', 'final-runtime-startup-reuse-complete.json']
        assert all((OUT / name).exists() for name in required)
        controllers = {'confirm_model.py', 'observe_model.py', 'observe_model_phase.py', 'observe_model_counters.py', 'run_followup.py', 'run_final_runtime.py', 'run_columns_runtime.py', 'run_columns2_runtime.py', 'verify_launcher.py'}
        live = []
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid(): continue
            try:
                args = [x.decode() for x in (directory / 'cmdline').read_bytes().split(b'\0') if x]
            except (OSError, UnicodeDecodeError): continue
            if len(args) > 1 and args[1].startswith(str(WORK) + '/') and Path(args[1]).name in controllers:
                live.append({'pid': int(directory.name), 'command': args})
            elif args and args[0].startswith(str(LAB) + '/') and Path(args[0]).name in {'llama-bench', 'llama-completion', 'probe', 'rows-probe', 'packed-probe'}:
                live.append({'pid': int(directory.name), 'command': args})
        (OUT / 'process-cleanup.json').write_text(json.dumps({'live_measurement_jobs': live}, indent=2) + '\n')
        assert not live
        hardware = {'time_unix': time.time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'ECC': {}, 'throttles': {}}
        for directory in Path('/sys/devices/system/edac/mc').glob('mc[0-9]*'):
            hardware['ECC'][directory.name] = {name: int((directory / name).read_text()) for name in ('ce_count', 'ue_count')}
        for path in Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*throttle_count'):
            hardware['throttles'][str(path)] = int(path.read_text())
        (OUT / 'hardware-after.json').write_text(json.dumps(hardware, indent=2) + '\n')
        paths = ['ggml/src/ggml-cpu/arch/x86/quants.c', 'ggml/src/ggml-cpu/quants.h', 'ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/repack.cpp', 'src/llama-model-loader.cpp']
        patch, source_hashes = [], {}
        for relative in paths:
            a, b = BASE / 'source' / relative, FINAL / 'source' / relative
            patch.extend(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True), fromfile='a/' + relative, tofile='b/' + relative))
            source_hashes[relative] = {'baseline': sha(a), 'optimized': sha(b)}
        (OUT / 'combined.patch').write_text(''.join(patch))
        artifacts = {}
        for name, build in [('baseline', BASE), ('optimized', FINAL)]:
            artifacts[name] = {str(p): sha(p) for p in (build / 'build/bin').iterdir() if p.is_file() and not p.is_symlink()}
        config = {'model': json.loads((OUT / 'model-download.json').read_text()), 'baseline_build': str(BASE), 'optimized_build': str(FINAL), 'launcher': str(LAB / 'run_gpt_oss_mxfp4_optimized.sh'), 'environment': {'GGML_CPU_MXFP4_PACKED': '1', 'GGML_CPU_MXFP4_COLS4': '1', 'GGML_CPU_MXFP4_PARALLEL_PACK': '1', 'GGML_CPU_REPACK_REUSE_BUFFER': '1', 'GGML_CPU_MXFP4_ROWS': '0', 'GGML_CPU_MXFP4_PAIR': '0'}, 'threads': 24, 'prompt_threads': 24, 'batch': 128, 'load_mode': 'none', 'lazy_mode': 'off', 'process_THP_disabled': True, 'NUMA_policy': 'interleave nodes0,1; CPUs0-23, strictworker affinity', 'source_sha256': source_hashes, 'artifacts_sha256': artifacts, 'normal_runtime_and_global_settings_unchanged': True}
        (OUT / 'selected-config.json').write_text(json.dumps(config, indent=2) + '\n')
        preferences = LAB / 'experiment-preferences.json'
        saved = json.loads(preferences.read_text())
        if 'before_mxfp4_optimization' not in saved:
            saved['before_mxfp4_optimization'] = saved.get('current_work')
        saved['current_work'] = 'Completed CPU-only MXFP4 optimization on this Xeon: exact AVX integer reduction, packed4-rowexpert weights,4-inputprefill reuse,parallel directpacking andsame-sizeCPU stagingreuse. Actual gpt-oss-20b model tested; normal DeepSeek/GLM launchers unchanged. Finalmeasurements andreport atresults/mxfp4-opt-20261023. GPUwork andsimulations remainpaused.'
        saved['current_authorization'] = 'User authorized autonomous CPU optimization, including bold privateengine changes and downloads, forabout5hours. Work executed onthe dedicatedLLMserver; no GPUcompute,purchases,seller messaging,orGPU predictions.'
        saved['mxfp4_optimized_build'] = str(FINAL / 'build')
        saved['gpt_oss_mxfp4_launcher'] = config['launcher']
        saved['mxfp4_optimization_report'] = str(OUT / 'REPORT_JA.txt')
        preferences.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'audit_passed': True, 'live_jobs': len(live), 'ECC': hardware['ECC'], 'max_throttle_count': max(hardware['throttles'].values(), default=0)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*'), LAB / 'experiment-preferences.json']:
                os.chown(path, owner.st_uid, owner.st_gid)
