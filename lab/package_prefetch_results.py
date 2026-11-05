#!/usr/bin/env python3
"""Package reviewable sources and evidence; exclude models and binaries."""
import hashlib
import json
from pathlib import Path
import zipfile

lab = Path(__file__).resolve().parent
root = lab.parent
out = lab / 'results/optimization-memory-sync-20261019'
report = json.loads((out / 'REPORT.json').read_text())
archive = root / 'xeon-memory-sync-optimization-20261019.zip'
allowed = {'.json', '.txt', '.log', '.stdout', '.stderr', '.patch', '.asm', '.cpp', '.py', '.sh', '.c', '.h'}
files = {}

def add(path):
    path = Path(path)
    if path.is_file() and path.suffix in allowed and not path.is_symlink():
        files[str(path.relative_to(root))] = path

def tree(path):
    for child in Path(path).rglob('*'): add(child)

tree(out)
for comparison in report['comparisons']:
    path = Path(comparison['path'])
    tree(path.parent)
    data = json.loads(path.read_text())
    for run in data['runs']:
        run_dir = Path(run['summary']).parent
        for name in ('summary.json', 'bench.json', 'bench.log'): add(run_dir / name)
for run in report['model_validation']['responses'] + report['response_reverse']['runs']:
    tree(Path(run['summary']).parent)
if report['activation'].get('launcher_smoke_summary'):
    tree(Path(report['activation']['launcher_smoke_summary']).parent)
for name in [
    'prepare_q8_prefetch.py', 'q8-prefetch-probe.c', 'run_q8_prefetch_micro.py',
    'prepare_memory_sync_candidates.py', 'native-barrier-probe.cpp', 'run_memory_sync_candidates.py',
    'summarize_wait_cpu.py', 'bench_deepseek.py', 'bench_deepseek_light.py',
    'run_cpu_ab.py', 'run_cpu_ab_light.py', 'verify_deepseek.py', 'verify_deepseek_light.py',
    'run_light_followups.py', 'stage_prefetch_candidate.py', 'run_final_prefetch_checks.py',
    'validate_prefetch_model.py', 'deepseek-logits-probe.cpp', 'run_prefetch_response_reverse.py',
    'activate_prefetch.py', 'summarize_prefetch_results.py', 'package_prefetch_results.py',
    'build_deep_candidate.py', 'deep-kernel-graph-probe.cpp', 'moe-pair-probe.cpp',
    'experiment-preferences.json', 'run_deepseek_flash.sh', 'run_deepseek_flash_before_prefetch.sh',
    'run_deepseek_flash_prefetch_staged.sh', 'rebuild_deepseek.sh', 'rebuild_deepseek_before_prefetch.sh',
    'rebuild_deepseek_prefetch_staged.sh',
]: add(lab / name)
for name in ('q8-prefetch8', 'native-barrier'):
    folder = lab / ('review-' + name + '-20261019')
    for file in ('prepare.json', 'build-record.json', 'configure.log', 'build.log', name + '.patch'): add(folder / file)
    relative = 'ggml/src/ggml-cpu/arch/x86/quants.c' if name == 'q8-prefetch8' else 'ggml/src/ggml-cpu/ggml-cpu.c'
    add(folder / 'source' / relative)
    add(lab / 'llama-private' / relative)
add(lab / 'llama-private-prefetch/ggml/src/ggml-cpu/arch/x86/quants.c')
for relative in (
    'ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ggml-cpu.cpp',
    'ggml/src/ggml-cpu/quants.h', 'ggml/src/ggml-cpu/CMakeLists.txt',
    'ggml/src/ggml-cpu/numa-replica.cpp', 'ggml/src/ggml-cpu/numa-replica.h',
    'ggml/src/ggml-cpu/tiled/tiled-kernel.h', 'common/chat.h', 'common/chat.cpp',
    'tools/completion/completion.cpp', 'tests/test-chat.cpp',
): add(lab / 'llama-private-prefetch' / relative)
add(lab / 'results/deep-review-final/REPORT_JA.txt')
add(lab / 'results/deep-review-source-adoption.json')
for file in (lab / 'q8-prefetch-20261019').glob('*'):
    add(file)
for build in ('build-avx-deep', 'build-avx-prefetch'):
    add(lab / build / 'compile_commands.json')
for file in ('REPORT_JA.txt', 'freeze-before.json', 'freeze-after.json'):
    add(lab / 'results/bottleneck-observation-20261019' / file)
add(lab / 'results/deepseek-tensor-inventory.json')
payload = {'START_HERE_JA.txt': (out / 'REPORT_JA.txt').read_bytes()}
payload.update({name: path.read_bytes() for name, path in sorted(files.items())})
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
payload['SHA256SUMS.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for name, data in payload.items(): bundle.writestr(name, data)
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None: raise RuntimeError('ZIP CRC verification failed')
    for name, digest in manifest.items():
        if hashlib.sha256(bundle.read(name)).hexdigest() != digest: raise RuntimeError('ZIP SHA256 verification failed')
check = {'archive': str(archive), 'bytes': archive.stat().st_size, 'payload_files': len(manifest),
    'crc_and_all_payload_sha256_verified': True, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'excluded': 'Models, executables, shared libraries and raw .bin logits/probe outputs. Raw outputs remain on the server.'}
archive.with_suffix('.package-check.json').write_text(json.dumps(check, indent=2) + '\n')
print(json.dumps(check, indent=2))
