#!/usr/bin/env python3
"""Bundle source changes and textual evidence, without model weights or binaries."""
import hashlib
import json
from pathlib import Path
import zipfile

lab = Path(__file__).resolve().parent
root = lab.parent
out = lab / 'results/optimization-numa-rows-20261020'
report = json.loads((out / 'REPORT.json').read_text())
archive = root / 'xeon-numa-work-distribution-20261020.zip'
allowed = {'.json', '.txt', '.log', '.stdout', '.stderr', '.patch', '.csv', '.cpp', '.py', '.sh', '.c', '.h'}
files = {}

def add(path):
    path = Path(path)
    if path.is_file() and not path.is_symlink() and (path.suffix in allowed or path.name == 'LICENSE'):
        files[str(path.relative_to(root))] = path

def tree(path):
    for p in Path(path).rglob('*'):
        add(p)

tree(out)
for comparison in report['comparisons']:
    path = Path(comparison['path'])
    tree(path.parent)
    for run in json.loads(path.read_text())['runs']:
        for name in ('summary.json', 'bench.json', 'bench.log'):
            add(Path(run['summary']).parent / name)
for validation in report['full_model_validation'].values():
    for run in validation['responses']:
        tree(Path(run['summary']).parent)
for name in (
    'prepare_numa_rows.py', 'refine_numa_rows.py', 'check_numa_rows.py',
    'capture_numa_rows.py', 'prepare_moe_groups.py', 'check_moe_groups.py',
    'run_numa_rows_experiments.py', 'finish_numa_rows.py', 'run_cpu_ab_numa_rows.py',
    'bench_deepseek_light.py', 'observe_numa_rows.py', 'observe_phase.py',
    'validate_numa_rows_model.py', 'verify_deepseek_light.py', 'deepseek-logits-probe.cpp',
    'collect_numa_rows_results.py', 'package_numa_rows.py', 'finalize_numa_rows.py',
    'build_deep_candidate.py', 'warm_deepseek.py', 'experiment-preferences.json',
    'run_deepseek_flash.sh', 'rebuild_deepseek.sh',
):
    add(lab / name)
for name in ('numa-rows', 'moe-groups'):
    folder = lab / ('review-' + name + '-20261020')
    for file in ('prepare.json', 'candidate.patch', 'build-record.json', 'configure.log', 'build.log'):
        add(folder / file)
    add(folder / 'build/compile_commands.json')
    for relative in ('ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ops.h', 'ggml/src/ggml-cpu/numa-replica.cpp'):
        add(folder / 'source' / relative)
for relative in (
    'ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ggml-cpu.cpp',
    'ggml/src/ggml-cpu/ops.h', 'ggml/src/ggml-cpu/quants.h',
    'ggml/src/ggml-cpu/arch/x86/quants.c', 'ggml/src/ggml-cpu/CMakeLists.txt',
    'ggml/src/ggml-cpu/numa-replica.cpp', 'ggml/src/ggml-cpu/numa-replica.h',
    'ggml/src/ggml-cpu/tiled/tiled-kernel.h', 'common/chat.h', 'common/chat.cpp',
    'tools/completion/completion.cpp', 'tests/test-chat.cpp', 'LICENSE', 'ggml/LICENSE',
):
    add(lab / 'llama-private-prefetch' / relative)
add(lab / 'build-avx-prefetch/compile_commands.json')
add(lab / 'results/optimization-memory-sync-20261019/REPORT_JA.txt')
add(lab / 'results/optimization-memory-sync-20261019/SOURCE_PROVENANCE.txt')
add(lab / 'results/bottleneck-observation-20261019/REPORT_JA.txt')
add(lab / 'results/model-prewarm-20261020T052158.json')
payload = {'START_HERE_JA.txt': (out / 'REPORT_JA.txt').read_bytes()}
payload.update({name: path.read_bytes() for name, path in sorted(files.items())})
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
payload['SHA256SUMS.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for name, data in payload.items():
        bundle.writestr(name, data)
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name, digest in manifest.items():
        assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
check = {'archive': str(archive), 'bytes': archive.stat().st_size, 'payload_files': len(manifest),
    'crc_and_all_payload_sha256_verified': True, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'excluded': 'Model weights, executables, shared libraries, raw .bin probe/logit outputs and raw perf.data. These remain on the server.'}
archive.with_suffix('.package-check.json').write_text(json.dumps(check, indent=2) + '\n')
print(json.dumps(check, indent=2))
