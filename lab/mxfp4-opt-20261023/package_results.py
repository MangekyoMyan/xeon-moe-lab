#!/usr/bin/env python3
"""Package sources, measurements, and replay instructions without model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

WORK = Path(__file__).resolve().parent
LAB = WORK.parent
OUT = LAB / 'results/mxfp4-opt-20261023'
BASE = LAB / 'review-glm-q8-columns4-20261020'
FINAL = LAB / 'review-mxfp4-final-20261023'
ARCHIVE = LAB.parent / 'xeon-mxfp4-optimization-20261023.zip'


def main():
    config = json.loads((OUT / 'selected-config.json').read_text())
    (OUT / 'REVIEW_NOTES.txt').write_text('''Private self-review completed for the measured AVX-only CPU path.
Scope: CPU kernels, CPU_REPACK representation, synchronous model loading, experiment harnesses.
No public ggml_type or external API was added. No upstream submission was made.

The integer reductions fit in int16 before their final int32 reduction. FP32 accumulation order is retained.
The packed4-row format is68bytes, equal to4ordinary17-byte MXFP4 blocks.
Registration is limited toAVX-only builds and supported3D tensor shapes.
Expert-ID workspace includes duplicate selections; bounds are checked before writes.
Packing workers own disjoint row groups and use thread-local temporaries.
Staging reuse is restricted to synchronous, non-mapped CPU_REPACK MXFP4 loads.
The staging buffer is discarded on size/type/path changes; GPU asynchronous upload buffers are not reused by this change.
Numerical tests, actual-model logits, complete Japanese replies and benchmark evidence are in the results directory.
The final launcher preserves normal current-date behavior; only comparison tests pin the date.

Limits: this is a private measured implementation, not a portability claim or a complete DeepSeek/V100 hybrid engine.
Direct raw Q8 inputs outside the F32 quantizer's [-127,127] range were not the adopted path's validation target.
Profile windows are approximate; overlapping counters are not wall-time attribution.
''')
    readme = '''Xeon MXFP4 optimization review bundle

最初に llm-lab/results/mxfp4-opt-20261023/REPORT_JA.txt を読んでください。
測定はCPUのみ、2xXeon E5-2630L v2、128GB、gpt-oss-20b MXFP4で実施しました。
GPUの推測シミュレーションではありません。

内容:
・基準版の完全なソースツリー（以前の私用AVX/Q8変更を含む）
・最終版の変更済みファイルとcombined.patch
・実験スクリプト、標準出力、診断、検証ハッシュ、実測図
・実行したコマンドとビルド設定、主要バイナリのハッシュ

容量を抑えるため、12.1GBのモデル、ビルドディレクトリ、実行ファイル、大きな中間テンソル/logitsの.binは含めていません。
省略した.binのサイズ・SHA256はOMITTED_BINARY_OUTPUTS.jsonにあります。原本はサーバーの作業ディレクトリに保持しています。
perf.dataは診断用として収録しています。共有ライブラリがない環境ではsymbols.txtを参照してください。

測定ログだけの再集計:
python3 llm-lab/mxfp4-opt-20261023/analyze_results.py --output llm-lab/results/mxfp4-opt-20261023
これはモデルやroot権限を必要としません。

最終ソースの復元:
python3 restore_sources.py
既存ディレクトリは上書きしません。復元後、記録した変更ファイルのハッシュを確認します。
実機で使ったビルドオプションはllm-lab/review-mxfp4-final-20261023/build-commands.jsonにあります。
絶対パスは元サーバーのものなので、新しい環境に合わせてください。
別のx86環境でAVX-only経路を確認する場合は、GGML_NATIVE=OFF、AVX/SSE42/F16C=ON、AVX2/FMA/AVX512/AVX_VNNI=OFFとし、C/C++ flagsに-march=ivybridgeを指定してください。
他のCPUで実行した速度をこのXeonの性能と同一視しないでください。

モデルは次の固定リビジョンを使っています:
https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf
SHA256:27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901

注意する測定条件:
llama-benchのstd::rand()は先行テストに影響されます。同じコマンド列でA/Bを比較してください。
会話テンプレートは現在日付を挿入します。日付をまたぐ回答比較にはpinned-chat-template.jinjaを使用しました。
初期の短い出力上限で途切れた試行は除外されています。
測定ドライバの監査・NUMA/CPU指定は元のサーバー向けです。別環境では設定とパスを確認してから実行してください。
'''
    restore = '''from pathlib import Path
import hashlib,json,shutil
root=Path(__file__).resolve().parent
lab=root/'llm-lab'
base=lab/'review-glm-q8-columns4-20261020/source'
dest=lab/'review-mxfp4-final-20261023/source'
if dest.exists():raise SystemExit('Destination exists; refusing to overwrite')
shutil.copytree(base,dest)
overrides=lab/'review-mxfp4-final-20261023/overrides'
for path in overrides.rglob('*'):
 if path.is_file():
  target=dest/path.relative_to(overrides);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
config=json.loads((lab/'results/mxfp4-opt-20261023/selected-config.json').read_text())
for relative,hashes in config['source_sha256'].items():
 assert hashlib.sha256((base/relative).read_bytes()).hexdigest()==hashes['baseline']
 assert hashlib.sha256((dest/relative).read_bytes()).hexdigest()==hashes['optimized']
print('Restored and hash-verified optimized source:',dest)
'''
    omitted, manifest = [], {}
    def add(archive, path, name):
        if path.is_symlink(): return
        archive.write(path, name)
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('00_README_JA.txt', readme)
        archive.writestr('restore_sources.py', restore)
        for path in sorted((BASE / 'source').rglob('*')):
            relative = path.relative_to(BASE / 'source')
            if not path.is_file() or path.is_symlink() or any(x in {'.git', '.codex', '.agents', '__pycache__', '.pytest_cache'} or x.startswith('.env') for x in relative.parts): continue
            add(archive, path, 'llm-lab/review-glm-q8-columns4-20261020/source/' + str(relative))
        for relative in config['source_sha256']:
            add(archive, FINAL / 'source' / relative, 'llm-lab/review-mxfp4-final-20261023/overrides/' + relative)
        for path in sorted(WORK.iterdir()):
            if path.is_file() and path.suffix in {'.py', '.cpp', '.c', '.json', '.patch', '.asm', '.log', '.txt'}:
                add(archive, path, 'llm-lab/mxfp4-opt-20261023/' + path.name)
        for name in ['without_thp.py', 'run_quantize_tradeoff.py', 'run_mxfp4_study.py', 'glm_q8_experiment.py', 'bench_deepseek_light.py', 'index_model_cache.py', 'observe_glm_phase.py', 'observe_glm_counters_light.py', 'run_gpt_oss_mxfp4_optimized.sh']:
            add(archive, LAB / name, 'llm-lab/' + name)
        for name in ['expert_probe.cpp', 'run_memory_study.py']:
            add(archive, LAB / 'server-study-20261023' / name, 'llm-lab/server-study-20261023/' + name)
        for path in sorted(OUT.rglob('*')):
            if not path.is_file() or path.name in {'archive-manifest.json'} or 'plot-cache' in path.parts: continue
            relative = 'llm-lab/results/mxfp4-opt-20261023/' + str(path.relative_to(OUT))
            if path.suffix == '.bin':
                omitted.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            else:
                add(archive, path, relative)
        for root in (BASE, FINAL):
            for name in ['build-commands.json', 'build.log', 'configure.log']:
                path = root / name
                if path.exists(): add(archive, path, 'llm-lab/' + root.name + '/' + name)
        archive.writestr('OMITTED_BINARY_OUTPUTS.json', json.dumps(omitted, indent=2))
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        members = len(archive.namelist())
        with tempfile.TemporaryDirectory(prefix='mxfp4-replay-') as directory:
            target = Path(directory)
            for name in archive.namelist():
                if name.startswith('llm-lab/results/') or name == 'llm-lab/mxfp4-opt-20261023/analyze_results.py':
                    archive.extract(name, target)
            data = target / 'llm-lab/results/mxfp4-opt-20261023'
            before = json.loads((data / 'SUMMARY.json').read_text())
            subprocess.run(['python3', str(target / 'llm-lab/mxfp4-opt-20261023/analyze_results.py'), '--output', str(data)], stdout=subprocess.DEVNULL, check=True)
            assert before == json.loads((data / 'SUMMARY.json').read_text())
    record = {'path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), 'included_files': members, 'manifested_files': len(manifest), 'omitted_binary_outputs': len(omitted), 'zip_crc_passed': True, 'relocated_reaggregation_identical': True}
    (OUT / 'archive-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
