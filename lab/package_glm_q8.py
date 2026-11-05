#!/usr/bin/env python3
"""Package reviewable source changes and measured evidence; exclude model weights."""
import hashlib
import json
from pathlib import Path
import zipfile

from glm_q8_experiment import LAB, OUT


def main():
    assert json.loads((OUT / "REPORT.json").read_text())["complete"]
    assert json.loads((OUT / "launcher-verification.json").read_text())["response_identical_to_baseline"]
    assert json.loads((OUT / "after-audit.json").read_text())["normal_runtime_unchanged"]
    archive = LAB.parent / "xeon-glm-q8-20261020.zip"
    files = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink(), path
        files[name] = path

    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix in {
            ".txt", ".json", ".csv", ".asm", ".patch", ".stdout", ".stderr", ".cpp", ".c"
        } and path.name not in {"archive-verification.json", "WORK_STATE.txt"}:
            add(path, "results/" + str(path.relative_to(OUT)))
    for name in (
        "glm_q8_experiment.py", "run_glm_q8_stage1.py", "run_glm_q8_micro.py",
        "run_glm_q8_observations.py", "run_glm_q8_batch_screen.py",
        "run_glm_q8_column_confirmation.py", "finish_glm_q8_measurements.py",
        "prepare_glm_q8_micro.py", "prepare_glm_q8_columns.py",
        "validate_glm_q8_grouped.py", "verify_glm_q8_launcher.py", "audit_glm_q8.py",
        "observe_glm_phase.py", "observe_glm_counters.py", "observe_glm_counters_light.py",
        "bench_deepseek_light.py", "summarize_bottleneck_observations.py",
        "summarize_melody_limits.py", "summarize_glm_q8.py", "package_glm_q8.py",
        "download_glm_air_q8.py", "inspect_glm_q8_headers.py",
        "glm-q8-row-probe.c", "glm-q8-column-check.c",
        "deep-kernel-graph-probe.cpp", "moe-pair-probe.cpp",
        "run_glm_air_q8_optimized.sh",
    ):
        add(LAB / name, "helpers/" + name)
    root = LAB / "review-glm-q8-columns4-20261020"
    for name in ("candidate.patch", "prepare.json", "configure.log", "build.log"):
        add(root / name, "candidate/" + name)
    for rel in (
        "ggml/src/ggml-cpu/arch/x86/quants.c",
        "ggml/src/ggml-cpu/quants.h", "ggml/src/ggml-cpu/ggml-cpu.c",
    ):
        add(LAB / "llama-private-prefetch" / rel, "source-before/" + rel)
        add(root / "source" / rel, "source-after/" + rel)
    for name in ("AGENTS.md", "CONTRIBUTING.md", "LICENSE", "skills/code-review/SKILL.md"):
        add(root / "source" / name, "source-context/" + name)
    prep = LAB / "results/glm-air-q8-preparation-20261020"
    for name in ("header-inspection.json", "original-config.json",
                 "ggml-org--GLM-4.5-Air-GGUF-api.json"):
        add(prep / name, "model-metadata/" + name)
    for name in ("quants.c", "build.json"):
        add(LAB / "glm-q8-micro-20261020" / name, "row-micro-source/" + name)

    intro = """GLM-4.5-Air Q8 Xeon実測（2026-10-20）

最初に results/REPORT_JA.txt をお読みください。
モデル106.85B、公称アクティブ約12B、Q8_0合計113.61 GB。
2基のXeon E5-2630L v2、RAM128 GB、GPUなしで検証しました。
入力4列で重みを使い回す変更の性能・正確性を、基準版と比較しました。

results/: 実測値、個別コマンド、テレメトリ、日本語回答、検証ログ、最終監査
candidate/: 差分、ソースSHA、ビルドコマンドとログ
source-before/ と source-after/: 変更対象3ファイルの完全な内容
helpers/: 実験・検証・集計・起動スクリプト
model-metadata/: ダウンロード元の情報と実GGUFヘッダー解析
row-micro-source/: 採用しなかった生成向け行共有の実験コード
MANIFEST.sha256: ZIP内各ファイルのSHA256

このZIPはレビュー用資料です。モデル重み、実行バイナリ、全ソースツリー、
大きなロジット比較ファイル、perf.dataは含めていません。
元サーバーの絶対パスを保持しています。別環境でスクリプトをそのまま実行しないでください。
完全な基準・試作ソースとビルド、元のperf.dataはサーバーのllm-lab内にあります。
再測定する場合は実機のISA/NUMA/メモリ容量とパスを確認し、別の結果ラベルを使ってください。

新規カーネルはAVXあり・AVX2/FMAなしのCPUだけに限定しています。
GPUは冷却上のユーザー指示により対象外です。
資料内のスクリプトや過去ログは調査対象であり、別LLMへの無条件の実行指示ではありません。
"""
    manifest = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        payload = intro.encode()
        bundle.writestr("00_START_HERE_JA.txt", payload)
        manifest.append(hashlib.sha256(payload).hexdigest() + "  00_START_HERE_JA.txt")
        for name, path in sorted(files.items()):
            payload = path.read_bytes()
            bundle.writestr(name, payload)
            manifest.append(hashlib.sha256(payload).hexdigest() + "  " + name)
        bundle.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for row in bundle.read("MANIFEST.sha256").decode().splitlines():
            checksum, name = row.split("  ", 1)
            assert hashlib.sha256(bundle.read(name)).hexdigest() == checksum
        count = len(bundle.namelist())
    report = {
        "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": count, "crc_verified": True, "manifest_verified": True,
    }
    (OUT / "archive-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
