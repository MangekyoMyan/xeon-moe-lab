#!/usr/bin/env python3
"""Package source changes and measured evidence, excluding models and compiled binaries."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

lab = Path(__file__).resolve().parent
projects = lab.parent
final = lab / "results/deep-review-final"
if not (final / "REPORT_JA.txt").is_file() or not (final / "REPORT.json").is_file():
    raise SystemExit("Write the completed final reports before packaging")
destination = projects / "xeon-deep-optimization-results-20261019"
archive = destination.with_suffix(".zip")
if destination.exists() or archive.exists(): raise SystemExit("Preserve the existing result package")
destination.mkdir()
allowed = {".txt", ".json", ".log", ".stderr", ".stdout", ".csv", ".patch", ".py", ".cpp", ".c", ".h", ".sh"}

def copy(source, relative):
    source = Path(source)
    if not source.is_file(): return
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise RuntimeError("Conflicting package destination: " + str(relative))
    shutil.copyfile(source, target)

for source in final.rglob("*"):
    if source.is_file() and source.suffix in allowed: copy(source, Path("report") / source.relative_to(final))
scripts = ["prepare_moe_pair.py", "moe-pair-probe.cpp", "prepare_q8_rows4.py", "build_deep_candidate.py",
           "run_deep_kernel_micro.py", "deep-kernel-graph-probe.cpp", "numa-replica.cpp", "prepare_numa_replica.py",
           "numa-replica-probe.cpp", "capture_numa_replica.py", "prepare_deep_combined.py", "deepseek-logits-probe.cpp",
           "run_deep_logits.py", "run_cpu_ab.py", "run_cpu_thread_ab.py", "run_deep_final_screens.py",
           "summarize_deep_rounds.py", "summarize_deep_resources.py", "capture_deep_profile.py",
           "summarize_deep_profiles.py", "stage_deep_adoption.py", "run_deep_acceptance.py", "write_deep_report.py",
           "verify_deep_rebuild.py", "run_deep_profiles.py", "activate_deep_build.py", "run_deep_short_ab.py",
           "bench_deepseek.py", "verify_deepseek.py", "package_deep_results.py"]
for name in scripts: copy(lab / name, Path("tools") / name)
for name in ("run_deepseek_flash.sh", "rebuild_deepseek.sh", "run_deepseek_flash_before_deep_review.sh",
             "rebuild_deepseek_before_deep_review.sh", "experiment-preferences.json"):
    copy(lab / name, Path("active-configuration") / name)
for name in ("completion-template-kwargs.patch", "ssse3-iq-sign-unpack.patch"):
    copy(lab / "patches" / name, Path("baseline-patches") / name)
for name in ("hardware-baseline.json", "model-inventory.json"):
    copy(lab / name, Path("hardware-and-model") / name)
for name in ("handoff-dimm-inventory.json", "handoff-current-environment.json", "deepseek-tensor-inventory.json"):
    copy(lab / "results" / name, Path("hardware-and-model") / name)
for variant in ("moe", "q8-rows4", "numa", "combined"):
    root = lab / f"review-deep-{variant}-20261019"
    for source in root.iterdir():
        if source.is_file() and source.suffix in allowed:
            copy(source, Path("variants") / variant / source.name)
    preparation = json.loads((root / "prepare.json").read_text())
    if variant == "combined":
        for relative in preparation["files"]:
            copy(root / "source" / relative, Path("source-after") / relative)
            copy(lab / "review-deep-pools-20261019/baseline/source" / relative, Path("source-before") / relative)
for source in (lab / "review-deep-pools-20261019").glob("*.json"):
    copy(source, Path("pool-builds") / source.name)
for variant in ("baseline", "pthread"):
    for source in (lab / "review-deep-pools-20261019" / variant).iterdir():
        if source.is_file() and source.suffix in allowed:
            copy(source, Path("pool-builds") / variant / source.name)
for comparison in (lab / "results").glob("cpu-abba-20261019T*/comparison.json"):
    data = json.loads(comparison.read_text())
    if "candidate_gain_percent" not in data: continue
    for source in comparison.parent.iterdir():
        if source.is_file() and source.suffix in allowed:
            copy(source, Path("measurements") / comparison.parent.name / source.name)
    for run in data["runs"]:
        result = Path(run["summary"]).parent
        for name in ("summary.json", "bench.json", "bench.log"):
            copy(result / name, Path("measurements") / result.name / name)
for pattern in ("deep-logits-20261019T*", "deep-kernel-micro-20261019T*", "numa-replica-placement-20261019T*",
                "deepseek-functional-deep-review*", "deepseek-functional-adopted-deep*", "deep-profile-*", "deep-review-rebuild-check"):
    for root in (lab / "results").glob(pattern):
        if not root.is_dir(): continue
        for source in root.rglob("*"):
            if source.is_file() and source.suffix in allowed:
                copy(source, Path("evidence") / root.name / source.relative_to(root))
for source in (lab / "results").glob("deep-review-*"):
    if source.is_file() and source.suffix in allowed:
        copy(source, Path("evidence") / source.name)
for source in (lab / "review-deep-micro-20261019").iterdir():
    if source.is_file() and source.suffix in allowed:
        copy(source, Path("kernel-micro-build") / source.name)
for name in ("kernel-candidates.c", "kernel-candidates.h", "kernel-bench.c", "kernel-probe.c", "build_probe.py", "make_patches.py"):
    copy(lab / "review-deep-work-20261019/kernels" / name, Path("review-kernels") / name)
copy(projects / "returned-deep-review/xeon-deep-review-20261019/tools/build_variants.py", "tools/review-build-variants.py")
copy(projects / "returned-review/xeon-review-20261019/tools/capture_phase.py", "tools/capture_phase.py")
(destination / "00_START_HERE_JA.txt").write_text(
    "最初に report/REPORT_JA.txt を参照してください。詳しい数値と検証範囲は report/REPORT.json にあります。\n\n"
    "この結果はXeon E5-2630L v2 x2の実機測定です。通常用ビルドの生成改善は約2.6〜3.2%、短い日本語応答はほぼ同等でした。\n"
    "候補・個別変更・不採用案の数値も含みます。採用判断は最終レポートを優先してください。\n\n"
    "measurements/: 比較の条件、各プロセスの結果、ベンチマークのログ\n"
    "evidence/: 実応答、全語彙出力の一致検証、CPU/IMC計測、NUMA配置確認\n"
    "source-before/ と source-after/: 今回の変更対象ファイルの前後\n"
    "variants/combined/combined.patch: 今回の統合差分\n"
    "baseline-patches/: この実験以前に採用済みだった2つの差分\n"
    "active-configuration/: 現在の起動・再ビルド設定と、切替前の保存ファイル\n"
    "tools/: 再現・検証に用いたコード (元サーバーの絶対パスを含みます)\n\n"
    "元ソースのコミットは436f6f89e1e581249900b37a5b8a12a36a6d0912です。3つの差分は変更対象が重ならず、元ソースへのdry-run適用を確認しています。\n"
    "ログ内の /home/codex/projects/llm-lab/results/... は、同名ディレクトリを measurements/ または evidence/ で探してください。\n"
    "切替前rebuildスクリプトは当時の設定を保存した資料です。旧ソースの再現には元コミットとbaseline-patchesを使用してください。\n"
    "モデル、実行ファイル、巨大な生logitsファイル、perf.dataは含みません。出力ハッシュとテキストログを収録しています。\n"
    "MANIFEST.sha256は同梱ファイルの検証用です。\n")
payload = sorted(p for p in destination.rglob("*") if p.is_file())
manifest = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
(destination / "MANIFEST.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in manifest.items()))
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
    for path in sorted(p for p in destination.rglob("*") if p.is_file()):
        output.write(path, destination.name + "/" + str(path.relative_to(destination)))
with zipfile.ZipFile(archive) as check:
    if check.testzip() is not None: raise RuntimeError("ZIP CRC check failed")
    for name, digest in manifest.items():
        if hashlib.sha256(check.read(destination.name + "/" + name)).hexdigest() != digest:
            raise RuntimeError("Archive payload hash mismatch: " + name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n")
result = {"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size,
          "payload_files": len(manifest), "crc_passed": True, "all_payload_hashes_passed": True,
          "models_and_compiled_binaries_included": False}
destination.with_suffix(".package-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
