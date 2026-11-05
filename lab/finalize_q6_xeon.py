#!/usr/bin/env python3
"""Finalize local, reviewable Q6 experiment artifacts after all runs finish."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import summarize_q6_xeon

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/optimization-q6-xeon-20261020"
ARCHIVE = LAB.parent / "xeon-q6-exploration-20261020.zip"


def main():
    active = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            name = (path/"exe").resolve().name
            if name.startswith("llama-") or name == "perf":
                active.append({"pid":path.name,"executable":name})
        except OSError:
            pass
    assert not active, active
    summarize_q6_xeon.main()
    report = json.loads((OUT/"REPORT.json").read_text())
    (OUT/"00_START_HERE_JA.txt").write_text("""Xeon Q6 探索の成果物

最初にREPORT_JA.txtを参照。詳細値はREPORT.json、変更はq6-xeon.patch。
候補はCPUのみで、今回のMelody1437 26B-A4B Q6_Kに対して検証した。

サーバー上の試験用起動:
  bash /home/codex/projects/llm-lab/run_melody_q6_xeon_experiment.sh --single-turn -p '日本語で短く自己紹介してください。'

元の通常起動スクリプトは保持されている。候補の1コピー4KiB配置は、常駐時の生成向け。
初期化には約27秒のコピー時間があり、生成時より一時的なメモリ消費が大きい。

このZIPは前回のfull-replica実験に対する追加成果物。
基準ソース: /home/codex/projects/llm-lab/review-full-replica-20261020/source
候補ソース: /home/codex/projects/llm-lab/review-q6-scales-20261020/source
候補ビルド: /home/codex/projects/llm-lab/review-q6-scales-20261020/build
前回の基盤変更: /home/codex/projects/xeon-full-replica-26b-20261020.zip

ZIPには変更対象2ファイルの前後、差分、試作kernel、計測スクリプト、ログ、検証結果を含む。
モデル、実行バイナリ、llama.cpp全ソース、約35MBずつの生logit配列は含まない。
生logitのハッシュと比較数はvalidation/comparison.jsonにある。
完全に別の環境でビルドするには、基準ソース一式とモデルが別途必要。

toolsのスクリプトは元サーバーのllm-lab配置を前提とする。
各ログのcommand/commandsが実際のコマンド。再実行時は新しい結果ディレクトリを指定する。
既存結果の上書きを避けるため、計測・ビルド用スクリプトは既存の出力先を拒否する。
MANIFEST_SHA256.jsonはZIPに含む各payloadのハッシュ。
""")
    review = """Q6 Xeon候補の私的コードレビュー

対象: ggml/src/ggml-cpu/arch/x86/quants.c と ggml/src/ggml-cpu/ggml-cpu.c の2ファイル。
リポジトリのskills/code-review/SKILL.mdに基づき、変更範囲・安全性・CPU backend・テスト・一般規約を確認。
外部投稿やPR作成は行っていない。

今回の検証範囲で、採用を妨げる正しさの問題は見つからなかった。

カーネル:
・変更は既存AVX分岐のみ。AVX2分岐は変更なし。
・先読みはi+2<nbで制限。オフセット0/64/128/192はいずれも210-byteブロック内。
・スケールの符号拡張を再利用。整数レーンの寄与と浮動小数点の加算順序は維持。
・型、行のstride、重み容量、GGUF形式を変更していない。
・生成された関数命令列のvpmovsxbwは18箇所から2箇所、vpshufbは8箇所から16箇所。
  これは静的な命令数であり、実行時間の削減割合ではない。

固定分担:
・GGML_CPU_Q6_STATIC_ROWS=1のみ有効。AVXでAVX2なし、Q6通常行列、入力1列、非参照経路に限定。
・入力変換と共有バリアを完了してから各threadの連続した行を割り当てる。
・商と余りで分割。各行は1回だけ処理し、余りや行数より多いthreadsも扱う。
・行がないthreadも必要な共有バリアには参加済み。MoEのMUL_MAT_ID割当は不変。
・CPU affinityを広げるNUMA設定を使用せず、0-23の厳密な固定を維持。

検証:
・境界付きのQ6/Q8バッファ、負のスケール、Q8=-128を含む入力、有限な半精度全bit値で基準関数と一致。
・test-quantize-fns、Q6 MUL_MAT/MUL_MAT_IDの16ケースを固定分担OFF/ONで通過。
・実モデル24threadsで8,912,896 logitsがA/B/Cでbit一致。
・同一の日本語応答が4回一致。速度は計測器を付けず別途確認。

残る制約:
・実機はこのXeonのみ。別CPU、長い会話、大きな入力batchの性能は未検証。
・固定分担は負荷偏りを悪化させる場合があるため既定ではOFF。試験用launcherだけでON。
・今回の高速化理由の細分化を示すPMU再計測は行っていない。
・事前展開は単独kernelのscreeningで不利だったため本体へ未統合。
"""
    (OUT/"CODE_REVIEW_JA.txt").write_text(review)
    tools = OUT/"tools"
    tools.mkdir(exist_ok=True)
    tool_names = [
        "prepare_q6_xeon.py","prepare_q6_xeon_v2.py","prepare_q6_expanded.py",
        "prepare_q6_full.py","q6-xeon-probe.c","run_q6_xeon_micro.py","run_q6_model.py",
        "validate_q6_model.py","run_q6_responses.py","summarize_q6_xeon.py",
        "finalize_q6_xeon.py","full_replica_experiment.py","bench_deepseek_light.py",
        "deepseek-logits-probe.cpp","without_thp.py","run_melody_q6_xeon_experiment.sh",
    ]
    for name in tool_names:
        shutil.copyfile(LAB/name,tools/name)
    for side, source in (("before", LAB/"review-full-replica-20261020/source"),
                         ("after", LAB/"review-q6-scales-20261020/source")):
        for relative in ("ggml/src/ggml-cpu/arch/x86/quants.c","ggml/src/ggml-cpu/ggml-cpu.c"):
            target=OUT/"source"/side/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/relative,target)
    for name in ("q6-xeon-20261020","q6-xeon-v2-20261020","q6-expanded-20261020"):
        target=OUT/"candidates"/name
        target.mkdir(parents=True,exist_ok=True)
        for path in (LAB/name).iterdir():
            if path.is_file() and path.suffix in (".c",".patch",".asm",".json",".log"):
                shutil.copyfile(path,target/path.name)
    audit={
        "normal_artifacts_unchanged":report["normal_artifacts_unchanged"],
        "active_llama_or_perf_processes":active,
        "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "numa_balancing":Path("/proc/sys/kernel/numa_balancing").read_text().strip(),
        "global_thp":Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().strip(),
        "launcher":str(LAB/"run_melody_q6_xeon_experiment.sh"),
        "candidate_cpu_library_sha256":hashlib.sha256(
            (LAB/"review-q6-scales-20261020/build/bin/libggml-cpu.so").read_bytes()).hexdigest(),
    }
    (OUT/"completion-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    preferences=LAB/"experiment-preferences.json"
    current=json.loads(preferences.read_text())
    (OUT/"preferences-before.json").write_text(json.dumps(current,ensure_ascii=False,indent=2)+"\n")
    current["before_q6_xeon_work"]=current["current_work"]
    current["current_work"]=(
        f"Completed AVX Q6 exploration on Melody 26B. Scale reuse plus bounded two-block prefetch "
        f"and opt-in static dense-row scheduling improved matched decode "
        f"{report['confirmation']['C']['gain_percent_vs_A']:.2f}% versus fine 4KiB single-copy baseline; "
        f"matched Japanese generation {report['responses']['generation_gain_percent']:.2f}%. "
        "All full-model logits and responses identical. Pre-expansion worsened streaming kernels. "
        "Candidate remains isolated with a separate launcher; normal and DeepSeek runtimes unchanged."
    )
    current["q6_xeon_results"]=str(OUT/"REPORT_JA.txt")
    current["q6_xeon_archive"]=str(ARCHIVE)
    current["q6_xeon_experimental_launcher"]=audit["launcher"]
    preferences.write_text(json.dumps(current,ensure_ascii=False,indent=2)+"\n")
    (OUT/"preferences-after.json").write_text(json.dumps(current,ensure_ascii=False,indent=2)+"\n")
    payload = [p for p in sorted(OUT.rglob("*")) if p.is_file()
               and p.suffix not in (".bin",) and not p.name.endswith("-logits")
               and p.name not in ("MANIFEST_SHA256.json","package-check.json")]
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
    (OUT/"MANIFEST_SHA256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    assert not ARCHIVE.exists()
    with zipfile.ZipFile(ARCHIVE,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in payload+[OUT/"MANIFEST_SHA256.json"]:
            archive.write(p,"xeon-q6-exploration-20261020/"+str(p.relative_to(OUT)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.read("xeon-q6-exploration-20261020/"+name)).hexdigest()==digest
    check={"archive":str(ARCHIVE),"bytes":ARCHIVE.stat().st_size,"payload_files":len(payload),
           "sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),"crc_ok":True,"payload_hashes_ok":True}
    (OUT/"package-check.json").write_text(json.dumps(check,indent=2)+"\n")
    print(json.dumps(check,indent=2))


if __name__=="__main__":
    main()
