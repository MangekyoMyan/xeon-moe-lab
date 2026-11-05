#!/usr/bin/env python3
"""Save measurements and a scoped interpretation of the Xeon Q6 exploration."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/optimization-q6-xeon-20261020"


def main():
    runs = []
    for path in sorted(OUT.glob("confirm-joint-*/summary.json")):
        data = json.loads(path.read_text())
        assert data["exit_code"] == 0 and data["max_disk_read_bytes"] == 0
        b = data["benchmarks"][0]
        assert (b["n_threads"], b["n_gen"], b["n_prompt"]) == (24, 128, 0)
        runs.append({"label":path.parent.name[-1],"tokens_s":b["avg_ts"],"file":str(path),
                     "boot_id":data["boot_id"],"library_sha256":data["cpu_library_sha256"]})
    assert len(runs) == 6 and len({r["boot_id"] for r in runs}) == 1
    by_label = {}
    for label in "ABC":
        rates = [r["tokens_s"] for r in runs if r["label"] == label]
        assert len(rates) == 2
        by_label[label] = {"process_rates":rates,"median_tokens_s":statistics.median(rates),
                           "min_tokens_s":min(rates),"max_tokens_s":max(rates)}
    a = by_label["A"]["median_tokens_s"]
    for label in "ABC":
        by_label[label]["gain_percent_vs_A"] = 100*(by_label[label]["median_tokens_s"]/a-1)
    response = json.loads((OUT/"responses-C/comparison.json").read_text())
    assert response["complete"] and response["responses_identical"]
    assert all(r["max_disk_read_bytes"] == 0 for r in response["responses"])
    timing = {}
    for label in "AC":
        rows = [r for r in response["responses"] if r["label"] == label]
        assert len(rows) == 2
        timing[label] = {name: statistics.median(r["timings_ms"][name] for r in rows)
                         for name in ("load","prompt eval","eval")}
    gain_response = 100*(timing["A"]["eval"]/timing["C"]["eval"]-1)
    latency = 100*(1-(timing["C"]["prompt eval"]+timing["C"]["eval"])/
                    (timing["A"]["prompt eval"]+timing["A"]["eval"]))
    tests = json.loads((OUT/"validation/comparison.json").read_text())
    assert tests["complete"] and tests["logits_bit_identical"]
    compiler = json.loads((OUT/"compiler-comparison.json").read_text())
    for path, digest in compiler["libraries"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert all(r["library_sha256"] == compiler["libraries"][str(
        LAB / ("review-full-replica-20261020/build/bin/libggml-cpu.so"
               if r["label"] == "A" else "review-q6-scales-20261020/build/bin/libggml-cpu.so"))]
               for r in runs)
    normal = {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
              for path,digest in compiler["normal_artifacts"].items()}
    assert all(normal.values())
    micro = {name:json.loads((OUT / path).read_text()) for name,path in (
        ("initial","micro-summary.json"),("scale_order","v2/micro-summary.json"),
        ("expanded","expanded/micro-summary.json"))}
    large = next(r for r in micro["expanded"] if r["name"] == "k2816-m262144-t24")
    effective = {name:262144*11*size/large["median_seconds"][name]/1e9
                 for name,size in (("original",210),("expanded",276))}
    patch = []
    for relative in ("ggml/src/ggml-cpu/arch/x86/quants.c","ggml/src/ggml-cpu/ggml-cpu.c"):
        old = (LAB/"review-full-replica-20261020/source"/relative).read_text()
        new = (LAB/"review-q6-scales-20261020/source"/relative).read_text()
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                         fromfile="a/"+relative,tofile="b/"+relative))
    (OUT/"q6-xeon.patch").write_text("".join(patch))
    report = {
        "complete":True,"model":"Melody1437-26B-A4B.i1-Q6_K.gguf",
        "configuration":"CPU24, strict CPU0-23, anonymous single copy interleaved 4KiB, process THP disabled",
        "labels":{"A":"Frozen fine-interleave baseline","B":"Q6 widened-scale reuse and two-block prefetch",
                  "C":"B plus static Q6 dense single-column row assignment"},
        "confirmation":by_label,"processes":runs,"confirmation_order":"ABCCBA",
        "repetitions_per_process":3,"tokens_per_repetition":128,
        "responses":{"timing_medians_ms":timing,"generation_gain_percent":gain_response,
                     "warm_prompt_plus_generation_latency_reduction_percent":latency,
                     "identical":True,"evaluated_tokens":response["responses"][0]["eval_tokens"],
                     "order":"ACCA"},
        "tests":tests,"micro":micro,"effective_weight_GB_s_large_matrix":effective,
        "effective_bandwidth_note":"Weight bytes / elapsed kernel time, not an IMC hardware measurement.",
        "normal_artifacts_unchanged":normal,
        "adoption":"Separate experimental launcher only; normal launchers and DeepSeek runtime unchanged.",
        "limitations":["Two processes per condition in the confirmation; no statistical confidence interval.",
                       "Microbenchmarks are synthetic raw valid quantized blocks, not full-model speed.",
                       "Q6 expansion retains both layouts in the probe and excludes conversion time.",
                       "No long-context benchmark, no GPU, no claim for IQ2 DeepSeek.",
                       "Sampled cycle hotspots are not exclusive compute or removable wall-time fractions."],
    }
    (OUT/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines = [
        "Xeon E5 v2 向け Q6 最適化探索 / 2026-10-20",
        "",
        "対象は26B-A4B Q6_K。前回の4KiB分散配置を基準にし、追加の効果を比較した。",
        "CPUのみ。2x E5-2630L v2、24スレッド、CPU0-23固定、128GB DDR3-1600。",
        "A=基準、B=Q6スケール再利用+2ブロック先読み、C=B+Q6通常行列の固定行分担。",
        "",
        "モデル全体の生成速度",
    ]
    for label in "ABC":
        r=by_label[label]
        lines.append(f"{label}: {r['median_tokens_s']:.4f} token/s、A比 {r['gain_percent_vs_A']:+.2f}%、プロセス範囲 {r['min_tokens_s']:.4f}-{r['max_tokens_s']:.4f}")
    lines += [
        "ABCCBA順、各条件2プロセス、1プロセス内は128トークンを3回。プロセス平均の中央値。",
        "初回ABCの探索値は確認値へ混ぜていない。同一boot・同じ18 CPU単位のコンパイルフラグ。",
        "全比較で追加HDD読み出し0。ロード時間を生成速度へ含めていない。",
        "",
        "実際の日本語応答",
        f"同じ質問、temp=0、seed=42、128トークン上限、実評価{response['responses'][0]['eval_tokens']}トークン。ACCA順、各2プロセス。",
        f"生成: A {timing['A']['eval']/1000:.4f}秒 -> C {timing['C']['eval']/1000:.4f}秒。生成速度 {gain_response:+.2f}%。",
        f"入力: A {timing['A']['prompt eval']/1000:.4f}秒、C {timing['C']['prompt eval']/1000:.4f}秒。",
        f"常駐後の入力+生成の待ち時間は {latency:.2f}% 短縮。4回答は全文一致。",
        f"参考load: A {timing['A']['load']/1000:.3f}秒、C {timing['C']['load']/1000:.3f}秒。",
        "両者とも前回の実験用コピー型バッファ。常駐向けで、冷間HDDロードの評価ではない。",
        "",
        "試した方向と結果",
        "1. Q6を少量ずつ展開する変更: スタック参照命令は減ったが、初回のキャッシュ内単独演算の速度は約16%低下。",
        "   一時退避の削減だけでは足りず、独立した計算を並行して進める効率を失う可能性がある。",
        "2. 既に16bitへ拡張したスケールを再利用: 同じ変換の繰り返しを減らす。重み形式・容量は不変。",
        "   命令列でも符号付き8bitから16bitへの拡張が18箇所から2箇所へ減った。",
        "3. 先読み距離2/4/8ブロック: 近い先読みを候補へ残した。遠い先読みは大きな行列で約21-30%悪化する条件があった。",
        "4. F16Cで半精度スケールを直接変換: テーブル参照に対する安定した改善なし。",
        "5. Q6の行をスレッドへ固定分担: 大きな行列で連続した行を担当。CPU固定と入力変換の共有バリアは維持。",
        "   対象はQ6の通常行列・1入力列・AVXのみ。MoE MUL_MAT_IDの仕事割当は変更していない。",
        "6. Q6をロード時相当に事前展開: 210 bytes/block -> 276 bytes/block、31.43%増。",
        "   キャッシュ内単独演算は約52%高速化したが、全語彙相当の大行列では12/24スレッドで約22/24%悪化。",
        f"   24スレッド時の重みサイズ/時間は、元 {effective['original']:.2f} GB/s、展開後 {effective['expanded']:.2f} GB/s。",
        "   この大行列では、読み出し量が増えた分に近い悪化であり、帯域制約が強いという解釈を支持する。",
        "   これはIMC実測値ではなく、有効な重みバイト数から計算した指標。モデル全体の上限値ではない。",
        "   全展開版は本体へ組み込んでいない。全Q6テンソルならGGUF分で約4.19GiB増える見積もり。",
        "",
        "正しさ",
        "候補を境界付きバッファで比較。14ブロック長x32種、Q8の-128やQ6の負スケールも含む。",
        "半精度の有限な全63,488ビットパターンも比較し、基準のAVX関数とビット一致。",
        "test-quantize-fns成功。MUL_MAT/MUL_MAT_IDのQ6ケースは固定分担OFF/ONそれぞれ16/16成功。",
        "CPU参照との比較であり、別GPUバックエンドを使ったものではない。",
        "24スレッドの実モデルでは、入力127トークン+後続位置で34x262144=8,912,896 logitがA/B/C全てビット一致。",
        "",
        "現在の扱いと次の方向",
        "通常のMelody/DeepSeekランチャーとactive CPUライブラリは変更なし。候補は隔離ビルドと別の試験用起動に残す。",
        "今回の変更はQ6向けで、IQ2/Q8を使うDeepSeekへの速度向上は主張しない。",
        "今後は重みを大きくせず、入力の再利用・同じ容量の内部配置・仕事の連続性を改善する方向が有望。",
        "ただし、どの方法が次に効くかは未検証であり、今回の結果から速度向上を保証しない。",
        "",
        "限界",
        "確認比較は各2プロセスで、信頼区間は出していない。長い会話履歴や大規模prefillは未測定。",
        "マイクロベンチは合成データで、展開時間・モデルの層間同期を含まない。実モデルの改善率に読み替えない。",
        "詳細値と再現コマンドはREPORT.json、各summary.json、validation/、responses-C/を参照。",
    ]
    (OUT/"REPORT_JA.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps({"confirmation":by_label,"response_gain_percent":gain_response,
                      "warm_latency_reduction_percent":latency},indent=2))


if __name__=="__main__":
    main()
