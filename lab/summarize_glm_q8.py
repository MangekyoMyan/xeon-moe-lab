#!/usr/bin/env python3
"""Assemble the measured GLM Q8 findings without extrapolating cold-load or CPU limits."""
import json

from glm_q8_experiment import LAB, OUT


def read(relative):
    return json.loads((OUT / relative).read_text())


def main():
    confirmation = read("column-confirmation.json")
    assert confirmation["complete"]
    validation = read("validation-review-glm-q8-columns4-20261020/comparison.json")
    assert validation["complete"] and validation["bit_identical"]
    stage = read("stage1.json")
    profile = read("observation-B/summary.json")
    light = read("observation-B-light/summary.json")
    assert light["valid"]
    launcher = read("launcher-verification.json")
    audit = read("after-audit.json")
    assert launcher["exit_code"] == 0 and launcher["response_identical_to_baseline"]
    assert audit["normal_runtime_unchanged"] and audit["system_settings_unchanged"]
    b, c = (confirmation["summary"][v] for v in ("B", "C"))

    def gain(key):
        return (c[key] / b[key] - 1) * 100

    def reduction(key):
        return (1 - c[key] / b[key]) * 100

    rows = []
    for path in sorted(OUT.glob("*/run.json")):
        data = json.loads(path.read_text())
        if not data.get("observations") or path.parent.name == "smoke":
            continue
        assert data.get("exit_code") == 0, path
        samples = data["observations"]
        temps = [v for row in samples for v in row.get("cpu_temperatures_c", {}).values()]
        rows.append({
            "label": path.parent.name, "elapsed_s": data["elapsed_seconds"],
            "max_rss_GiB": max(row["rss_kib"] for row in samples) / 1048576,
            "max_hwm_GiB": max(row["hwm_kib"] for row in samples) / 1048576,
            "max_observed_disk_read_bytes": max(row["disk_read_bytes"] for row in samples),
            "max_sampled_cpu_temperature_C": max(temps) if temps else None,
        })
    paired = [r for r in rows if r["label"].startswith(("columns-prefill-", "columns-response-"))]
    assert len(paired) == 8
    assert all(r["max_observed_disk_read_bytes"] == 0 for r in paired)
    assert all(r["identical"] for r in confirmation["responses"])
    assert {r["prompt"]["count"] for r in confirmation["responses"]} == {92}
    assert {r["generation"]["count"] for r in confirmation["responses"]} == {116}

    summary = {
        "complete": True,
        "model": {
            "name": "GLM-4.5-Air Q8_0", "parameters": 106852251264,
            "nominal_active_parameters_billions": 12,
            "repository": "ggml-org/GLM-4.5-Air-GGUF",
            "revision": "6d3b6301efec560414534beafe6a5cb2bf5940fa",
            "file_bytes": 113612251680,
            "file_GiB": 113612251680 / 2**30,
            "quantized_matrix_type": "Q8_0",
            "context_tested": 4096,
        },
        "hardware": {
            "board_dmi": "Sugon X9DR3-F", "cpu": "2 x Xeon E5-2630L v2",
            "physical_cores": 12, "logical_cpus": 24,
            "ram_GiB": 131972956 / 1048576,
            "dimm_population": "4 x 32 GB DDR3-1600; 2 populated channels per socket",
            "isa": "AVX/F16C/SSSE3; no AVX2/FMA",
            "gpu_used": False, "swap_used": False,
        },
        "conditions": {
            "threads": 24, "threads_batch": 24, "batch": 128, "ubatch": 128,
            "load": "mmap", "numa_policy": "interleave=0,1; CPU 0-23 pinned",
            "weights_replicated": False, "repack": False,
            "paired_order": ["B", "C", "C", "B"],
            "prefill_repetitions_per_process": 3, "prefill_tokens": 128,
            "japanese_prompt_tokens": 92, "japanese_generation_eval_runs": 116,
            "japanese_temperature": 0, "japanese_seed": 42,
            "japanese_reasoning": "off",
            "all_paired_runs_zero_observed_HDD_read": True,
        },
        "thread_comparison": stage["thread_medians"],
        "thread_24_gain_percent": (
            stage["thread_medians"]["24"] / stage["thread_medians"]["12"] - 1) * 100,
        "prefetch_comparison": stage["prefetch_medians"],
        "prefetch_gain_percent": stage["prefetch_gain_percent"],
        "candidate": {
            "build": str(LAB / "review-glm-q8-columns4-20261020/build"),
            "change": "Share one Q8 weight row across four MoE prefill input columns on AVX-only CPUs",
            "paired_summary": confirmation["summary"],
            "pp128_throughput_gain_percent": gain("pp128_tok_s_median"),
            "japanese_prefill_throughput_gain_percent": gain("response_prompt_tok_s_median"),
            "japanese_generation_throughput_change_percent": gain("response_generation_tok_s_median"),
            "japanese_prompt_time_reduction_percent": reduction("response_prompt_ms_median"),
            "japanese_eval_total_time_reduction_percent": reduction("response_eval_ms_median"),
            "fixed_sequence_logits_bit_identical": 5152768,
            "japanese_responses_byte_identical": True,
            "all_backend_q8_tests_passed": 128,
        },
        "adoption": {
            "scope": "GLM-only launcher; baseline and other model launchers preserved",
            "launcher": str(LAB / "run_glm_air_q8_optimized.sh"),
            "launcher_verification": launcher,
            "cpu_library_sha256": audit["candidate_cpu_sha256"],
            "normal_runtime_unchanged": audit["normal_runtime_unchanged"],
            "system_settings_unchanged": audit["system_settings_unchanged"],
        },
        "prefill_baseline": read("prefill-B/run.json")["benchmarks"],
        "batch512_baseline": read("prefill-b512/run.json")["benchmarks"],
        "candidate_screen": read("columns-screen/run.json")["benchmarks"],
        "initial_load": {
            "observed_setup_to_first_generation_marker_s": 1267.2269,
            "strict_cold_test": False,
            "note": "Some file pages already cached; source preparation occurred during loading. Excluded from warm performance comparison.",
        },
        "bottleneck_light_counters": light,
        "bottleneck_profiles": profile,
        "selected_weight_estimate": read("selected-weight-estimate.json"),
        "telemetry": rows,
        "row_micro_experiment": read("row-micro/summary.json"),
        "limitations": [
            "Cycle, stall, memory-latency and wait-loop fractions overlap; they are not additive wall-time fractions.",
            "IMC counters are system-wide and diagnostic runs have observer overhead.",
            "LFB samples do not identify the eventual local or remote DRAM source; QPI traffic is not measured.",
            "No claim about long context, all prompt distributions, model quality, or absolute hardware bandwidth ceiling.",
            "mmap interleave policy does not relocate already cached file pages.",
            "The small generation throughput difference is not attributed to the prefill-only kernel.",
        ],
    }
    metrics = light["metrics"]
    max_rss = max(r["max_hwm_GiB"] for r in rows)
    max_temp = max(r["max_sampled_cpu_temperature_C"] or 0 for r in rows)
    candidate = summary["candidate"]
    report = f"""GLM-4.5-Air Q8 実測・最適化結果（2026-10-20）

結果
113.61 GBのQ8モデルが、128 GB構成の2ソケットXeonでCPUのみで動作した。
通常の生成速度は約2.37 tokens/s。入力は新しいQ8カーネルで改善した。
確認測定: pp128 {b['pp128_tok_s_median']:.4f} -> {c['pp128_tok_s_median']:.4f} tokens/s
  スループット +{candidate['pp128_throughput_gain_percent']:.2f}%。
日本語入力: {b['response_prompt_ms_median']/1000:.3f} -> {c['response_prompt_ms_median']/1000:.3f}秒
  入力時間 -{candidate['japanese_prompt_time_reduction_percent']:.2f}%。
日本語生成: {b['response_generation_tok_s_median']:.4f} -> {c['response_generation_tok_s_median']:.4f} tokens/s。
日本語の入力+生成: {b['response_eval_ms_median']/1000:.3f} -> {c['response_eval_ms_median']/1000:.3f}秒
  時間 -{candidate['japanese_eval_total_time_reduction_percent']:.2f}%。
入力+生成はログのprompt evalとevalを加算した値。モデルロードとサンプリング等は含まない。
生成速度の小さな変動は、入力専用カーネルの効果とは判断しない。

実機とモデル
DMI: Sugon X9DR3-F。Xeon E5-2630L v2 x2、12物理コア/24論理CPU。
AVX/F16C/SSSE3あり、AVX2/FMAなし。DDR3-1600 32 GB x4、各ソケット2チャネル。
OSが認識するRAMは125.86 GiB。swapなし。HDD保存。GPUは使っていない。
モデル: GLM-4.5-Air、106.85B parameters、公称アクティブ約12B、glm4moe。
128 expertsから8選択、共有expert 1、先頭dense層1、46層。
459個の行列テンソルがQ8_0。小さなnorm/bias等321個はF32。
低ビット混合によって100 GBにしたモデルではない。
ファイル合計113,612,251,680 bytes = 113.61 GB = 105.81 GiB。
第一分割はメタデータ、第二分割に重みがある。両方必要。結合せず第一分割を指定する。
配布元: https://huggingface.co/ggml-org/GLM-4.5-Air-GGUF
固定revision: 6d3b6301efec560414534beafe6a5cb2bf5940fa
取得時に全体SHA256検証済み。今回、検証記録・実ファイルサイズ・更新時刻を照合した。
download-verified.jsonと事前準備のheader-inspection.jsonを参照。

比較条件
基準B: build-avx-prefetch。試作C: review-glm-q8-columns4-20261020/build。
CPU 0-23へ固定、NUMA interleave=0,1、入力24/生成24スレッド。
CPUのみ、mmap、重み複製なし、repackなし、batch/ubatch=128。
pp128は各プロセスでウォームアップ後3回、B,C,C,Bの順で比較。
表の代表値は各ビルドの2プロセス平均速度の中央値。
日本語は同一質問、temperature=0、seed=42、reasoning off、context=4096。
92入力トークン・116生成evalで4回の回答本文はバイト一致した。
比較8プロセスは観測したHDD読み出しが全て0。プロファイラとは同時に走らせていない。
CPU16コンパイル単位のフラグは基準版と試作版で一致する。
mmapのNUMA指定は既にページキャッシュにある重みの配置を強制移動しない。

既存最適化の効果
12 -> 24スレッド生成: {stage['thread_medians']['12']:.4f} -> {stage['thread_medians']['24']:.4f} tokens/s
  +{summary['thread_24_gain_percent']:.2f}%（24,12,12,24の順、各tg64を2回）。
Q8先読みなしA -> ありB: {stage['prefetch_medians']['A']:.4f} -> {stage['prefetch_medians']['B']:.4f} tokens/s
  +{stage['prefetch_gain_percent']:.2f}%（A,B,B,A、各tg128を2回）。実質横ばいと扱う。
既存の8192幅Q8四行処理は、このモデルの4096/1408/10944/12288幅には適用されない。
IQ2専用、Q6専用の最適化も今回のQ8行列には適用されない。
基準版pp512/24スレッドでbatch128は8.866、batch512は8.928 tokens/s。
batch拡大だけの差は約0.7%。本比較では128を維持した。

新しい変更
MoEの入力処理で、Q8の重み1行を4入力列へ使い回す。
重みロードと一部の変換を共有し、各出力の積和順序は元の処理と揃える。
モデルの量子化形式、品質設定、重みの全体配置は変更していない。
AVXあり/AVX2・FMAなし、Q8_0、複数トークン、非参照実行に限定。
1トークンずつの生成と残り1～3列は既存処理。
全重みのFP16展開、全重みの両ソケット複製、大きな追加バッファは使わない。
変更3ファイルとパッチ、生成・検証スクリプトを保存した。

正確性
保護ページ付き3,072出力と有限half 63,488パターンの一致。
test-quantize-fns成功、Q8 MUL_MAT/MUL_MAT_ID 128/128成功。
dense/MoEグラフで1/4/12/24スレッド、端数、非連続入力、重複expert等を比較。
実GLMの固定系列で5,152,768ロジットが全て有限、全てビット一致。
日本語4応答も完全一致。これは検証した入力の一致であり、全入力の証明ではない。
CODE_REVIEW_JA.txt、validation-*/comparison.jsonを参照。

生成中のボトルネック
観測負荷を抑えた18秒カウンタ窓:
  DRAM読み取り {metrics['dram_read_GB_s']:.2f} GB/s
  socket0 {metrics['socket0_read_GB_s']:.2f} / socket1 {metrics['socket1_read_GB_s']:.2f} GB/s
  IPC {metrics['ipc']:.3f}
  L2未完了読み出しを伴う停止 {metrics['cycle_activity.stalls_l2_pending:u_fraction_of_cycles']*100:.1f}% / cycles
  全停止 {metrics['cycle_activity.stalls_total:u_fraction_of_cycles']*100:.1f}% / cycles
  HDD読み出し {metrics['window_disk_read_bytes']} bytes、major/minor faults {metrics['major_faults']:.0f}/{metrics['minor_faults']:.0f}
  CPU移動 {metrics['migrations']:.0f}、温度最大 {metrics['cpu_temperature_max_C']:.0f} C、
  thermal throttle増分 {metrics['throttle_delta_sum_including_smt_duplicates']}。
別のcyclesサンプリングではQ8内積に約90.45%、OpenMPに約6.81%。
逆アセンブルでpauseループと確認できた待ち合わせは約6.50%。
遅いloadのサンプル遅延重みの約96.83%がQ8内積内にある。
LFB/MABが多いが、その値から最終的なlocal/remote DRAM比率は求められない。
これらの割合は重なり合い、壁時計時間の独立した内訳ではない。足してはいけない。
IMCはシステム全体、サンプル観測には負荷がある。通常速度は別の無プロファイル測定を使う。
初回のnuma_maps走査付きカウンタ窓では31.42 GB/sだった。
補測では巨大なマップの走査を外した。生データは両方残している。

解釈
生成では毎トークンの大量のQ8重み読み出しと、その内積処理が中心。
構造から求めた選択重みの量は約13.68 GB/token（実DRAM交通量そのものではない）。
約2.37 tokens/sを掛けると約32.4 GB/sとなり、観測したDRAM読み取りと概ね整合する。
待ち合わせは存在するが、それだけを除いて桁違いに速くなる証拠はない。
今回の入力処理には重み再利用の余地があり、そこで改善を得られた。
生のメモリ帯域限界を別途測った試験ではないため、「ハードの絶対限界」とは断定しない。
QPI交通量は直接測っていない。26Bモデルの両側複製結果をこのモデルへ外挿しない。

見送った案
生成向けの2行/4行同時計算を、実モデル幅を含む5形状、1/12/24スレッド、
16交互ラウンドで比較した。出力一致は通った。
小さなキャッシュ内行列では大きく速くても、大きな4096/1408幅では横ばい又は低下。
全モデルの改善根拠にならないため、生成向け行共有は組み込まなかった。
row-micro/summary.jsonに全結果を残した。新しい入力4列処理とは別の実験である。

メモリ・ロード
今回の通常測定で観測した最大HWMは{max_rss:.2f} GiB、CPU温度最大{max_temp:.0f} C。
モデルを常駐させた生成でHDD待ちは発生していない。
最初のロード準備から最初の生成開始マーカーまでは約1267秒（21.1分）。
既存キャッシュがあり、ロード中にソース準備もしたので厳密なcold比較ではない。
その後の日本語基準測定のload timeは7.40秒。HDDから毎回7秒で読めるという意味ではない。
OSのMemAvailableには回収可能なファイルキャッシュが含まれるため、
その数字だけでモデル常駐や余裕を判断していない。
context=4096を確保した短い応答を検証した。4096全長や128Kでの速度・安定性は未測定。

利用・再現
GLM専用ランチャー: /home/codex/projects/llm-lab/run_glm_air_q8_optimized.sh
  bash /home/codex/projects/llm-lab/run_glm_air_q8_optimized.sh -p '日本語で自己紹介してください。'
入力24/生成24、context4096、batch128、CPUのみ、reasoning off、既定最大256トークン。
任意引数を末尾へ渡せる。測定時は--temp 0 --seed 42 -n 192を使用した。
通常のDeepSeek/Melodyランチャーと基準ライブラリは保存し、GLM専用に新ビルドを使う。
最終確認はlauncher-verification.jsonとafter-audit.jsonを参照。
比較の再現はglm_q8_experiment.pyとrun_glm_q8_column_confirmation.py。
個々のrun.jsonに正確なコマンド、ビルドSHA、時刻、テレメトリがある。
結果ラベルの上書きを防いでいるため、再実行する際は別の結果ディレクトリを使う。
"""
    (OUT / "REPORT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "REPORT_JA.txt").write_text(report)
    print(json.dumps({"candidate": candidate, "max_hwm_GiB": max_rss,
                      "max_sampled_cpu_temperature_C": max_temp,
                      "light_metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
