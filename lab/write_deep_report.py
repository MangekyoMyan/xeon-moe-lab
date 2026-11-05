#!/usr/bin/env python3
"""Write the completed measured-results report from saved evidence."""
import datetime
import json
from pathlib import Path

lab = Path(__file__).resolve().parent
results = lab / "results"
def read(name):
    return json.loads((results / name).read_text())

adoption = read("deep-review-source-adoption.json")
functional = read("deep-review-functional.json")
short_abba = read("deep-review-short-response-abba.json")
rebuild = read("deep-review-rebuild-check/validation.json")
if not adoption["launcher_switched"] or not functional["all_passed"] or not rebuild["all_passed"]:
    raise SystemExit("Complete adoption and acceptance before writing the final report")
rounds = read("deep-review-round-summary.json")["rounds"]
canonical = [r for r in rounds if r["note"].startswith("Current launcher build versus canonical combined rebuild")]
profiles = {key: read("deep-review-profile-" + key + ".json")
            for key in ("baseline-decode", "final-decode", "final-prefill")}
logits = [json.loads(p.read_text()) for p in sorted(results.glob("deep-logits-20261019T*/comparison.json"))]
if not logits or not all(r.get("all_outputs_bit_identical") for r in logits):
    raise SystemExit("Full-model exactness evidence is incomplete")
smoke_paths = sorted(results.glob("deepseek-functional-adopted-deep-launcher-*/summary.json"))
if len(smoke_paths) != 1:
    raise SystemExit("One adopted launcher smoke check is required")
smoke = json.loads(smoke_paths[0].read_text())
if smoke["exit_code"] or not smoke["semantic_checks_passed"]:
    raise SystemExit("Adopted launcher smoke did not pass")
report = {
    "schema": 1, "finished_jst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
    "hardware": "Sugon X9DR3-F DMI, two Xeon E5-2630L v2, 12 physical / 24 logical CPUs, 128 GB installed, AVX/F16C/SSSE3 without AVX2/FMA, two NUMA nodes, HDD.",
    "model": "Existing DeepSeek-V4-Flash IQ2_XXS-XL, 73.126 GiB, 284.3346B stored parameters. Not V4.1.",
    "upstream_commit": "436f6f89e1e581249900b37a5b8a12a36a6d0912",
    "review_archive_sha256": "85544b603a5dd8e09af7b03017b9485fe02819f772bf4d75a9f373865a6e91cd",
    "decision": {"adopted": True, "default": "Paired IQ2 MoE gate/up and four-row Q8 AVX kernel only for K8192 single-column cases, on top of the prior SSSE3 unpack patch.",
                 "canonical_generation_gain_percent": [r["gain_percent"] for r in canonical],
                 "numa_replica_default": False, "numa_replica_opt_in": "GGML_CPU_NUMA_REPLICA=1",
                 "workers": {"generation": 12, "prefill": 24}, "batch": 512, "ubatch": 512,
                 "gpu": "Disabled per user's cooling pause", "model_weights_changed": False},
    "method": {"performance_units": "Fresh-process means; ABBA and BAAB are separate rounds. Within-process repetitions are not independent processes.",
               "decode_screen": "p0 n32 depth0 r3 t12, batch512 ubatch512, strict CPU0-23, OS interleave0,1, mmap, warm weights, OpenMP ON and BLAS OFF unless explicitly compared.",
               "prefill_screen": "p512 n0 depth0 r1 t24; same settings otherwise.",
               "caution": "Synthetic token stream changes with p/n/r; compare only matched workloads. Diagnostic profile timings and single-run Japanese timings are not the controlled throughput claims."},
    "comparisons": rounds, "numa_resource_cost": read("deep-review-resource-costs.json"),
    "microbenchmarks": read("deep-kernel-micro-20261019T110518Z/comparison.json"),
    "full_model_exactness": logits, "functional": functional, "short_response_abba": short_abba,
    "normal_conversation_branch_coverage": read("deep-review-natural-branch.json"),
    "profiles": profiles, "rebuild_validation": rebuild, "source_adoption": adoption,
    "launcher_smoke": {"summary": str(smoke_paths[0]), "response": smoke["response"], "passed": True},
    "numa_placement": {"source": str(results / "numa-replica-placement-20261019T112648Z/placement.json"),
                       "tensors": 107, "replicated_weight_bytes_per_node": 3253207040,
                       "physical_placement_confirmed": True, "worker_fallback_calls": 0},
    "bottleneck_interpretation": "Decode remains a mix of quantized dot-product work, memory stalls and synchronization. IMC bandwidth is below the earlier streaming test, but different access patterns prevent calling that unused bandwidth freely available. Prefill is more compute-oriented. The small NUMA replication gain does not support a large gain from localization alone.",
    "limits": ["Exact logits checked at 34 positions (126-159) of one fixed 160-token sequence, not arbitrary prompts or contexts.",
               "Two Japanese prompts check recall, capital and arithmetic; no broad model-quality evaluation.",
               "Performance applies to the tested CPU, quantization, shapes, software revision and warm-cache conditions.",
               "NUMA replica is a private Linux two-node experiment with selected Q8 shapes, disabled by default.",
               "No matched Ryzen run, RAM expansion, GPU test, or cold-HDD throughput claim in this round."],
    "next_candidates": ["Profile remaining scalar Q8 and IQ2 kernels by real matrix shape; keep the exact-output requirement.",
                        "Explore an expert-level gate/up/activation/down pipeline with explicit synchronization and graph correctness tests.",
                        "Treat smaller IQ2 lookup tables or alternative weight layouts as new hypotheses; current large LUT/repack variants regressed on this Xeon.",
                        "Speculative generation needs correct cache/state rollback and a suitable draft mechanism before throughput claims."],
}
out = results / "deep-review-final"
out.mkdir(exist_ok=True)
(out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
lines = [
    "Xeon / DeepSeek 実機最適化結果 (2026-10-19)", "",
    "今回の変更は通常ランチャーへ適用済みです。追加メモリなしのMoE共有処理とQ8演算を採用しました。",
    "通常用ビルドの生成比較では約2.6〜3.2%改善。実験用ビルドでは約4.6〜5.1%でした。通常利用向けの基準は約3%とします。",
    "短い日本語応答の反復比較は約0.13%差で、実用上はほぼ同等でした。用途による差があります。元の実行ファイル・ランチャーも保存しています。", "",
    "通常用の実行ファイル同士の最終比較: " + "; ".join(f"{r['order']} {r['baseline_median']:.6f} -> {r['candidate_median']:.6f} tok/s ({r['gain_percent']:+.3f}%)" for r in canonical) + "。", "",
    "実機: Xeon E5-2630L v2 x2、12コア/24スレッド、128GB、AVX/F16C/SSSE3、AVX2/FMAなし。DMIはSugon X9DR3-F。",
    "モデル: 既存のDeepSeek-V4-Flash IQ2_XXS-XL、約73.13GiB。V4.1の測定ではありません。GPUは使用していません。", "",
    "測定結果 (tok/s。各値は独立したプロセスの平均値を集計した中央値)",
]
for row in rounds:
    lines.append(f"  {row['note']} [{row['order']}]: {row['baseline_median']:.6f} -> {row['candidate_median']:.6f}, {row['gain_percent']:+.3f}%")
lines += ["", "採用した変更", "  1. 単一トークンのIQ2 MoE gate/up計算で、入力量子化・expert振り分け・同期の一部を共有。43層で実際に通ることを確認。",
          "     同じ演算カーネルとclampを維持。重み・数式・モデル設定は変更していません。",
          "  2. Q8を4行ずつ処理するAVXカーネルをK=8192・入力1列だけに適用。実機の単体比較では約13〜14%改善。",
          "     K=1024では約32%悪化したため、その形状には適用していません。",
          "  3. NUMA複製を選択可能に実装。107行列、約3.03GiB分を各CPUのローカルメモリに配置。実際の全ページ配置も確認。",
          "     統合版への上積みは約0.85%、追加RSSは約6.06GiB、計測開始まで約4.9秒増。通常は無効です。",
          "     この約4.9秒はロード・初期化・ウォームアップを含む概測で、純粋なロード時間ではありません。", "",
          "     毎回モデルを起動する短い応答では追加の準備時間を回収しにくいため、常用設定には含めていません。", "",
          "正しさの検証", "  個別候補と統合版(NUMA複製の有無の両方)で、34位置 x 129280語彙 = 4,395,520個のfloat出力が元とビット単位で一致。",
          "  入力は固定127トークン+33継続トークン。128境界をまたぐ検証です。任意の文脈全体への保証ではありません。",
          "  行列の端数、view/broadcast、複数スレッド、NUMA両ノード、バッファ書き込み・コピー、guard pageも検証。",
          "  通常用に再ビルドしたCPUコンパイル条件・変更ファイルを照合し、量子化テストと3つのプローブも再確認。", "",
          "日本語の実応答 (各1回の機能確認。速度比較の統計には使用していません)"]
for run in functional["runs"]:
    lines.append(f"  {run['variant']} / {run['prompt_tokens']}入力トークン: 内容確認OK、時間(ms) {run['timings_ms_single_run_only']}")
lines += ["  元と変更後の応答は短文・長文とも一致。BLUE-731、東京、42を回答。1行指定への厳密な追従は判定していません。",
          f"  別途行った59トークン入力の実応答ABBA: 生成時間 {short_abba['median_ms']['A']['eval']:.2f} -> {short_abba['median_ms']['B']['eval']:.2f} ms、生成速度 {short_abba['generation_throughput_gain_percent']:+.3f}%。",
          "  合成ベンチと実応答で改善率は異なります。すべての会話が同じ率だけ速くなるという結果ではありません。", "", "変更後のボトルネック (perf/IMCの診断窓)"]
for key, p in profiles.items():
    lines.append(f"  {key}: DRAM read {p['dram_read_GB_per_second']:.2f} GB/s, write {p['dram_write_GB_per_second']:.2f} GB/s, IPC {p['ipc']:.3f}, OpenMP {p['openmp_percent_sampled_cycles']:.2f}% cycles")
    lines.append(f"    L2ミス未解決時の実行停止サイクル: {100*p['cycles_stalled_with_l2_miss_pending_fraction']:.2f}% (帯域律速の時間割合そのものではありません)")
    if p["phase"] == "decode":
        lines.append(f"    Q8 dot {p['q8_dot_percent_sampled_cycles']:.2f}%, IQ2 dot {p['iq2_dot_percent_sampled_cycles']:.2f}% cycles")
lines += ["  演算・メモリアクセス待ち・同期が混在しています。関数内のサイクルはメモリ待ちも含み、OpenMP比率は経過時間の割合ではありません。",
          "  以前の連続読み出しテストより帯域が低くても、アクセスパターンが違うため、その差をそのまま高速化余地とは扱えません。",
          "  ローカル重みの効果が約1%だった一方、同じ重みを使う準備・同期の共有で約3%改善。帯域だけで性能が決まる状態とは判断できません。", "",
          "通常の実行", "  /home/codex/projects/llm-lab/run_deepseek_flash.sh", "  入力24、生成12スレッド、batch/ubatch512、mmap、OS側NUMA interleaveを継続。reasoningの通常設定も維持。",
          "  NUMAを試す場合: GGML_CPU_NUMA_REPLICA=1 /home/codex/projects/llm-lab/run_deepseek_flash.sh",
          "  旧版を直接実行: /home/codex/projects/llm-lab/run_deepseek_flash_before_deep_review.sh",
          "  再ビルド: /home/codex/projects/llm-lab/rebuild_deepseek.sh", "",
          "見送った案と次の検討", "  大きいIQ2 LUTは実機で約41〜43%悪化。索引repackやOpenMPを外す案も採用を正当化する改善を示していません。",
          "  次は残るQ8/IQ2の形状別カーネル、expert単位の処理パイプライン、小さいLUTなど。投機生成は状態復元の正しさを先に扱う必要があります。",
          "  RAM増設・Ryzenとの同条件比較・GPU・HDDのコールド起動は今回の結論に含めません。", "",
          "REPORT.jsonに測定値・生ログの場所・ハッシュ・検証範囲を収録。ZIPには差分、前後のソース、再現用スクリプト、ログを同梱。モデルとバイナリは含めません。"]
(out / "REPORT_JA.txt").write_text("\n".join(lines) + "\n")
print(out)
