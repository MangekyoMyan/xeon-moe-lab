#!/usr/bin/env python3
"""Write the measured findings and the limits of their interpretation."""
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
OUT = LAB / 'results/mxfp4-opt-20261023'
s = json.loads((OUT / 'SUMMARY.json').read_text())
p = json.loads((OUT / 'PROFILE_SUMMARY.json').read_text())
validation = json.loads((OUT / 'launcher-validation.json').read_text())
assert validation['passed']
def metric(phase, label): return s['primary']['phases'][phase][label]['pooled_tokens_s']
def gain(phase): return s['primary']['phases'][phase]['speed_gain_percent']
secondary = s['secondary']['phases']['tg256']
load_a = s['startup']['baseline']['load_ms'] / 1000
load_b = s['startup']['reuse']['load_ms'] / 1000
spin_a = sum(r['percent'] for r in p['baseline']['symbols'] if r['object'].startswith('libgomp') and r['symbol'] in {'0x0000000000024722','0x0000000000024726','0x000000000002472e','0x000000000002472b'})
spin_b = sum(r['percent'] for r in p['optimized']['symbols'] if r['object'].startswith('libgomp') and r['symbol'] in {'0x0000000000024722','0x0000000000024726','0x000000000002472e','0x000000000002472b'})
report = f'''Ivy Bridge Xeon向けMXFP4最適化・実機レポート
2026-10-23開始、日本時間10-24まで実施

結果
llama.cppのCPU処理とモデル読込みを変更し、同じgpt-oss-20b MXFP4ファイルで改善を確認した。
最終構成では入力処理が約20～21%高速化。256トークン生成は、固定した2つの試験系列で約4.5%／7.7%高速化した。
OSキャッシュにあるモデルのロード時間は、通常版の約{load_a:.2f}秒から約{load_b:.2f}秒へ短縮した。
重みの数値・量子化精度を変更していない。GPU・GPU予測シミュレーションは使用していない。

測定環境と対象
・Xeon E5-2630L v2 x2、12物理コア／24論理CPU、128GB ECC LRDIMM、2 NUMAノード。
・AVX／SSSE3／F16Cあり、AVX2／FMAなし。今回の最終設定は生成24／入力24スレッド。
・gpt-oss-20b、20,914,757,184パラメータ、24層、各層32 expertから4個を選択。
・ファイル12,109,566,624バイト。MXFP4 expert重み10,152,345,600バイト、共通部分は主にQ8_0とF32。
・取得先: https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/15dcfc7e7c608c6161507211ed43c5b590801b4b
・SHA256: 27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901
・比較元は既存のreview-glm-q8-columns4-20261020。過去のAVX・Q8改善を含む私用ビルドであり、無改造の上流版ではない。
・通常のDeepSeek／GLM実行環境は保持。CPUクロック、電力制限、NUMA balancing、システム全体のTHP設定は変更していない。
・両版ともプロセス内だけTHP無効、4KiBページでnode0/1へinterleave、CPU0-23へ固定。load-mode none、lazy off、GPU offload 0。

最終の性能比較
同一コマンド列をABBA・BAABの計8回、各テスト2反復。A/B各4起動・8サンプル。
集計値は総トークン数÷総測定秒数。プロファイラは使用せず、ロード・ウォームアップを除いた時間。

条件                     通常版       最終版       スループット改善
入力128トークン          {metric('pp128','A'):7.3f}      {metric('pp128','B'):7.3f} tok/s    +{gain('pp128'):.2f}%
入力512トークン          {metric('pp512','A'):7.3f}      {metric('pp512','B'):7.3f} tok/s    +{gain('pp512'):.2f}%
生成256トークン          {metric('tg256','A'):7.3f}      {metric('tg256','B'):7.3f} tok/s    +{gain('tg256'):.2f}%

追加の生成系列（ABBA4回、各2反復）:
{secondary['A']['pooled_tokens_s']:.3f} → {secondary['B']['pooled_tokens_s']:.3f} tok/s、+{secondary['speed_gain_percent']:.2f}%。
llama-benchはstd::rand()を使い、先行する入力テストが後続の生成トークン列を変える。
主比較はpp128,pp512,tg256、追加比較はpp128,tg256。各系列内でのみ前後比較する。
初期の並び替え版で観測した15.07 tok/s／約11%改善は別構成・別実行の値であり、最終版の改善率として採用していない。
改善幅は入力系列や処理量で変わる。上記はこのGGUFと短いコンテキストでのCPU実測であり、DeepSeek-V4.1-Flash全体やGPU併用時の速度へ換算していない。

採用した変更
1. MXFP4内積の整数計算を整理。4個の小さな整数積を16bitのまままとめてから32bitへ進める。
2. 4行分の重みをまとめ、入力データ・スケールの読み直しを減らす。符号処理は値に12を加えた表と整数補正で扱う。
3. 4行×32重みを68バイトに並べ直すCPU_REPACK形式を追加。元の4×17バイトと同容量で、数値は完全に保存する。
4. 入力処理では同じexpertを使う4入力間で重み展開を共有する。1入力の通常生成は4行版を使う。
5. 読込み時の並び替えを直接コピーにし、大きいテンソルはOpenMPで行グループを並列処理する。
6. 同期CPU_REPACK/MXFP4読込みで、同サイズの一時バッファを再利用する。サイズ・型・処理経路が変われば解放する。

実装上の条件
CPU用形式はAVXあり・AVX2/FMAなしのビルドで、3次元MXFP4 expert行列の出力行数が4の倍数の場合に選択する。
入力は既存のF32→Q8_0変換を利用する。FP32重み行列全体への展開は行わない。
通常のQ8_0変換が作る整数範囲[-127,127]を検証対象とする。浮動小数点の積・加算・最終縮約順を維持した。
新しいMoE経路では、同じexpert IDが複数回選ばれる場合も行対応表があふれないよう容量と境界を確認する。
一時バッファ再利用はCPU_REPACK、MXFP4、同期読込み、非mmap経路に限定。非同期GPUアップロードには適用しない。

読込みの改善
途中の並び替え版は、同サイズの約141MBバッファをテンソルごとに確保・解放し、ロード時間が約29秒まで増えていた。
直接コピーと並列化で約22秒へ短縮し、バッファ再利用で約{load_b:.2f}秒になった。
minor fault観測値は並列のみ約{s['startup']['parallel']['minor_faults_observed']:,.0f}回、再利用あり約{s['startup']['reuse']['minor_faults_observed']:,.0f}回。
通常版のロード時間約{load_a:.2f}秒に対し、最終版は約{(1-load_b/load_a)*100:.1f}%短縮。
これはファイルがOSキャッシュにある条件。HDDからのコールドロード時間ではない。
expert重みの常駐バイト数は同じ。一時領域は1テンソル分で、同じサイズが続く間だけ保持する。
約141MBの独立テストでは、全バイトの復元と直列／並列出力の一致を確認した。

残るボトルネックの観察
ロードとウォームアップ後の実モデル生成中に、別走行で約18秒の診断窓を採取した。
DRAM読出し: {p['baseline']['read_GBs']:.2f} → {p['optimized']['read_GBs']:.2f} GB/s。
IPC: {p['baseline']['IPC']:.3f} → {p['optimized']['IPC']:.3f}。命令を減らした最適化なので、IPC低下だけで性能低下とは判断しない。
L2未完了読出しを伴う停止/cycles: {p['baseline']['L2_pending_stall_cycle_fraction']*100:.1f}% → {p['optimized']['L2_pending_stall_cycle_fraction']*100:.1f}%。
最終版のサンプルはQ8内積42.60%、MXFP4内積39.75%。確認したOpenMPのpause待ちループは約{spin_b:.2f}%（通常版約{spin_a:.2f}%）。
従って、現在はQ8の共通部分、MXFP4 expert処理、メモリ待ち、同期のいずれも残っている。
読み出しが約39GB/sになったことは確認できたが、処理時間を「CPU何%・メモリ何%」に一意に分解できたわけではない。
今回の診断窓は両版ともディスク読出し0、major fault0、NUMA hint fault／migration増加0。
最終版でminor fault1,158回を観測した。発生元は切り分けておらず、ディスク待ちやNUMA移動とは扱わない。
診断窓はログマーカーから開始する近似区間。IMCはホスト全体、停止率とサンプル比率は重なり合い、壁時計時間の割合として加算しない。

検証
・基準を含む22のカーネル候補を探索。行列比較は{s['kernel_comparison_sets']}設定セット、{s['kernel_timed_samples']:,}反復測定。
・ガードページ、端数ブロック、全有限FP16スケール、MXFP4指数などを確認。
・CPU_REPACKの小行列1,296ケースを確認。重複ID、入力のbroadcast、1/24スレッドを含み、614,400個のF32出力が一致。
・大きな並び替えでは141,004,800バイト全体を元形式と照合し、並列結果も直列結果と一致。
・実モデルの127／511トークン入力から各34地点、合計13,673,984個のlogitsが基準とビット単位で一致。
・起動スクリプトから、計算「42」、記憶「青い傘731」、日本語NUMA説明の完成した全文が基準と一致。
・日付をまたぐとGPT-OSSのテンプレートが現在日付を自動挿入する。回答比較用テンプレートだけ基準日の2026-10-23へ固定した。
  通常の起動スクリプトやモデルファイルの日付処理は変更していない。
・最初の64トークン上限で途切れた回答試行と、CLIの静かなログを誤って失敗判定した試行は採用判定から除外し、記録を残した。
・観測したCPU温度の最大は{s['max_observed_cpu_temperature_C']:.1f}℃。最後のECC CE/UEは両CPUとも0、熱スロットルカウンター0。
・既存ランタイム／システム設定の前後監査が一致。今回の測定プロセスはすべて終了。

採用しなかった主な案
F16Cへの単純置換、先読みの増量、2行／2入力共有は、条件によって遅くなった。
4入力＋余り2入力の併用は4入力版を明確に上回らず、最終版には4入力版だけを入れた。
4行共有だけでは大容量の読出しが遅くなる条件があったため、同容量の連続配置と組み合わせた。

起動
bash /home/codex/projects/llm-lab/run_gpt_oss_mxfp4_optimized.sh
モデルを読み込んだまま対話する。1回だけの確認は --single-turn -p '質問' を追加する。
このスクリプトはCPU専用、コンテキスト4096、生成上限1024。通常のモデルのreasoning設定はauto。
検証時は再現性のため --temp 0 --seed 42 --reasoning-effort low を指定した。

成果物
REPORT_JA.txt: このレポート
SUMMARY.json / MODEL_BENCHMARKS.csv: 最終性能と全サンプル
KERNEL_SUMMARY.csv: 小規模比較の全集計
PROFILE_SUMMARY.json / model-profile-*/: 実モデルの診断データ
combined.patch / selected-config.json: 最終変更とソース・実行物のハッシュ
launcher-validation.json / reuse-validation.json / final-runtime-validation.json: 検証記録
model-performance.png,pdf / startup-performance.png,pdf: 実測図

通常ランタイム: 従来のDeepSeek/GLM用ファイルを保持。
最終候補: /home/codex/projects/llm-lab/review-mxfp4-final-20261023
実験ソース: /home/codex/projects/llm-lab/mxfp4-opt-20261023
'''
(OUT / 'REPORT_JA.txt').write_text(report)
print('Report written')
