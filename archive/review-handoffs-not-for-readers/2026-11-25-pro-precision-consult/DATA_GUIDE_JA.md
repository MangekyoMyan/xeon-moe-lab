# レビュー資料の読み方

[相談本文](CONSULT_PRO_JA.md) に今回の依頼と運用者の目的がある。資料は11月24～25日の確定コミットに固定した。通常の推論実装、隔離試験にだけ入れた変更、試験終了時の状態は、各研究の `FINAL_STATE.json` または `SERVICE.json` で照合できる。

## 目的から元データを選ぶ

| 確認したい内容 | ファイル |
|---|---|
| 何をいつ測ったか | 70件の実行順、固定入力と目標トークンID、F0・候補・F1の計時を含む。[試験順](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/MODEL_DESIGN.json) · [入力ID](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/driver/FIXTURES.json) · [候補プロセスの1件ごとの結果](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/runs/candidate/measurements.jsonl) |
| しきい値の校正と対照 | Sは日本語96トークンの第0層で校正。Rはそのマスクを37位置回転、Aは第0層の全位置。補正はgate・upの選択入力列だけ。[対象を決めた時点の記録](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/DESIGN.json) · [融合CUDAの7位算出](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/backend/seventh.inc) · [判定・転送・残差補正の実装](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/backend/margin_core.inc) |
| 後段の誤差とrouterの入れ替わり | 73,728内積のFP64参照とGPU実測、全40層×各96位置×2課題での両経路の実際の選択ID、同数の循環シフトの分布。[内積原表](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/dot_reference.tsv) · [選択ID原表](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/reference-router-sets.tsv) · [トークン別の誤差](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/LOCAL_RESULT.json) · [層別の校正と感度](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/ROUTER_RESULT.json) |
| 最終候補に及んだ影響 | 固定された履歴の各位置について、1位・2位、他方での順位、確率、参照次トークンのNLL、TV、JS、KL、top-5共通数などがある。[品質集計](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/QUALITY_RESULT.json) · [通常と全補正・日本語の位置別TSV](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/risks/F-A/JA96.tsv) · [比較コード](https://github.com/MangekyoMyan/xeon-moe-lab/blob/05aa15cb990c1e2e05e5332bedeb9a910b87091d/archive/review-handoffs-not-for-readers/2026-11-24-prefill-arithmetic/precision/score_logits.cpp) |
| 本体の遅延と判断だけの費用 | F0/F1を外側対照とし、候補ではD/Sを各2回、R/Aを各1回。全モデルの入力完了時計、融合routerのCUDA event、CPU greedy走査を別々に記録した。[課題ごとの時計](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/TIME_RESULT.json) · [GPUの生計時](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/router_micro/samples.tsv) · [CPUの生計時](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/output_cost.tsv) |
| 同じ入力のCPUとCUDAで何が変わるか | しきい値を2要因で変えた全語彙の比較、第0層の実入力・expert ID・参照内積、単一層だけを置き換えた介入試験。本来のGGUF重みへの数値的な忠実さ、元のGPU版との一致、課題の正答は別の評価対象になる。[最初に差が出た場所](https://github.com/MangekyoMyan/xeon-moe-lab/blob/05aa15cb990c1e2e05e5332bedeb9a910b87091d/archive/review-handoffs-not-for-readers/2026-11-24-prefill-arithmetic/PRECISION_CAUSE.json) · [単一層の介入](https://github.com/MangekyoMyan/xeon-moe-lab/blob/05aa15cb990c1e2e05e5332bedeb9a910b87091d/archive/review-handoffs-not-for-readers/2026-11-24-prefill-arithmetic/CAUSAL_RESULT.json) · [実用課題と採点基準](https://github.com/MangekyoMyan/xeon-moe-lab/blob/05aa15cb990c1e2e05e5332bedeb9a910b87091d/archive/review-handoffs-not-for-readers/2026-11-24-prefill-arithmetic/CHAT_FIXTURES.json) |
| GPUの実装候補と局所コスト | 小さなバッチでの2回内積と列をまとめる方式。後者は入力列数が変わり、Voltaのネイティブ経路も切り替わりうる。独立したCUDAグラフで前後の順を入れ替えて測った。[残差生成カーネル](https://github.com/MangekyoMyan/xeon-moe-lab/blob/7ec7f304ea98ab05ea7bf56442ea4e9f3a70e442/archive/review-handoffs-not-for-readers/2026-11-25-volta-residual/residual.cu) · [演算プローブ](https://github.com/MangekyoMyan/xeon-moe-lab/blob/7ec7f304ea98ab05ea7bf56442ea4e9f3a70e442/archive/review-handoffs-not-for-readers/2026-11-25-volta-residual/probe.cpp) · [精度と経路別時間](https://github.com/MangekyoMyan/xeon-moe-lab/blob/7ec7f304ea98ab05ea7bf56442ea4e9f3a70e442/archive/review-handoffs-not-for-readers/2026-11-25-volta-residual/RESULTS.json) |
| 実機側の通常構成 | 24物理コアの固定、2ソケットNUMA、モデルの重み配置、通常起動条件、GPU、採用ライブラリを照合できる。[通常設定](https://github.com/MangekyoMyan/xeon-moe-lab/blob/081851f84f83ad575bd9c57d90d88679d42c1275/current/README.md) · [直近の実機確認](https://github.com/MangekyoMyan/xeon-moe-lab/blob/c2b14b339bcd94bd34e0d2402ac3a72e6bbd9d25/archive/review-handoffs-not-for-readers/2026-11-25-selective-margin/FINAL_STATE.json) |

GitHubで大きなTSVのプレビューが省略された場合は、ファイルの **Raw** または下記のZIPで全行を取得できる。比較元のCPU/GPU時系列とDRAM・OpenMPの元資料は別の確定版にもある。[24/24時系列のリリース](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-timeline-20261124)

## Linuxでの検算

[レビュー本体ZIP](https://github.com/MangekyoMyan/xeon-moe-lab/releases/download/precision-consult-20261125/xeon-pro-precision-review-20261125.zip) の直下に相談本文、公開時の3件の記録一式、通常設定のスナップショット、当時のソースと検算用スクリプトを置いた。フォルダは `evidence/arithmetic/`（CPU・CUDA）、`evidence/volta/`（V100単体）、`evidence/margin/`（僅差選別）、`runtime/current/`（運用台帳）である。各記録は元のファイル構成を保つ。Python 3.11以降で、展開したディレクトリから実行できる。

```bash
python3 -B replay_review.py
```

## 生の全語彙スコア

最新の選別試験については、公開済みの位置別TSVに加え、各位置 **129,280語すべて** の実測値も渡している。ファイルはリトルエンディアンのfloat32で、展開してそのまま読み込める。各ZIPに入力ID、行の対応、参照次トークン、元の実プロセスのSHA-256、比較コードを同梱した。

| 補足ZIP | 含まれる配列 |
|---|---|
| [固定文2件のF32](https://github.com/MangekyoMyan/xeon-moe-lab/releases/download/precision-consult-20261125/xeon-pro-fixed-logits-20261125.zip) | 日本語96とPython96。各65位置、F/S/R/Aの4条件。Dと2回目のFの全語彙はFと同一。 |
| [実用対話8件のF32](https://github.com/MangekyoMyan/xeon-moe-lab/releases/download/precision-consult-20261125/xeon-pro-chat-logits-20261125.zip) | 対象6課題は計145位置、F/S/R/A。入力28・173の対照2件は計30位置、実行したFとDの同一配列を各1本。 |

F0・F1・Dが同じバイト列のときはFを1回格納し、`LOGIT_INDEX.json` が元の全ファイル名と照合先を示す。各F32の1行は517,120バイト。固定文では行0が96入力の直後、続く64行は同じ参照トークンを一つずつ入力した直後。実用対話は通常の貪欲生成で得たIDを共通履歴として用いている。このため、固定文の参照語への一致と実用課題の採点では分母と意味が異なる。

補足ZIPをレビュー本体の展開先のひとつ上に置く場合は、次のコマンドで元のF32のハッシュと全5通りの比較（F-S、F-R、F-A、S-A、R-A）を照合できる。`--rescore` はC++17コンパイラを使用する。ハッシュと形状だけを確認するときはこの引数を省く。

```bash
python3 -B verify_logits.py --rescore ../xeon-pro-fixed-logits-20261125.zip ../xeon-pro-chat-logits-20261125.zip
```

ファイル一覧・バイト数・SHA-256は [GitHubの配布ページ](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/precision-consult-20261125) から取得できる。モデル本体が必要になる新しいGPU計測は、レビューを受け取った後に実機側で実施できる。
