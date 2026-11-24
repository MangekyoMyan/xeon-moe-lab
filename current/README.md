# x9ram の通常推論構成

2026年11月24日に通常APIで受入を確認したDeepSeekの構成です。Supermicro系X9DR3-F（DMIの製造元はSugon）、E5-2697 v2×2、512 GiB、PG500-216を使用しています。CPUライブラリの採否と実際の配備先は下記に記載しています。

**現在のAPIは元のCPUライブラリ（SHA256先頭 `33a9e346`）を使っています。** この版にはrow-local配置コードがあり、通常設定では無効です。導線を切り出した簡素化版（`2626bb83`）は実機で出力・配置を検証済みです。旧CPU上の生成が2課題ともわずかに遅かったため、比較用に保存しました。[稼働版とバイト一致するキャッシュのソース](https://github.com/MangekyoMyan/xeon-moe-lab/blob/169a2f7eab8d52071661f0da6859d38df4206f8b/archive/review-handoffs-not-for-readers/2026-11-19-fbc-h1-iq08/cpu08/control/source/xeon-local/iq3-cache.cpp)

通常の待受で、同じ短文への生成速度は 9.76 tok/s、最初の本文は1.83秒でした。[24/24の採用記録](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-adoption-20261124)

| 通常設定 | 値 |
|---|---|
| モデル | OrcaRouter DeepSeek-V4.1-Flash-1017-abliterated、Mix-256G、7分割 |
| コンテキスト | 131,072、K/Vともにf16、1スロット |
| 入力処理・生成 | 入力24・生成24物理コア（各ソケット12）、batch/ubatchともに1024 |
| expertの配置 | routed expertの0–36層はCPU、37–39層はGPU |
| RAMの重みキャッシュ | 予算120 GiB、80テンソルの19,200件のexpert射影に127,401,984,000バイトを割当て |
| メモリ配置 | NUMAノード0・1の交互配置。プロセスのTHPは無効 |
| GPU | 150 W、ECC有効、sm70。画像用のF16 projectorもロード |

4つのライブラリについて、配備ファイルのパス・SHA256・読み込み順を確認できます。モデルの配布リビジョン、全7ファイルと画像projectorのSHA256、選択計画、サービス設定のSHA256も同じ台帳に収録しています。[実行構成の台帳](runtime.json)・[このモデルのexpert選択計画](models/deepseek41-mix-256g/frequency19200.plan)

リポジトリのルートで実行すると、手元のLinuxからプロファイルの数値を表示できます。

```bash
python3 current/check.py show
```

x9ramでは以下で実ファイルと稼働プロセスを照合できます。稼働中の確認には、プロセスの環境と保護監視記録を読める権限が必要です。

```bash
sudo python3 /home/codex/projects/llm-lab/current/check.py verify --live
```

通常の照合は本体と画像projectorの全ファイルサイズ、およびダウンロード時の検証記録を確認します。重み本体をもう一度すべて読みたい場合は `verify --full-weights` を指定してください。

通常APIを起動するときは、サーバーで既存のコマンドを実行してください。

```bash
sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/start.py
```

IQ compactのキャッシュヒット時はあらかじめ展開した重みを使い、ミス時は元の量子化重みを読みます。MXFP4は入力側の準備結果を再利用します。CUDA側はexpert IDを安定ソートして複数入力をまとめます。これらの演算経路と、モデル別のexpert選択計画、CPU/GPUの配置・メモリ予算・ワーカー数は別々に管理します。[CPU交換前後の実測](CPU_UPGRADE.md)

CPU側の簡素化差分では、特定のCPU番号に行を結び付けるコードと配置監査の導線をキャッシュの実装から切り出しました。元の配置実験は、独立したパッチを別のソースディレクトリに適用して復元できます。[簡素化版のキャッシュ実装](cpu/iq3-cache.cpp)・[row-local実験の入口](experiments/row-local/README.md)

CPUの比較ビルドは、このサーバーに保存したコンパイラ、ヘッダー、既存オブジェクトを使用します。未変更の対照は2つの翻訳単位とCPUライブラリを元とバイト単位で照合しました。候補ではキャッシュのC++翻訳単位だけを差し替えました。リンク時に参照したbaseと、プロセスが読み込んだbaseの識別子を記録しています。[ビルド記録](../archive/review-handoffs-not-for-readers/2026-11-19-runtime-simplification/build/BUILD.json)・[再利用したオブジェクトの出所](../archive/review-handoffs-not-for-readers/2026-11-19-runtime-simplification/CPU_OBJECT_PROVENANCE.json)・[実測報告](../archive/review-handoffs-not-for-readers/2026-11-19-runtime-simplification/RESULTS_JA.txt)

x9ramでCPUの対照と簡素化版を再構築する場合は、次のコマンドを実行してください。出力先にはまだ存在しないディレクトリを指定します。ビルド完了時に2つのCPUのSHA256が表示されます。

```bash
python3 /home/codex/projects/llm-lab/ecc9-simplification-20261119/build_cpu.py --output /tmp/xeon-iq-compare
```

リポジトリの `accepted/` と2026年11月5日のReleaseは、当日の96 GiB構成を再現するための保存版です。[11月5日の基準](../profiles/e5-2630lv2-512gb/CHECKPOINT.json)
