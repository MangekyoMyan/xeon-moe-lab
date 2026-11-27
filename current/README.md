# x9ram の通常推論構成

2026年11月27日に、通常APIの起動引数、設定した環境変数、使用中の実行ファイルと共有ライブラリを再照合しました。実機はSupermicro系X9DR3-F（DMIの製造元はSugon）、E5-2697 v2 ×2、512 GiB、PG500-216です。11月24日に採用した24コア構成を使用しています。[11月27日の実プロセスと9ファイルのSHA-256](verification-2026-11-27.json)・[CPU交換前後の実測](CPU_UPGRADE.md)

**稼働版の変更ソースは `overlays/` にあります。** CPUライブラリのSHA-256は `33a9e346` で始まります。モデルの計算グラフとKV管理にはリポジトリ内の固定したllama.cppを使用し、CPU・CUDA・loader・schedulerの変更はソース20ファイルとして収録しています。各ファイルが入る場所、SHA-256、実際のライブラリとビルド記録を追えます。[ソースの対応表](ORIGINS.json)・[比較実験と生データ](MEASUREMENTS.md)

通常の待受で、同じ短文への生成速度は 9.76 tok/s、最初の本文は1.83秒でした。[24/24の採用記録](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-adoption-20261124)

| 通常設定 | 値 |
|---|---|
| モデル | OrcaRouter DeepSeek-V4.1-Flash-1017-abliterated、Mix-256G、7分割 |
| コンテキスト | 131,072、K/Vともにf16、1スロット |
| 入力処理・生成 | 入力24・生成24物理コア（各ソケット12）、batch/ubatchともに1024 |
| expertの配置 | 1トークン生成時のrouted expertは0–36層がCPU、37–39層がGPU。0–36層のshared expertもGPU。まとまった入力のGPUオフロードしきい値は一般／MXFP4とも64 |
| RAMの重みキャッシュ | 予算120 GiB、80テンソルの19,200件のexpert射影に127,401,984,000バイトを割当て |
| メモリ配置 | NUMAノード0・1の交互配置。プロセスのTHPは無効 |
| GPU | 150 W、ECC有効、sm70。画像用のF16 projectorもロード |

4つのライブラリについて、配備ファイルのパス・SHA256・読み込み順を確認できます。モデルの配布リビジョン、全7ファイルと画像projectorのSHA256、選択計画、サービス設定のSHA256も同じ台帳に収録しています。[実行構成の台帳](runtime.json)・[このモデルのexpert選択計画](models/deepseek41-mix-256g/frequency19200.plan)

## 処理をコードから追う

| 処理 | 実装 |
|---|---|
| モデルが選ぶ6 expert、slot順の集約、shared expert、mHC | [DeepSeekのモデルグラフ](../vendor/llama.cpp/src/models/deepseek41.cpp)・[MoEのグラフ構築](../vendor/llama.cpp/src/llama-graph.cpp) |
| CPUのexpert内積、compactヒット／元形式への切替、MXFP4入力の準備 | [CPUのMAT_ID演算](overlays/ggml/src/ggml-cpu/ggml-cpu.c)・[稼働版のcompact cache](overlays/xeon-local/iq3-cache.cpp)・[MXFP4入力側の処理](overlays/xeon-local/mx-rhs.cpp) |
| GPUのexpert ID整列と演算、CPU/GPU間の入力コピー・同期 | [CUDA backend](overlays/ggml/src/ggml-cuda/ggml-cuda.cu)・[ID整列](overlays/ggml/src/ggml-cuda/sort-ids.inc)・[backend scheduler](overlays/ggml/src/ggml-backend.cpp) |
| ロード時の展開とホスト重み登録、圧縮KVと会話履歴 | [モデルloader](overlays/src/llama-model.cpp)・[expert選択計画の読込み](overlays/xeon-local/selection-plan.h)・[KV管理](../vendor/llama.cpp/src/llama-kv-cache-dsv4.cpp) |

稼働版ではF32のclamped SwiGLUで、生のgate値に上限を掛けてからSiLUを計算し、上下を制限したupと掛けます。制限値は演算パラメーターから読み、このモデルでは10です。[通常版の活性化演算](../vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp#L3546)

## 構成を確認する

リポジトリのルートで実行すると、手元のLinuxからプロファイルの数値を表示できます。起動時の引数・環境変数は、パス・待受アドレス・GPU識別子をプレースホルダーにしたテンプレートでも確認できます。[通常APIの起動設定](server-config.template.json)

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

### ソースの復元

`python3 scripts/prepare_source.py --profile current` は、固定した上流ソースに20ファイルを重ねて `work/current-source/` を作ります。既に作成済みのツリーは `python3 scripts/prepare_source.py --profile current --verify` で対応表に照合できます。Volta向けの構成・ビルドコマンドは `python3 scripts/build_plan.py --profile current --cuda-root /path/to/cuda-12.9` で表示できます。

11月27日に復元したツリーでは、CUDA 12.9.86・GNU 14.2.0でCMakeの構成と、変更対象のC・C++・CUDA計8翻訳単位のコンパイルを確認しました。この検証はオブジェクト生成までです。稼働APIが実際にロードしたライブラリは、各ビルド記録のSHA-256と照合しています。[ビルド記録とソースの対応](ORIGINS.json)

### 旧CPUで試したrow-local配置

通常のCPUソースにはrow-local配置コードがあり、通常設定では無効です。`cpu/iq3-cache.cpp` はその導線を切り出した比較版（CPUライブラリ `2626bb83`）です。旧CPUの2課題でわずかに遅く、比較用として保存しました。[簡素化版のキャッシュ実装](cpu/iq3-cache.cpp)・[row-local実験の入口](experiments/row-local/README.md)

CPUの比較ビルドは、このサーバーに保存したコンパイラ、ヘッダー、既存オブジェクトを使用します。未変更の対照は2つの翻訳単位とCPUライブラリを元とバイト単位で照合しました。候補ではキャッシュのC++翻訳単位だけを差し替えました。[ビルド・オブジェクトの由来・実測](https://github.com/MangekyoMyan/xeon-moe-lab/tree/633e8862c9e80f87d937b4a85ba174b7122558be/archive/review-handoffs-not-for-readers/2026-11-19-runtime-simplification)

x9ramでCPUの対照と簡素化版を再構築する場合は、次のコマンドを実行してください。出力先にはまだ存在しないディレクトリを指定します。ビルド完了時に2つのCPUのSHA256が表示されます。

```bash
python3 /home/codex/projects/llm-lab/ecc9-simplification-20261119/build_cpu.py --output /tmp/xeon-iq-compare
```

リポジトリの `accepted/` と2026年11月5日のReleaseは、当日の96 GiB構成を再現するための保存版です。[11月5日の基準](../profiles/e5-2630lv2-512gb/CHECKPOINT.json)
