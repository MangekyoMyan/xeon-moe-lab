# Xeon MoE Lab — 2026-11-05 baseline

旧XeonとVoltaで大きなMoEモデルを動かした際の、採用済みソース・設定・測定結果を公開する実験用リポジトリです。

**2026-11-05時点の構成と結果を保存しています。** ハードウェア固有の調整を含むため、CPUなどの構成を変更する際は、この版を比較の基準として再計測・再調整する前提です。

## 基準構成

| 項目 | 保存時の構成 |
|---|---|
| CPU | Intel Xeon E5-2630L v2 ×2、12物理コア / 24論理CPU |
| マザーボード | 実機DMI: X9DR3-F、vendor: Sugon |
| RAM | 512 GiB、ECC LRDIMM |
| GPU | PG500-216、32 GBクラス、sm70 |
| 運用 | GPU150 W、ECC有効、ファン上限70% |
| モデル | OrcaRouter DeepSeek-V4.1-Flash-1017-abliterated Mix-256G |
| context | 131072、KV f16 |
| upstream | llama.cpp `dde879bc4f954b809892fd0d660515f42536394a` |
| build | GNU14.2、Ivy Bridge AVX/F16C、AVX2/FMAなし、CUDA12.9 / sm70 |

詳細は [基準点](profiles/e5-2630lv2-512gb/CHECKPOINT.json) と [現在設定のテンプレート](profiles/e5-2630lv2-512gb/server-config.template.json) にあります。テンプレートには実APIキーを含めていません。

## 採用済みと実験の区別

採用済みの機能は、MXFP4の4/2/1入力カーネル、単一トークン向けIQ compact cache最大96GiB、expert重みのhost登録、IQ/MXFP4のoffload閾値です。GPU常駐expert層は37–39、generation12 / prefill24スレッド、batch/ubatch1024です。

次は **未採用の実験** です。

- Apex型のexpert分担: CPU/GPUへ同時分担する、独立した1層の試作。全モデルのtok/s・品質や自動コントローラーは未検証。
- expertコスト予測器: オフライン予測器と任意の観測器。実行先を変更しない。
- GPUカーネル候補: 偏った分布で悪化する条件・中間精度の差があり、一律採用していない。

保存したチェックポイントでは、`gpu3-stream1024` とIQ compact cacheの `multi96` は、モデル全体での検証状態が `pending` として記録されています。画像入力も `full_image_inference_verified=false` です。採用バイナリ・設定の記録と、全モデルでの性能・品質検証の完了は区別しています。過去の `pending` / `tested` は当時の値を保持しています。

## 保存内容

| 場所 | 内容 |
|---|---|
| `vendor/llama.cpp/` | 固定upstreamのソース。モデル・語彙GGUF等のデータfixtureを除外 |
| `accepted/overlays/` | 採用バイナリの構築に使った改造ソースと補助ヘッダー |
| `accepted/ORIGINS.json` | 改造ファイルの出所・適用先・SHA256 |
| `profiles/` | ハードウェア、設定テンプレート、基準点 |
| `archive/development-history-not-for-readers/` | 開発内部記録。AIへの引継ぎ・所有者向け報告と、それらに対応する過去の実験コード・数値ログ |
| `provenance/` | コピー元とSHA256、コンパイラ設定、保存対象の検証 |
| GitHub Release | 保存時点で実際に使っていたプロジェクトの実行ファイル・共有ライブラリ |

モデル重み、APIキー、GitHub/Hugging Face認証ファイル、実会話・サーバーログ、CUDA toolkit本体は保存対象外です。合成ルーティングの小さな `.bin` はモデルではなく、試験入力として保存しています。全過去ディレクトリの無差別バックアップではなく、選択したソース・証跡の保存です。

## 過去の開発記録

詳細な試行錯誤や測定根拠を調べる場合は、[開発内部記録（一般閲覧者向けではありません）](archive/development-history-not-for-readers/)を参照してください。AIへの作業指示や所有者向けの報告も含むため、通常の利用説明から分けています。

## 復元と再構築

保存時点のバイナリを保持するため、`baseline-2026-11-05-e5-2630lv2` Releaseへruntime archiveを添付しています。これは現在使っていたファイルのコピーで、新しくビルドした候補ではありません。CUDA12.9のユーザー空間ライブラリ、互換ドライバー、libc/OpenMP等は別途必要です。復元先での長時間推論や新CPUでの性能は、改めて確認してください。

```bash
# 展開後、runtime内のファイル照合だけを行う。APIは起動しない。
python3 scripts/verify_runtime.py /path/to/extracted/runtime-baseline
```

ソースを再構成する場合:

```bash
python3 scripts/verify_archive.py
python3 scripts/prepare_source.py
python3 scripts/build_plan.py --cuda-root /path/to/cuda-12.9
```

`prepare_source.py` は `work/accepted-source` に元ソースと採用済み改造を重ねます。`build_plan.py` の既定動作はコマンド表示のみです。実際にビルドするときだけ `--execute` を指定します。ビルドの出力先は `work/build` です。

**この集約再構築手順は、ソースの照合と計画表示まで確認済みで、新しいフルビルドは未実施です。** 既存のバイナリと同じバイト列になる保証もありません。元のビルドコマンド・フラグは開発内部記録と `provenance/` に保存しています。

開発内部記録の過去スクリプトには元サーバーの絶対パス、root権限、モデル配置、保存済みbuild treeへの依存があります。再構築の入口は上記の `scripts/` で、過去の実験スクリプトをまとめて実行する手順は提供していません。

## CPU交換後

このタグの測定値を基準に、新しい日付のprofile/studyでCPU・NUMA・メモリ速度を比較します。再調整する主な項目は、スレッド数・affinity・expert分担の閾値や係数です。旧構成での評価結果と新構成での結果は区別して記録します。

## ライセンス

上流ソースのライセンス・著作権表記は保持しています。[第三者コード](THIRD_PARTY_NOTICES.md)と[謝辞・参考OSS](ACKNOWLEDGMENTS.md)を参照してください。独自部分のライセンスは未決定です。この公開によって、含まれる第三者コードや資料のライセンスを変更するものではありません。
