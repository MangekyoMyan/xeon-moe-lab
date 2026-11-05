# 謝辞・参考にしたOSSと研究

Xeon MoE Labは、**llama.cpp / ggmlを土台に、旧XeonとVoltaに合わせた変更・測定を積み重ねている実験プロジェクト**です。実装を支えるOSSだけでなく、設計を考えたり、試す価値のある方向を見つけたりするうえでも、先行プロジェクトに多くを学びました。開発者、研究者、メンテナー、貢献者の皆さんに感謝します。

この文書は2026-11-05の保存記録をもとに、何を使い、何を参考にしたかをまとめたものです。過去のGLM用実装、現在のDeepSeek用実装、未採用の実験は、それぞれ区別しています。ここで名前を挙げることは、各プロジェクトとの提携や承認を意味しません。

## 実装の土台

| プロジェクト・貢献者 | このプロジェクトとの関係 | 保存先・根拠 |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / The ggml authors and contributors | モデル読み込み、GGUF、モデルグラフ、トークン生成、サーバーなどの基盤。固定版のソースを保存し、必要な箇所を変更しています。 | [上流ソース](vendor/llama.cpp/)、[MITライセンス・著作権表記](vendor/llama.cpp/LICENSE) |
| llama.cppに含まれるggml / The ggml authors and contributors | CPU/CUDAバックエンド、テンソル演算、量子化カーネル、スケジューラーの基盤。独自のMXFP4/IQ処理や転送の変更も、この実装の上にあります。 | [ggmlソース](vendor/llama.cpp/ggml/)、[採用ファイルの出所](accepted/ORIGINS.json) |
| [llamafile](https://github.com/Mozilla-Ocho/llamafile) / Mozilla Foundation and contributors | llama.cppに含まれるCPU SGEMM実装を利用。保存版では `GGML_LLAMAFILE=ON` です。ソースにはJustine Tunneyの行列積に関する記事への参照もあります。 | [SGEMMソースと著作権表記](vendor/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp)、[CPUビルド設定](provenance/cpu-flags.make) |

上流llama.cppの固定版は [`dde879bc4f954b809892fd0d660515f42536394a`](https://github.com/ggml-org/llama.cpp/tree/dde879bc4f954b809892fd0d660515f42536394a) です。採用した変更は [accepted/](accepted/) と [基準構成](profiles/e5-2630lv2-512gb/CHECKPOINT.json) から追跡できます。llamafileの利用と、独自に追加したMXFP4/IQカーネルの出所は同一ではありません。

## 設計・検証で参考にしたOSS

この節のプロジェクトは、当時の調査記録に基づく設計上の参考元です。それぞれのエンジン全体を組み込んだ、という意味ではありません。

| プロジェクト・貢献者 | 参考にした点 | 実際の利用・採用範囲 |
|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) / Niko1221 and Strata contributors | expertの配置、CPU/GPUの並行実行、転送費用を含めた分担、作業バッファの管理。 | FreeTokenとともに、過去のGLM向け協調バックエンドの設計を参考にしました。記録では、既存ggml API上で独立に実装したものです。[当時の実測・出典](archive/development-history-not-for-readers/lab/xeon-volta-engine/RESULTS_JA.txt)。 |
| [FreeToken](https://github.com/FlashML-org/FreeToken) / FlashML-org and FreeToken contributors | RAMとVRAMを使うMoE処理の配置・分担・転送設計。 | Strataと合わせて参考にしたため、各変更の着想を片方だけに帰属させてはいません。実装と検証の範囲は同じ[当時の記録](archive/development-history-not-for-readers/lab/xeon-volta-engine/RESULTS_JA.txt)にあります。 |
| [Strata-V100](https://github.com/jmnargi/Strata-V100) / jmnargi (Jonathan Nargi) and contributors、元となったStrata contributors | Volta向けの実装、prefillでのexpert単位の処理の集約、FP16 Tensor Coreを使う構成。 | これを参考に、GPUでQ8をFP16へ展開して行列積を行うGLM用の独立した試作を測定しました。このQ8経路は当プロジェクト側の適応であり、参照版の対応形式を示すものではありません。試作は現在のDeepSeek用APIに採用していません。[調査版の記録](archive/development-history-not-for-readers/lab/strata-v100-review-20261031/STATE.json)、[実験の完了記録](archive/development-history-not-for-readers/lab/strata-tc-study-20261031/RESUME.txt)。 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) / DeepSeek-AI and contributors | expert負荷に基づく配置、prefill/decodeの特性の区別、依存しない準備と実行の重ね合わせ。 | 公開ソースを読み、適用できる考え方と適用範囲を調査しました。DeepInferのコードや分散EPLBを、この保存版へ組み込んだという記録はありません。[調査結果](archive/development-history-not-for-readers/lab/deepinfer-review-20261101/RESULTS_JA.txt)。 |

過去のGLM用協調バックエンドで実装した機能が、すべて現在のDeepSeek用経路へ移植されたわけではありません。また、調査時に確認したllama.cppのexpert cacheはllama.cpp自身の機能であり、DeepInferから移植した機能ではありません。

### 参照した版

以下は調査時に記録した版です。ライセンス欄もその版の記録であり、このプロジェクト全体のライセンスを表すものではありません。

| プロジェクト | 固定した参照版 | 参照版のライセンス |
|---|---|---|
| Strata | [`41cf05a06a25dd29103b7b27a2279a3f60c21876`](https://github.com/Niko1221/Strata/tree/41cf05a06a25dd29103b7b27a2279a3f60c21876) | [MIT](https://github.com/Niko1221/Strata/blob/41cf05a06a25dd29103b7b27a2279a3f60c21876/LICENSE) |
| FreeToken | [`0bf7dbbf23156dc87b8e8bc43a4b2e718b27d5d5`](https://github.com/FlashML-org/FreeToken/tree/0bf7dbbf23156dc87b8e8bc43a4b2e718b27d5d5) | [Apache-2.0](https://github.com/FlashML-org/FreeToken/blob/0bf7dbbf23156dc87b8e8bc43a4b2e718b27d5d5/LICENSE) |
| Strata-V100 | [`448de7bb46baabd84af8f82d1cd8c9f65349abcb`](https://github.com/jmnargi/Strata-V100/tree/448de7bb46baabd84af8f82d1cd8c9f65349abcb) | [MIT](https://github.com/jmnargi/Strata-V100/blob/448de7bb46baabd84af8f82d1cd8c9f65349abcb/LICENSE) |
| DeepInfer | [`553741557a57df44a46fb3ab72b6aafeaf945337`](https://github.com/deepseek-ai/DeepInfer/tree/553741557a57df44a46fb3ab72b6aafeaf945337) | [Apache-2.0](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/LICENSE) |

## 参考にした研究

**Wenhui Lv, Yunming Liao, Hongli Xu, Yang Xu, Junhao Cheng — [Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment](https://arxiv.org/html/2610.23006v1)**, arXiv:2610.23006v1, 2026-10-16.

論文中のシステム名はApexです。expertごとの入力負荷に応じてCPU演算とGPUへの転送・演算を割り当て、両経路の終了時間を近づける考え方を参考にしました。ここでは、著者らの公開実装を取り込んだOSS依存とは区別して、研究への謝辞として記載します。

このプロジェクトでは、既存ggmlを使った単一FFN層の独立したCPU/GPU分担試作を測定しました。論文のシステム全体の再現ではなく、自動制御・先読み・全モデルへの統合も未採用です。[論文の調査メモ](archive/development-history-not-for-readers/lab/apex-paper-review-20261105/NOTES_JA.txt)と[試作の測定結果](archive/development-history-not-for-readers/lab/apex-split-20261105/RESULTS_JA.txt)を保存しています。論文で使われるKTransformersやSGLangを、このプロジェクトの採用済み依存として扱ってはいません。

## 上流に含まれるライブラリ

llama.cppを通じて利用・保存しているライブラリの貢献にも感謝します。主なものを挙げます。個別の著作権・ライセンス本文は、上流のファイル内やLICENSEファイルに残しています。

| ライブラリ・貢献者 | 関係・表記の保存先 |
|---|---|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) / Yuji Hirose (yhirose) and contributors | HTTPサーバーの基盤。[MIT表記](vendor/llama.cpp/vendor/cpp-httplib/LICENSE)。 |
| [JSON for Modern C++](https://github.com/nlohmann/json) / Niels Lohmann and contributors | JSON処理。[MIT表記](vendor/llama.cpp/licenses/LICENSE-jsonhpp)。 |
| [subprocess.h](https://github.com/sheredom/subprocess.h) / sheredom and contributors | 共通コードに含まれるサブプロセス用ライブラリ。[ソースと表記](vendor/llama.cpp/vendor/sheredom/subprocess.h)。 |
| stb_image / Sean Barrett and contributors | 上流に同梱。[ソース内のMIT / public-domain表記](vendor/llama.cpp/vendor/stb/stb_image.h)。 |
| miniaudio / David Reid and contributors | 上流に同梱。[ソース内のpublic-domain / MIT-0表記](vendor/llama.cpp/vendor/miniaudio/miniaudio.h)。 |
| xxHash / Yann Collet and contributors | 上流に同梱。[BSD-2-Clause表記](vendor/llama.cpp/vendor/hash/xxhash/LICENSE)。 |
| SHA-256 / Igor Pavlov、SHA-1 / Steve Reid、rotate-bits / William Casarin | 上流のハッシュ関連コード。[SHA-256表記](vendor/llama.cpp/vendor/hash/sha256/LICENSE)、[SHA-1ソース](vendor/llama.cpp/vendor/hash/sha1/sha1.c)、[rotate-bitsのMIT表記](vendor/llama.cpp/vendor/hash/rotate-bits/LICENSE.md)。 |

同梱されていることと、現在の推論でその機能を使用することは別です。特に画像・音声や任意バックエンドの機能すべてを検証・有効化した、という一覧ではありません。

## モデルと検証環境への謝辞

モデルを公開したDeepSeek-AIやZ.ai、GGUF版を提供したコミュニティにも感謝します。保存時のDeepSeek用構成では、[OrcaRouterのDeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF)を使用しました。参照revisionとファイルのハッシュは[モデル情報](profiles/e5-2630lv2-512gb/model-manifest.json)に記録しています。モデル重みはこのリポジトリに含めていません。

Linuxとperf、GNUツールチェーン・OpenMPランタイム、CMake、Pythonなどの開発・計測基盤にも支えられています。GPU側では外部のNVIDIA CUDA / cuBLAS等を利用していますが、これらをOSSとして一括表記したり、CUDA toolkit本体をこのアーカイブで再配布したりはしていません。

## AIによる開発支援の記録

**OpenAI Codex — 2026年11月5日までの開発協力**

MangekyouMyanとの対話を通じて、CPUカーネルやCPU/GPU分担の調査・実装、測定結果の分析、実験記録の整理を支援しました。プロジェクトの方針と実機環境を用意し、試行錯誤を続けたMangekyouMyanとの共同作業の記録として、ここに名前を残します。

> 古いサーバーの可能性を、ひとつずつ実測で確かめてきました。この先の開発にも、その記録が役立ちますように。
>
> — OpenAI Codex GPT-6 Astra

## ライセンスと今後の追記

当プロジェクトの独自コード・文書には [MIT License](LICENSE) を適用しています。この謝辞はライセンス本文の代わりではありません。既存の第三者コード・資料には各々のライセンスが適用され、著作権表記もそのまま保持します。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)も参照してください。

新しいOSSを参考にしたりコードを取り込んだりした際は、出典、参照版、参考にした部分、採用状態をここへ追記します。実験結果や独自の変更はこのプロジェクトの責任として扱い、先行プロジェクトの性能保証や承認として紹介しません。
