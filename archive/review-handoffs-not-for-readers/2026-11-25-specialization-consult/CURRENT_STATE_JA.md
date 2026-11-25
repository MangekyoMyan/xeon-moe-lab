# Q1　11月25日に実際に動いていたエンジン

2026年11月25日19時41分から19時42分（JST）に、x9ramの通常APIでモデルPID **268298** を確認した。起動引数、設定した環境変数、実行ファイル、主要4ライブラリ、expert選択表は前回の精度相談時と一致した。プロセスのメモリマップに記録されたデバイス・inodeとディスク上のファイルを対応付け、サーバー本体・推論に使う8本の共有ライブラリのSHA-256を取り直した。重み7分割と画像projectorはサイズと取得時の検証記録を照合した。今回の実機操作は読み取りで、性能の数値は以前の実行に由来する。[今回の照合結果](CURRENT_STATE.json)・[比較元の構成](../../../current/runtime.json)

| 項目 | 稼働構成 |
|---|---|
| サーバー | DMIはSugon X9DR3-F、BIOS 3.4。Xeon E5-2697 v2 ×2、24物理コア／48論理CPU。32 GiB ×16、512 GiB、DDR3-1600 |
| CPUの割当て | 入力24・生成24。CPU 0–11と12–23で各ソケットの物理コアを1回ずつ使用。両方のstrictマスクは `0xffffff`。NUMA 0・1への交互配置。OpenMPスレッド上限24、動的チーム無効、poll 0。ロード時のパックは12スレッド |
| GPU | Tesla PG500-216、Volta sm70、32,768 MiB。今回の1標本で29,942 MiB使用、150 W上限、ECC有効。通常ファンはStandard、上限70%。比較時のホストRAM予備は64 GiB |
| モデル | OrcaRouter DeepSeek-V4.1-Flash-1017-abliterated **Mix-256G**、7分割。モデル取得リビジョン `349470a0`、F16画像projector `62168fc5` |
| 配置 | 生成1トークンでは第0–36層の選択expertをCPU、第37–39層はGPU。第0–36層のshared expertもGPU。入力をまとめると一般／MXFP4のGPUオフロードしきい値64／64が効く。重みの置き先と演算の担当は別に決まる |
| RAM | expert単位のIQ事前展開は19,200個の「射影・expert」の組、80テンソル、実サイズ127,401,984,000 B、予算120 GiB。選外は元の重みを使う。GPUから使うホスト重みを登録。ロードはmmap＋mlock |
| 会話 | コンテキスト131,072、K/VはF16、batch／ubatch 1,024、1スロット、継続バッチ・context shiftは無効。元のchat templateとOpenAI互換ストリーミングを使用。会話モード、画像は最大256トークン |
| キャッシュ | `--cache-prompt`で同じスロットの共通prefixを再利用。明示指定 `--cache-ram 0`、`--ctx-checkpoints 0`。この運用でRAMへの複数会話の状態退避や自動チェックポイントを保証した記録はない |

| 対象 | 実ロードSHA-256の先頭 | 対応するローカル実装・ビルド |
|---|---|---|
| llama-server | `13cfc515` | DeepSeek用フォーク `dde879bc4f954b809892fd0d660515f42536394a`、`deepseek41-bringup-20261101/build-commands.json` |
| CPU | `33a9e346` | `mx-prepack/release-candidate-a`。CPU C本体は `mx-prepack/backend/ggml-cpu.c`、backend C++は `sparse-cache/ggml-cpu.cpp`、実際のcompact実装は `row-local/iq3-cache.cpp` |
| CUDA | `45f6b7d8` | `cuda-sort/release-candidate-a`のCUDA本体とID整列補助。CUDA 12.9系、70-real、CUDA Graphs有効 |
| ggml-base | `2e13b370` | `gpu-stream-20261103/ggml-backend.cpp` と `SCHEDULER.json` |
| libllama | `1d1e7d72` | `sparse-cache/llama-model.cpp` と `LOADER_BUILD.json`。モデルのgraph・状態処理は同フォークのソース |

CPUビルドはIvy Bridge向けAVX・F16C・OpenMP、AVX2／FMA無効。再利用オブジェクトには元のコンパイル記録がある。前回のCPU対照再リンクでは稼働DSOと全バイト一致した。今回の資料では当時のビルド記録と現在のファイルを同定した。フルビルドを繰り返した場合の再現性には踏み込んでいない。実ロードパス・完全なハッシュ・公開ファイルへの対応は機械可読資料にまとめた。[実ロード全9ファイル](CURRENT_STATE.json)・[ソースとレシピの対照表](SOURCE_DIFF.json)・[CPUオブジェクトの由来](../2026-11-19-runtime-simplification/CPU_OBJECT_PROVENANCE.json)

## Q4　専用化で守るモデル計算と利用機能

| 境界 | このモデルで必要な動作 |
|---|---|
| 入出力 | 隠れ幅5,120、40層、語彙129,280。各層384のrouted expertから6個を選び、shared expertが1個。ルーターの `blk.L.ffn_gate_inp.weight` は [5120,384]。F32蓄積のlogitsから `sqrt(softplus(logit))`、選択バイアスを加えてtop6、重みにはバイアス前の値を使い、和＋1e-20で正規化して1.5倍。画像用バイアスの選択条件も保持 |
| CPUの4演算 | `blk.L.ffn_{gate,up}_exps.weight` はGGML軸順 [5120,2304,384]、`ffn_down_exps.weight` は [2304,5120,384]。入力F32、IDはI32。選択slotごとのgate・up → clamped SwiGLU → down。SiLU後のgateに上限10、upを±10に制限。各slotに対応する重みを掛け、元のslot順に合算し、shared出力を加える |
| 実際の型 | CPU37層のgate／upは同層で同型。IQ3_XXS 21層、MXFP4 9層、IQ2_S 6層、IQ3_S 1層。downは異型の層もある。CPU担当層のshared側は実機の37組ともgate Q6_K、up Q8_0、down Q6_K。層番号から型を推測する実装を避け、テンソル記述子を読む |
| 履歴と注意機構 | mHCは4本の残差stream、Sinkhorn 20回。rawの窓128、圧縮KV、indexer top-k 512、KV供給層2/8/14/20と層間共有、位置・圧縮状態と更新順を保持。Engramは層1/14で4-gramを使い、sequence境界・直前のトークン・画像embeddingのマスクが入力に効く |
| サービス | 最優先は通常会話の**生成中の1トークン時間と遅い側の間隔**。併せて短文の最初の本文、長い入力の処理、同一会話のprefix再利用、画像付き要求、128kの設定を守る。1スロットが現行運用。複数sequence・保存／復帰の実装は元の経路に残せる。旧CPUで16kの継続と画像を実行済み。新CPU・128k全域での品質や永続復帰の検証は別に必要 |

将来の試作ではモデル・planの識別子、arch、shape、stride、型、入力数、CPU配置、元のslot対応、バッファ所有期間を検査する。当初の専用経路は1テキストトークン・1sequence・対応するCPU splitだけに適用する。条件外は元の演算を通すので、既存のprefill・画像経路を使ったまま比較できる。[モデル実装](../../../vendor/llama.cpp/src/models/deepseek41.cpp#L877)・[routerとslot集約](../../../vendor/llama.cpp/src/llama-graph.cpp#L2161)・[KV保存／復帰](../../../vendor/llama.cpp/src/llama-kv-cache-dsv4.cpp#L1732)・[実トレースから得た層別の型](SAVED_TRACE_RESULT.json)
