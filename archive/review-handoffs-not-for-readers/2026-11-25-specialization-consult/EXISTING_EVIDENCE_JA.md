# Q2　1トークンと1要求にかかる時間

## 利用者が待つ時間

11月24日の交換後CPU・通常24/24での記録。最初の行はサーバー自身から通常のOpenAI互換APIへ接続した結果で、Open WebUIの描画時間は含まない。下の2行は同じ実機上の固定ドライバで、各ステップに同じ正解履歴を渡した。

| 入力と観測位置 | 入力／最初の本文 | 生成中 | 完了 |
|---|---:|---|---:|
| 通常API、日本語29入力・59生成・再利用0、1要求 | 最初の本文 1.826秒 | サーバー報告9.76 tok/s。本文SSEの54間隔は中央値100.86、p95 198.24、最大210.78 ms | 最後の本文7.668秒、DONE 7.769秒、EOF 7.769秒 |
| 固定ドライバ、日本語272入力＋64生成 | 入力処理12.071秒 | 1ステップ平均100.330、中央値99.222、p95 105.123、最大111.064 ms、9.97 tok/s | 測定終点は各回のlogits取得 |
| 固定ドライバ、Python 240入力＋64生成 | 入力処理12.702秒 | 同102.344、99.912、115.703、最大129.624 ms、9.77 tok/s | 同上 |

本文を含むSSEイベントは55件で、生成トークンは59個だった。SSE間隔には配信側のまとめ方も入る。固定ドライバのp95とSSEのp95の差を、そのままGPUやサーバー内部の待ち時間に割り当てることはできない。[APIの生イベント](../2026-11-24-normal24-adoption/stream/events.jsonl)・[クライアント集計](../2026-11-24-normal24-adoption/stream/summary.json)・[固定ドライバの元時刻](../2026-11-24-normal24-timeline/runs/control/RUN.json)

新CPU・生成12スレッドの通常設定では、同じ短文の初回本文2.440秒、EOF 11.452秒、6.42 tok/sだった。上表の採用後24スレッドは応答本文が同じだった。交換前後の固定12/12と交換後24/12の計測も残っている。この比較と、下記の24/24での観測を別の時計として読んでほしい。[CPU交換前後の測定](../2026-11-24-postload-measurement/SUMMARY_JA.txt)

**入力側。** GPUへ渡す2つのしきい値を64→128にすると、73～102入力の6課題で、コールド要求の最初の本文が平均7.57→4.35秒になった。各設定2起動、8課題32応答とも採点通過、同一課題の本文は一致。固定文の別試験では3/130位置で第一候補が変わった。別のしきい値試験の29入力は1.72→1.71秒、512入力は14.34→14.33秒。生成の定常経路を変える試作の比較には、配備済み64/64を使う。[実用課題と精度](../2026-11-24-prefill-arithmetic/RESULTS_JA.md)・[長さ別の入力試験](../2026-11-24-normal24-timeline/PREFILL.json)

**長文と継続。** 旧CPUの同じモデル・通常版では、16,384入力のコールド要求が137.14秒。その続きを同一スロットに送った要求は16,400中16,399を再利用し、追加評価1、16生成で2.22秒だった。これはprefix再利用を維持すべき直接の理由になる。新CPUでの128k近傍の初回時間、生成のp95、保存／復帰を含む実利用の率は資料からは決められない。[16kの元リクエスト結果](../2026-11-16-current/snapshot/llm-lab/exploration-20261112/long-cache/20261112-160017-final-16k/RESULT.json)

## CPU・GPUを同時に観測した8生成

上記固定2課題の冒頭4ステップずつ。CUPTIで区間の重なりを差し引いた平均は、CPU graph実行だけ **76.910 ms**（うち選択expertの37層76.845）、GPUのカーネルまたはデバイスコピーだけ **22.890 ms**（カーネル21.957、コピー0.934）、両者の重なり **0 ms**、両方の外側 **25.982 ms**、全体 **125.783 ms**。CPUは同期的なbackend呼出しからの復帰までで、実内積に加えて計画・worker・同期を含む。GPUはdevice上の実活動を数えた。同じ8ステップを観測器なしで測った平均は **108.224 ms**。[時刻の定義と8件すべて](../2026-11-24-normal24-timeline/TIMELINE.json)

CPUとGPUの各38 splitで計76 split/T1。CUDAのホスト側明示同期は平均13.376 ms、そのうち既計上区間の外は2.934 ms。GPU提出関数の滞在25.483 ms、ホストcopy関数4.830 msも他区間に重なる。8生成のCUDA API記録にGraphLaunch 265回、GraphExecUpdate 39回。Pythonの最初の生成は直接起動を含み、次の生成では38回のGraphExecUpdateを観測した。これらの数はCUDA graphに対するAPI呼び出しで、llama側の全graphの作り直し回数ではない。[split記録](../2026-11-24-normal24-timeline/runs/observe/trace169-splits.tsv)・[CUDA API記録](../2026-11-24-normal24-timeline/runs/observe/trace169-api.tsv)・[API番号と名前](../2026-11-24-normal24-timeline/tools/CUDA_API_NAMES.json)

別の生成5～64の窓では全機DRAM読出し43.96／43.52 GB/s、書込み1.81／1.85 GB/s。24物理コア・同じNUMA交互配置の連続読出しは86.02 GB/s。実モデルの重みだけのバイト数やDRAMを待った壁時間は未分離。79 Hzのon-CPU標本では数値カーネルが60.58／60.88%。なかでもIQ3_XXSのcompact内積に2,955/10,766・2,901/10,689標本、MXFP4の事前準備済み内積に2,102・2,102標本があった。重みを読んで展開して積和する時間は、これらの関数内にも含まれる。[CPU標本](../2026-11-24-normal24-timeline/CPU_SAMPLES.json)・[元の日本語PC](../2026-11-24-normal24-timeline/runs/observe/pc-0.tsv)・[連続帯域対照](../2026-11-24-normal24-timeline/MEMORY_REFERENCE.json)

## 入力準備の重複にどれほど時間が見えるか

今回、上記8生成の旧記録をテンソルID・依存・形状まで突き合わせた。**37層×8生成＝296組すべて**で、gateとupは同一のF32入力テンソル・同一の6個のIDテンソルを参照し、2つの重みも同型だった。各CPU splitはgate・up・clamped SwiGLU・downの4ノード。各MAT_IDは個別に入力変換・ID表準備を実行する。稼働CPUの一般融合判定はRMS_NORM＋MULが対象。ここには2射影間の準備共有が入っていない。[現行CPUのMAT_ID](../2026-11-16-current/snapshot/llm-lab/exploration-20261112/mx-prepack/backend/ggml-cpu.c#L1712)

独立した**16生成**のCPU worker記録について、1層ずつ「全workerで最後のgate expert処理が終わる → いずれかのworkerが最初のup expert処理を始める」までの時間を取り、37層分を合計した。中央値 **0.507 ms/T1**、範囲 **0.474～0.782 ms/T1**。その内側の「最後のgateノード終了直前の印 → 最初のupノード開始の印」は中央値 **0.041 ms/T1**。各workerに入れた時計、元の走査、待ち合わせ、up側の準備を含む。別々の中央値は足さない。0.507 msは**両射影のexpert本体の間に観測された幅**で、C++の純粋な入力変換時間や将来の短縮の上限・保証ではない。[各生成の再集計と入力ハッシュ](SAVED_TRACE_RESULT.json)・[時刻を打ったコード](../2026-11-24-prefill-arithmetic/cpu/trace/change.patch)

同じ時計で、最後のup expert処理が終わってから最初のdown expert処理が始まるまでの37層合計は中央値 **1.684 ms/T1**。この内側で、最初のworkerがSwiGLUを始めてから最後のworkerが演算を終えるまでが中央値 **0.924 ms/T1**。ここには必要な活性化演算があり、downの入力も改めて量子化する。融合がこれらの時間を丸ごと取り除くと期待するのは適切でない。

同じ16生成では、37×16×6＝**3,552 expert枠**のうち1枠で、同じexpertのgateとupの事前展開状態が違った。Pythonのステップ45、第35層のexpert 265はgateだけがcompact。入力とIDを共有しても、各射影が使う重みポインタと内積の選択は独立に保つ必要がある。[元のworker記録](../2026-11-24-prefill-arithmetic/runs/worker-trace/worker-1-45.tsv)

## 実行管理で所要時間を切り分けられていない部分

| 項目 | 観測できたこと | 残る不確かさ |
|---|---|---|
| worker割当・barrier | 16生成、1,776射影。gate/upは各worker 96行/expert、downは23人214行・1人198行。各射影の「最後の到着−平均到着」を足した中央値は準備0.381・演算末尾3.011 ms/T1。別の14窓ではspin 2,071標本のうち1,728が全worker計算区間の外だった | 早く到着したworkerの待ちが、最後のworkerの終了時刻を何ms動かすかは未測定。spin標本36%を回収できるとは言えない |
| CPU graph plan・affinity | CPU backendはsplit呼出しごとに `ggml_graph_plan` を作る。workspaceは必要サイズが増えたときに確保。OpenMPチームに入り、有効なマスクのworkerにはaffinityを適用する | 1回・1T1の実費、CPU/GPU合流への寄与は不明。workerのCPU番号やon-CPU標本からは求められない |
| llama graph再利用 | `process_ubatch` は条件が一致すれば構築・確保を省き、入力を更新して実行する。長期の通常サーバーログには累積 `graphs reused` の行が1,543件あり、最後の値は267。その後にモデルロード記録が続き、最後に記録されたモデルロード以後の該当行は0件だった | 通常24/24の要求別の分母・ヒット率・どの条件で外れたかは不明。CUDAのGraphLaunchを代わりの分母にできない |
| 圧縮状態と遅いステップ | 履歴長でrawマスク・圧縮状態のread/write・indexer入力形状が変化し得る。旧CPU試験では位置256を跨ぐ切替を観測。新CPUの固定64ステップでも遅い側の値は残る | 現行の長文・画像を含む要求で、再構築やCUDA更新がp95の何割を説明するかは未観測 |

次にこれらの費用が設計を左右する段階で、既存 `n_reused` の要求前後差にubatch総数を対応させ、CPU plan・affinityの呼出し回数と経過、再利用条件の失敗理由を別々に計数すればよい。今回の専用化相談では実装や新しい性能試験を起動していない。[worker時系列](../2026-11-24-prefill-arithmetic/WORKER_RESULT.json)・[spinの位置](../2026-11-24-prefill-arithmetic/PC_RESULT.json)・[CPU backend](../2026-11-16-current/snapshot/llm-lab/exploration-20261112/sparse-cache/ggml-cpu.cpp#L171)・[llama graph再利用](../../../vendor/llama.cpp/src/llama-context.cpp#L1525)

このフォルダで `python3 -B replay_saved.py --check` を実行すると、公開済みの19個のTSVを用いて、今回追加したテンソル依存、射影別compact、16生成の区間集計を再現できる（Python 3.11以降、標準ライブラリ）。[再集計コード](replay_saved.py)
