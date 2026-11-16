最初に読むファイル — Xeon MoE Lab 現行レビュー
2026-11-16（JST）

実装担当のAstraへ、このZIPをそのまま渡せます。
先に ASTRA_CURRENT_REVIEW_2026-11-16_JA.txt を読み、必要な担当報告とraw結果へ進んでください。
本書の対象は commit 7d8b077afcf1258602366245f1567e820255defa の共有snapshotです。
公開mainの今後の更新や、現在の実機を自動追跡する資料ではありません。

1. 内容

ASTRA_CURRENT_REVIEW_2026-11-16_JA.txt
  統合評価、今回の新しい知見、既存案との区別、実装順と採否基準。
CURRENT_PERFORMANCE_2026-11-16.png
  保存された8入力の速度を再集計した図。final/hot14で対照群が異なる。
evidence/
  final/hot14/品質/holdout使用履歴の再集計、期待case集合gateの人工再現。
cpu/
  採用MXFP4の独立bit一致、cache容量と選択の厳密計算、混在形式案。
  CPU_ABS_RHS_ADDENDUM_JA.txtは144Bを保つ入力abs案。性能は判定保留。
  prior_reference/には前回160/176B候補のC++原文だけを参照用として収録。
gpu/
  採用sort、hot backendの所有権/同期/slot、H coverage、配置上界、算術追証。
root/
  current/final/runtime照合、Open WebUI v0.11.1の原文helper合成再現、図の作成script。
intake/
  固定Git/Releaseと9稼働backing filesのhash照合、ELFの静的metadata。
streaming/
  直接SSE到着の記録器、使用説明、外部接続なしのtest。
CONTENTS_SHA256.json
  このレビューZIPの内容一覧。自己参照を避けるためmanifest自身を除く。
PARENT_FINAL_VALIDATION.json
  配布対象のJSON/Python構文、CPU担当manifest、既存結果、repo未変更の最終確認。

配布ELF、model、元Release ZIP、公式Open WebUIの全文sourceは含めません。
sourceの引用・照合先は対象commitに固定しています。source path:lineとSHAは詳細報告にあります。
agent由来RESULT内の /workspace/... は今回の実行場所の記録です。再現時は下記引数で手元のpathを指定します。

2. 対象sourceの入口

GitHub:
https://github.com/MangekyoMyan/xeon-moe-lab/tree/7d8b077afcf1258602366245f1567e820255defa/archive/review-handoffs-not-for-readers/2026-11-16-current

Release ZIP v2:
https://github.com/MangekyoMyan/xeon-moe-lab/releases/download/review-2026-11-16-current/xeon-moe-current-review-20261116-v2.zip

Release SHA256:
e4176a9dbe5dc6f252d23b2455edf8576c86d6e0376e830532c4988d5bd0253b

Hは checkout/archive/review-handoffs-not-for-readers/2026-11-16-current。
以下の /ABS/HANDOFF はH、/ABS/PINNED_REPOは対象commitのcheckoutへ置換してください。
/ABS/NEW... はまだ存在しない出力file/dirを指定します。親directoryは用意しておきます。
コマンドはレビューZIPを展開したrootから実行する例です。Pythonは-Oを付けずに使います。

3. 保存資料だけの再集計（実model/API不使用）

  python3 evidence/reanalyze_current_evidence.py \
    --handoff /ABS/HANDOFF \
    --output /ABS/NEW_REANALYSIS.json

既定では既知SHAのanalyze_heldout.pyを一時copyして人工fixtureを与える試験も行います。
純粋なJSON再集計だけなら --skip-gate-probe を追加します。model/controllerは実行しません。

  python3 cpu/analyze_cache_budget.py \
    --snapshot /ABS/HANDOFF \
    --output /ABS/NEW_CACHE_ANALYSIS.json

  python3 gpu/recompute_gpu_review.py \
    --snapshot /ABS/HANDOFF \
    --output /ABS/NEW_GPU_ANALYSIS.json

GPU scriptの --host-sort を指定した場合だけ、元のhost sort testをcompile/runします。
再集計結果にはhost試験未実行という差が出ます。GPU/modelはどちらの場合も使用しません。

  python3 intake/verify_snapshot_intake.py \
    --repo /ABS/PINNED_REPO \
    --zip /ABS/xeon-moe-current-review-20261116-v2.zip \
    --output /ABS/NEW_INTAKE.json

--extract-elf-dir は静的検査用ELFを新規dirへコピーする任意option。実行はしません。

4. CPUの限定合成試験（実model/API不使用）

C/C++ compilerとx86 AVX/SSSE3/F16C対応環境が必要です。
source checkoutを書き換えず、指定した新規dirにobjectと試験binaryを作ります。

  python3 cpu/run_mx_contract_probe.py \
    --snapshot /ABS/HANDOFF \
    --output /ABS/NEW_MX_DIR

  python3 cpu/run_iq_abs_rhs_probe.py \
    --snapshot /ABS/HANDOFF \
    --output /ABS/NEW_ABS_DIR \
    --phase test

absの局所benchmarkを必要に応じて実行する場合は、直前に作った同じdirを指定します。
これは実機の通常API性能を測るコマンドではありません。

  python3 cpu/run_iq_abs_rhs_probe.py \
    --snapshot /ABS/HANDOFF \
    --output /ABS/NEW_ABS_DIR \
    --phase bench --focused

配布rawには初回wallの大きな揺れと、thread CPU time追試を残しています。
benchmark_passedはprocess終了を示すだけで、speedup採用判定ではありません。
初回と追試のbit一致件数を重複加算していません。

5. Open WebUIのbatch helper再現（外部接続・model不使用）

公式commit:
https://github.com/open-webui/open-webui/blob/d3e8bf3405e848cfba377814d0aa7ba7290e414d/backend/open_webui/utils/middleware.py

その原文を用意し、次を実行します。

  python3 root/probe_openwebui_batching.py \
    --source /ABS/middleware.py \
    --output /ABS/NEW_BATCHING.json

Git blob SHA cc5f57e0c320c2b75e106af409e67bea791ee898を要求します。
4つのhelperだけを原文ASTから取り出し、fake emitter/save/時刻で評価します。
実Open WebUI、DB、browserは起動しません。fragmentはmodel tokenではありません。

6. SSE到着の記録器

streaming/README_JA.txtを先に読んでください。
このscriptだけは、明示して実行すると指定URLへstreaming生成要求を1回送ります。
今回の作業では実model/APIに対して実行していません。

  python3 streaming/record_sse_arrivals.py \
    --url 'http://実際のhost:port/completion' \
    --request-json /ABS/REQUEST.json \
    --output /ABS/NEW_ARRIVAL.json

既存requestのstreamをtrueにして1回送信します。自動再試行/redirectなし。
raw本文やAPI keyは保存しません。短いfragmentのhashは匿名化ではなく同一性照合用です。
1 native generationが対象。複数index/id_slot混在は非対応として記録します。
HTTP body read、SSE完成、native累計、OpenAI usageを区別します。
WebUI内部emit/browser描画の計測は別途必要です。

外部接続なしのtests:

  python3 -B -m unittest discover -s streaming -p 'test_sse_recorder.py' -v

fake socket上の実HTTPResponseを含みます。実API/loopback接続はありません。
保存試験13 tests＋23 subtests、親による最終13 tests再実行もpass。

7. 図の再作成

任意。MatplotlibとNumPyが必要です。

  python3 root/plot_saved_results.py \
    --results evidence/REANALYSIS.json \
    --output /ABS/NEW_PERFORMANCE.png

8. 実施済みと未実施

実施済み: 固定source/配布bytesの同定、保存結果再集計、限定host合成試験、原文batch合成再現。
未実施: 実機再推論、full GPU試験、実WebUI描画trace、実model回答品質採点、通常API変更。
添付TXTの実装案は採用済みpatchではありません。通常構成の150W/ECC/fan/reserve方針を保つ前提です。

追加の詳細が必要なら、最初の統合文書の該当章から担当TXTとsource SHAへ進めます。
READMEと統合文書だけで方針を読み、JSONと再現scriptで数字・条件を追える構成です。
