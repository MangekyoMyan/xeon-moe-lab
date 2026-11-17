CPU/GPU並列pilotへ進むためのレビュー一式 (2026-11-17)
対象: xeon-moe-lab d98a17c56fadc92cf79e21ecbe80732dedeed899

最初に読むファイル
  ASTRA_GPU_WORKSPACE_NEXT_ACTION_JA.txt — 実機Astra向け、次の作業と返却項目。
  GPU_WORKSPACE_REVIEW_JA.txt — 評価、根拠、今回の発見、優先順。

proposals/ は未統合のコード案です。
  empty_selection_bypass.patch: 固定sourceへgit apply --check済、compile/runは未実施。
  bounded_moe_capture_proposal.h + bind_parent_cells_proposal.h:
    最大8cellsのcapture-only案、header syntax-only済。engine link/runは未実施。
  header_syntax_check.cpp: syntax-only用TU。model/engine testではありません。

plans/
  CAPTURE_CELLS.json: 元35planに対する実route8cells、IDs/slot/positionまで固定。
  EXPERIMENT_CONTRACT.json: 比較モード、timing境界、数値判定、前提と未測定欄。
  BANK_TRADEOFF_RECOMPUTED.json: 原35案再計算、条件付き8層案、訓練内の全集計。
  CONDITIONAL_CONCENTRATED_CANDIDATE.plan: H0費用が大きい時だけ検討する対案1つ。

reviews/evidence/ — raw/source/plan/hash/targetsの独立監査。条件付きDPの別実装検算。
reviews/hotpath/ — 現sourceへの結合点、O/C/S/P、H0、route37内部origin問題。
reviews/activation/ — capture地点/callback契約/generationのsource行と解釈。
tools/ — 今回自作したoffline再集計/提案生成用script。供給repo scriptは実行していません。

実機操作担当はAstra。人間にcommand実行を要求する資料ではありません。
本レビューではモデル/GPU/API/供給ELFを動かしていません。新しい速度値はありません。
repo本体は未変更。FILE_MANIFEST.jsonは同梱payloadのsize/SHAを記録します。
ソースの一次資料は各review内の固定commit URLを参照してください。
