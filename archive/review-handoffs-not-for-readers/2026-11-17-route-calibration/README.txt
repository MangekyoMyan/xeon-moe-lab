route37 calibration / caller overhead study — 2026-11-17
======================================================
RESULTS_JA.txt: 実機結果と限界。CONSULT_PRO_JA.txt: 次の容量/並列化の相談。
DESIGN.json: 実行前に固定したF-before/D/F-after、4call数値gate、64call block、6case。
BUILD.json: 使用binary/source/library pin。checkpoints/initial-callerは未実行の初回build履歴。
runs/: 全成功/失敗原本を保存。今回の3runはすべて正常終了。F32ログは各4×129280要素。
FINAL_ANALYSIS.json: 全block・全64call時刻からの統計。最初のT1も含む。
TIMING_AUDIT_TERRA.txt: 独立raw監査。sourceのレビューはCALLER_REVIEW_TERRA.txt。
RESTORED_RUNTIME.json: 通常API復帰のreceipt。元のengine/source/configは変更していない。

公開archiveではhealth.jsonをguard-health.jsonへ名前だけ変更し、server-config.jsonは除外する。
command/environmentは秘密を含まないLAUNCH_MANIFESTに保存済み。RAW_MAPPING/FILES_SHA256で照合。
元model/ELF/秘密鍵/API keyは含めない。運用・LLMレビュー資料であり通常の閲覧者向け文書ではない。

共有物だけでの再集計（実モデル・GPU・ネットワーク不要）:
  python3 replay_shared.py --output /tmp/route37-review-replay.json
出力は新規pathを指定。このscriptは一時copyでguard-health名とruntime-config内容を復元し、
原本hashに一致することを確認して解析する。commandはデータとして読むだけで実行しない。

上記はレビュー担当向けの再現方法であり、今ユーザーに実行をお願いしている操作ではない。
通常APIは元構成で復帰。長いモデル起動等を依頼する場合は別途「これを実行してください」と明示する。

今回のcaller整備は受領Proレビューのlinear export/output cap/F-D設計に基づき親assistantが実装。
数値・時刻・台帳は親とTerraが独立確認。元llama.cpp/派生CPU/CUDA実装とOSSの帰属はrepo既存資料参照。
