xeon-moe-lab route-capture review / 2026-11-17
base commit: caf35bcdd27c0240572e05a60633f07fca272844

実機側アストラへは、このZIP全体とASTRA_ROUTE_CAPTURE_NEXT_ACTION_JA.txtを渡してください。
これは保存証拠のレビューと次の実装・測定案です。新しい実機benchmark結果ではありません。
通常APIやモデルの構成は変更していません。

読む順番
========
1. ASTRA_ROUTE_CAPTURE_NEXT_ACTION_JA.txt
   次の作業の具体的な範囲、担当、F/D比較、言語screen、容量・bank評価。
2. ROUTE_CAPTURE_REVIEW_JA.txt
   結論、計算根拠、source参照、反復fixtureの限界、並列化への接続。
3. proposals/DRIVER_PROPOSAL_NOTES_JA.txt
   最小export patchの範囲・tradeoff・実施済み検証。

内容
====
proposals/      元driverと照合可能なpatchと提案全文、構文検査receipt。
plans/          token化前の6本文例と、12family/20caseの内容・分割仕様。
reviews/        証拠、hot path、言語/用途の独立詳細レビューと解析JSON/source hashes。
verification/  親の別scriptによるraw再集計、確認結果。
tools/         保存済みJSON/sourceだけを読むPython stdlib再現script。
FILES_SHA256.json  同梱ファイルのsize/SHA。manifest自身を除く。

スクリプト
==========
次は実機engineを起動しない保存データの再集計です。指定commitのrepoが必要です。
出力先はrepo外にします。Pythonのassertが有効な通常実行を使ってください。

python3 tools/audit_route_capture_parent.py --repo <caf35bc-worktree> --out <new-output>/PARENT_EVIDENCE_REPLAY.json
python3 tools/audit_capture_evidence.py --repo <caf35bc-worktree> --output <new-output>/evidence
python3 tools/audit_driver_cost.py --repo <caf35bc-worktree> --output <new-output>/hotpath

配布側run_capture.py、build script、ELF、モデル、GPU、APIをこのbundleのscriptが起動することは
ありません。source/hash/receiptの検査と、新たなlive process検査を混同しないでください。

6本文例はtoken化前、20caseは仕様・split案です。実行済みtoken datasetではありません。
proposalsのpatchは適用可能性と構文のみ検査し、full link/実機採用はしていません。
過去raw・共有原本を上書きせず、新作業コピーで検討してください。
