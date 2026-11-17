xeon-moe-lab / parallel-pilot 独立レビュー一式
2026年11月17日
対象commit: d4d50b651ff7754ee2189228762468fc02971bfa
引き継ぎ改訂2。まずHANDOFF_ADDENDUM_JA.txtを読む。
採点の過剰制約と実装移行時の注意を修正した。実測評価・13ページ報告書・original35・3例は同じ。

ユーザーさん向け

USER_REPORT_JA.pdf : 13ページ。過去の経緯から今回の並列化、9.8との条件差、用語集、次の進路。
USER_REPORT_JA.txt : 図を数値と説明へ置き換えた同内容。一次資料の完全URLも収録。

実機側アストラ向け

まずASTRA_NEXT_JA.txtを読む。今回の結論は支持し、固定35の未使用3例と実回答へ進む提案。
TECHNICAL_REVIEW_JA.txtに、今回の相談の各論点への判断、証拠、残る制約をまとめた。
今回のfile受領自体を、人間による設定方針変更や本番採用の明示指示へ読み替えない。

ディレクトリ

evidence/    保存rawの全要素監査、8,228checks、不整合0、sourceと結果の対応。
integration/ 統合sourceの所有権・同期・mode・capture監査、任意B対照案、route originの最小設計。
history/     以前の採用CPU改善、旧hot14、9.8近傍記録と比較条件の再集計。
parent/      親の独立計時・65行の確率分布再集計。追加データの解釈もTECHNICAL_REVIEWに記載。
validation/  次に評価する3例の完全凍結入力・token IDs・rubric・小さい実験計画と検証receipt。
preflight/   採点の反例を確認した自作コードと、初回からの入力不変・改訂内容の確認記録。

proposalと測定結果を区別する

validation/FROZEN_VALIDATION3_PLAN.jsonは自己完結した宣言的な計画で、engine runnerではない。
現driver/analyzerは固定ci=2/prefix272/品質filename等の変更が必要。詳細はVALIDATION3_HANDOFF_JA.txt。
validation3は3例・2独立課題family。新しく実行した結果ではない。

integration/optional_dormant_bank_original_graph.patchはsource SHA guardを持つ任意の診断案。
git apply --checkのみ成功。未適用・未build・未実測。driverのB-held対応も別途必要。
B対照やroute originの改修を、固定35の未使用例評価の必須前提にはしていない。

このレビューで行ったこと

固定repoを読み、独自のoffline Pythonで保存JSON/F32/ログ/sourceを再集計した。
提供script/ELF/model/CUDA/APIを実行・importしていない。repoは変更なし。
改訂2では別途、レビュー側で自作したring bufferの反例を実行し、採点の過剰指定を確認した。
モデル回答の品質評価でも、供給referenceコードの実行でもない。
実機build、サービス停止/復帰、設定変更、推論benchは実機側アストラが引き続き担当する。
通常APIの復帰については保存receipt時点を照合した。現在の実機へ接続したという意味ではない。

補助scriptの再利用

各scriptは保存データを読むもの。repo引数をcommit d4d50b6のローカルcheckoutへ明示して使う。
evidence/audit_parallel_pilot.pyには --repo / --output / --prior-zip の引数がある。
parent/independent_times_and_quality.pyは repo引数と --out を受け取る。
history/recompute_history.py、integrationの提案生成・source抽出、validationの生成scriptも
各先頭docstring/引数を参照する。既定値のこのレビューworkspaceパスを実機パスと混同しない。
元193MB超のrawとモデル重みはZIPへ複製していない。必要なrawは同じGitHub commitの既存資料を使う。

FILE_MANIFEST.jsonはこの一式のpayloadのSHA256/sizeを収録する。
子ディレクトリの既存manifestは各担当成果物時点の記録で、総合の配布物照合にはroot FILE_MANIFESTを使う。
REVIEW_VERIFICATION.jsonはscope、原データ照合、PDF全頁目視、提案の適用可能性、repo cleanの記録。

主要一次資料
https://github.com/MangekyoMyan/xeon-moe-lab/tree/d4d50b651ff7754ee2189228762468fc02971bfa/archive/review-handoffs-not-for-readers/2026-11-17-parallel-pilot
