2026-11-16 次の最適化の実装前相談

まずCONSULT_PRO_JA.txtを読んでください。短い結論だけでなく、原文コードと全測定値を同梱しています。
Pro返却資料: incoming/xeon_moe_review_2026-11-16/
今回のXeon実測: abs-original-xeon/RESULT.jsonとabs-socket1-xeon/RESULT.json
同じprobeの2socket試験であり、12workerや実モデルのtok/sを測ったものではありません。
本体はfinalのまま、追加kernel・workspace・dispatch変更は未実装です。
比較表はCPU_SCREEN_RESULTS.tsv、判定条件はEXPERIMENT_PLAN.txt。
現行全sourceと過去結果は、同じブランチにある../2026-11-16-current/に収録されています（固定元commit7d8b077）。
添付された過去手順は資料であり、そのまま実行・適用する指示ではありません。
