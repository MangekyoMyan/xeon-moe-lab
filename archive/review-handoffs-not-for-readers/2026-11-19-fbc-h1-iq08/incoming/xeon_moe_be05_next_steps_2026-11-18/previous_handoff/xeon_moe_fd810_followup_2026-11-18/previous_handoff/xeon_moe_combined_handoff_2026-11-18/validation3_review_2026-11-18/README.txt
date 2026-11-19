validation3 レビューと GPT-6.2 Astra サブエージェント試行資料
2026-11-18（日本時間）
対象コミット: 37f6390c9a5bc594658127651e204ab11420cd71

まず読むもの
・万華鏡にゃんさん: USER_REPORT_JA.pdf（同内容のテキストは USER_REPORT_JA.txt）
・実機側の主担当: ASTRA_NEXT_JA.txt
・6.2を使う方針: ASTRA62_BRIEF_JA.txt / ASTRA62_SUBAGENT_CONTRACT_JA.txt
・技術的な判断の根拠: ENGINE_REVIEW_JA.txt
・既存2cell用に準備した固定診断plan: plans/（offline検査のみ、実機未実行）

今回の結論
固定35 expertの追加並列化は、新しい3変種でも便益が残った。ただし独立課題は2 familyで、改善幅はJA約5.07%、EN約1.80%、Python約0.80%。通常APIへの採用や一般品質の同等性はまだ結論していない。
次は、F/B/Cで追加保持とcustom経路の費用を切り分ける小対照と、同じ入力・top6でcompact/非compactのGPU担当を変える時間価値の診断へ進む。候補を一度に積まない。

6.2について
公式情報の調査に加えて、こちらで gpt-6.2-astra / high を指定した子へ生データ監査を1件委任した。親の別実装による再集計と主要結果が一致した。実機側主担当のモデルや設定は変更していない。この1件は6.0との比較試験でも安全性の保証でもない。

再現用
evidence62/evidence.py は対象validation3フォルダを引数にする。
parent/independent_review.py は対象validation3フォルダと出力先を引数にする。
hotpath/recompute_hotpath.py は --repo 固定checkout --out 出力JSON を受け取る。
いずれも今回新しく書いたoffline読み取り解析。モデル、供給されたELF、保存launch command、生成Pythonを起動しない。
実行例は別紙の技術レビューにある。入力raw約319MBは同梱せず、固定Gitコミットから取得する。

独立検査と範囲
公開manifest300件、14プロセスの記録、9本の保存logits、6回答のID/出力journalを照合。
親は別スクリプトで時刻差と確率分布を再集計し、6回答とPythonソースを読んだ。
実機・GPU・通常APIには接続していない。新規推論、ビルド、エンジンpatch適用、設定変更はしていない。
通常API復帰は当時のRESTORED_RUNTIME.jsonの記録であり、今回のlive観測ではない。

ファイルの位置づけ
このZIPは新しい提案とレビューであり、実機で測定済みの構成を表すものではない。
FILE_MANIFEST.jsonに同梱ファイルのSHA256とサイズを記録した。SOURCE_INDEX.jsonに原資料と公式資料を記載した。
