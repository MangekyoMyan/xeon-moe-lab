xeon-moe-lab route-calibration review
2026-11-17 / pinned fa6671f7b5281bad28947fe5324e2495e792a86f

読む順序
========
1. ASTRA_ROUTE_CALIBRATION_NEXT_ACTION_JA.txt
   実機側Astra向けの具体的な次手。現在のF/Dを受理し、容量→小さい並列pilotへ進む。
2. ROUTE_CALIBRATION_REVIEW_JA.txt
   結論、source上の理由、実測の再集計、予測/未測定の区別、言語と表示の解釈。
3. plans/UBATCH_EXPERIMENT_PLAN.json
   固定条件、測定phase、条件分岐、最小入力長、検査/時間の機械可読plan。
4. pilot/PILOT_CASES_JA_EN_CODE.json と PILOT_STAGE_MANIFEST.json
   12family/20caseの全文prompt/reference、train10/validation5/holdout5、5caseずつの実行段階。
   PILOT_TEXTS_JA_EN_CODE.txtは同じ本文の読みやすい展開。
   1280call/47360recordsはbaseline routeを一巡採る場合。候補比較や数値試験の追加callは別。

今回の重要な判断
==============
・F/D4callの全F32一致と1044callの保存証拠を独立に確認。同じ観測suiteの反復は必須ではない。
・cap1に揃えてもCUDA compute4669.26MiBは変わらなかった。
・旧bankの3072/2560MiB guardはどちらもvision前。今回2122.25MiBはvision+text warmup後。
  949.75MiB+payloadの差を旧guard時点の必要削減量と誤読しない。追加の必須guardにもしていない。
・fresh1024の対応M0をまず測る。小bankが既存guardと後続warmupを満たせるなら、512を必須にせず
  bounded positive pilotへ進む。追加容量が必要なら512、根拠があれば256。
・MXFP4 full384 staging一projection2295MiBは512/256でも残る。総arena半減は予測しない。
・小さいubatchで同じweightを再転送する可能性があるため、全prefill時間も比較する。

証拠と再計算
============
verification/PARENT_RECOMPUTED.json
  親がraw F32/時刻を別scriptで再計算。guard旧phaseのfree/必要改善量はunknown。
tools/parent_recompute.py
  実行例（モデルを動かさず、指定pinのfileを読むだけ）:
  python3 tools/parent_recompute.py <pinned-repo> <repo外のoutput.json>
reviews/evidence/
  全共有hash、provider/build、台帳、時刻、数値、復帰receiptの監査とstdlib再集計script。
reviews/workspace/
  実loaded scheduler/allocator/CUDA/model sourceの24件hash、行付き抜粋、shape算術、cost追記。
pilot/
  元6caseの言語/route再集計と、新20caseの作成script・参照program検査receipt。

追加資料
========
plans/CAPACITY_LONG_TEXT_JA.txt
  1024token境界比較に向けた新規長文。capacity専用、bank split対象外。token数は未検証。
plans/TEACHER_FORCE_TOKEN_BOUNDARY_JA.txt
  template/tokenizer境界とnext-tokenの一つずれを防ぐ設計。
reviews/evidence/allocator_probe/ALLOCATOR_RESERVE_PROBE_PLAN_JA.txt
  必要な場合だけ一回のreserve帰属を調べる設計。実装/構文検査済みpatchではない。

実行したこと／していないこと
==========================
実行したのは独自Pythonによるsource/rawの読み取り・整数/統計再計算と、新作reference programの
Python/C++境界確認。供給repoのengine/ELF/script、モデル、GPU、APIは実行していない。
source review対象repoは無変更。新作filesは外側の成果物directoryへ保存した。
ubatch512/256実測、positive bank allocation/admission、モデルの自力回答品質は未実施。
通常APIの復帰は共有receiptの内容を照合したもので、Proがremoteを直接観測した結果ではない。

user-facing本文を含む完成資料はこのpackage内にある。FILES_SHA256.jsonは自身を除く全fileの
byte数/SHA256。ZIPはpackage directory名を保つ。verification/DELIVERY_VERIFICATION.jsonに
検査scopeと新corpus版の一致を記録した。
実機操作の実行者は実機側Astra。人間に追加commandを求める資料ではない。
