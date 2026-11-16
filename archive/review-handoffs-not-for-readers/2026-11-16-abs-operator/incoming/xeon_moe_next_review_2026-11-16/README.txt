xeon-moe-lab 次の最適化レビュー / 2026-11-16

最初に読むファイル
  ASTRA_NEXT_OPTIMIZATION_REVIEW_2026-11-16_JA.txt

対象の相談票とsource
  repository: https://github.com/MangekyoMyan/xeon-moe-lab
  commit: 0c9eb472d2460048fb44cca949ad3722ed27b93b
  archive/review-handoffs-not-for-readers/2026-11-16-next-optimization/CONSULT_PRO_JA.txt

この返却物は実装前の設計レビュー、保存ログの独立再集計、限定correctness素材。
本体へ適用するpatch、実Xeonの新しい性能測定、実GPU collectorは含まない。
CPU試作はこの端末のEPYC上でIvy Bridge向けに新規compileして確認したもの。
生成ELF・object・モデル・過去の配布binaryは含めていない。

推奨順序
  1. IQ2_S gate/up、IQ3_S downの実形状・6 experts・12 workersをcurrent compact基準で測る。
  2. shared absとworker-private lazyを、prepare/同期/dispatch込みで比較する。
  3. 全40層route censusとhotのhost traceを別用途として準備する。
  4. 必要ならGPU timing events、別候補IQ3_S offset4へ進む。
詳細な条件、止めどころ、未検証事項は統合報告に記載。

directory構成
  cpu_evidence/
    CPU_SCREEN_REVIEW_JA.txt      Xeon raw、旧bias62、旧2-row、cache、Amdahlの監査
    XEON_SCREEN_REANALYSIS.json   独立再集計とsource manifest
    reanalyze_xeon_screen.py      stdlibのみ。保存資料を読み、指定出力へ再集計
    XEON_ABS_SCREEN.png           全12 abs条件、2socket、中央値とmin–max（CIではない）
    plot_xeon_abs_screen.py       上記JSONから図を作る。matplotlibを使用
  cpu_integration/
    CPU_INTEGRATION_REVIEW_JA.txt workspace/barrier/fallback/private lazyの設計
    abs_integration_contract.cpp shared/private公開契約の限定std::thread fixture
    run_abs_integration_contract.py  compileして上記fixtureを実行
    contract_run/RESULT.json      692case / 295,392出力のreceipt
    OFFSET4_INDEPENDENT_READ_REVIEW_JA.txt  新候補の独立コード照合
    SOURCE_MANIFEST.json / ENVIRONMENT.json
  offset4/
    IQ3S_OFFSET4_REVIEW_JA.txt     新4bit形式の式、範囲、fallback、性能未測定の限定
    iq3s_offset4.inc              新format、prepare、fast/compat reader
    iq3s_offset4_probe.cpp        現compactとのbit比較
    run_iq3s_offset4_probe.py      新規compile、correctness、objdump
    analyze_offset4_assembly.py   保存objdumpから3関数と静的opcode集計を抽出
    offset4_run/                 RESULT、assembly、静的集計。ELFなし
    assembly_reproduction/       同じASMから再集計一致・異常header拒否のreceipt
  gpu_trace/
    GPU_TRACE_DESIGN_JA.txt       3段階のhook、clock、lifetime、overhead、source:line
    TIMING_EVENT_ADAPTER_SKETCH.txt  内部ABI adapter案。未compile・未実行
    TRACE_SCHEMA_JA.txt           schema、validatorのscopeと限界
    analyze_moe_trace.py          stdlibのみのoffline validator
    test_analyze_moe_trace.py     10合成test
    SYNTHETIC_*.json              すべて人工データ。実測性能ではない
    TRACE_ANALYZER_TEST_RESULTS.json / GPU_TRACE_SOURCE_MANIFEST.json
  provenance/
    INTAKE_VERIFICATION.json     新handoff/incoming/commit差分の照合
  PACKAGE_VERIFICATION.json      最終packageの検査結果
  FILES_SHA256.json              このmanifest自身を除く全同梱fileのSHA256とsize
  LICENSE_XEON_MOE_LAB.txt / LICENSE_LLAMA_CPP.txt

再現方法
========

指定commitのcheckoutを /ABS/PINNED_REPO とする。以下の出力名・directoryは新規を使う。
sourceは巨大な全modelを含めてdownloadする必要はなく、pinned repositoryの記載sourceを使う。
前回のincoming45filesはrepository内の参照を利用し、このZIPへ重ねて同梱していない。

1. 保存Xeon結果の再集計（新しい性能測定は行わない）

  python3 cpu_evidence/reanalyze_xeon_screen.py --repository /ABS/PINNED_REPO --output /ABS/NEW_SCREEN.json

2. shared/privateの限定契約確認（compilerとpthreadが必要）

  python3 cpu_integration/run_abs_integration_contract.py --repository /ABS/PINNED_REPO --output /ABS/NEW_CONTRACT_DIR

実ggml backend/OpenMP/allocatorの試験ではない。実際のgraph/work_data再利用試験は次段階。

3. offset4の算術確認（compilerとobjdumpが必要）

  python3 offset4/run_iq3s_offset4_probe.py --repo /ABS/PINNED_REPO --output /ABS/NEW_OFFSET4_DIR

このrunnerに性能benchmark modeはない。新payloadのformat versioningは未統合。
新readerだけを現cacheへ接続してはいけない。新payloadに旧readerを使えない理由は報告を参照。

4. 同梱assemblyの静的集計を再現（compile/ELF実行なし）

  python3 offset4/analyze_offset4_assembly.py --assembly offset4/offset4_run/probe.asm --output /ABS/NEW_ASM_DIR --expected-summary offset4/offset4_run/ASSEMBLY_SUMMARY.json

別compilerで新規compileしたASMを調べる場合、--expected-summaryを省略する。
これは命令数とstack operandの静的観測であり、cycle数・動的spill数・速度予測ではない。

5. offline trace validator

  python3 -B -m unittest discover -s gpu_trace -p test_analyze_moe_trace.py -v
  python3 -B gpu_trace/analyze_moe_trace.py --input gpu_trace/SYNTHETIC_TRACE_INPUT.json --output /ABS/NEW_TRACE_ANALYSIS.json

SYNTHETIC_HOT_TRACE_INPUT.jsonは宣言したhot subsetのみ。全40層censusの観測とは異なる。
実collectorのhook位置、phase identity、event generationの正しさをこのvalidatorだけで保証しない。

sourceとライセンス
=================

codeと静的assemblyは対象repoのcompact kernel / llama.cpp由来の実装を参照・派生している。
対象repoおよびllama.cppのMIT licenseを同梱した。上流sourceを追加配布する場合も
元の著作権・license表示を維持する。依存header/kernelの完全sourceは上記pinned commitにある。
新probeは既存kernelをincludeして比較し、既存incoming probeのSHAを検査する。

記録内の/workspace/...は今回の実行時path。別端末では上の引数を自分のcheckoutへ置き換える。
保存receiptのpathや数値を別の端末で測った値へ書き換えない。再実行は新しいreceiptへ残す。
