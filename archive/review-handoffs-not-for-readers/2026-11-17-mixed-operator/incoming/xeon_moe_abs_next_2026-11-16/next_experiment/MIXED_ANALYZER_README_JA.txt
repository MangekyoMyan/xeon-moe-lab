新 mixed operator protocol 用の offline analyzer
==============================================

対象は「これから行う」F/E外側・E内off/shared比較のschema1である。
以前のreal-*/fixed-*結果をこのschemaの結果と混ぜない。
stdlibのみ。engine/source/model/ELF/APIを実行せず、JSONLと保存expected bytesだけを読む。

実行:
  python3 analyze_mixed_run.py --run-root RUN_ROOT --output NEW_ANALYSIS.json

入力はRUN_ROOT/process-0〜process-7。それぞれSTATUS.json、raw.jsonl、
expected-C0.bin、expected-C6.bin、expected-Cplan.binを置く。
省略時の出力はRUN_ROOT/MIXED_ANALYSIS.json。既存reportを上書きしない。
異常・partialはpassed=falseのreportと終了code1を返し、入力証拠を削除・変更しない。
出力先自体に書けない場合は明示的エラーをstderrへ出す。

固定期待:
  outer F,E,E,F,E,F,F,E
  pairs (F0,E1),(F3,E2),(F5,E4),(F6,E7)
  cache順rotation 0,1,2,0（隣接pairごと、基底C0,C6,Cplan）
  3cases/process、16blocks/case、4arms/block、32graphs/arm
  8process全体で1536chunks、49,152timed graph calls
  各E process/caseでOとSは各1024graph calls

48route IDs、Cplan39selected IDs、input SHA、final-plan SHAはpinned commit31b1292の
固定fixtureと同値をanalyzer内に持つ。priority/generated_case_v2/mixed_case.jsonと一致。
Cplan masksはselectedとordered ID列の交差から別途計算する。
selectedの順序はmembership一致に加えて全8process同じ配列であることを要求する。
C0は空、C6は48route全選択、Cplanは39ID。bytesはcount*6,635,520で照合する。

統計:
  primary: 各E process内のO2arms/S2armsから16paired ratios、中央値、ratioの算術平均、
           同量のgraph総時間比を別々に出す。4process反復は別のまま保存。
  placebo: Fのlogical O/Sを同じ方法で出す。全actual_mode=0なのでmode切替効果ではない。
  cross: 各pairでF4arms平均 / EのOまたはSの2arms平均。Fの生sumをそのままEへ割らない。
         Fは2048graphs/case、E logicalは1024graphs/caseなので、F総時間/2で等量化する。
  aggregation: process別/隣接pair別の比を残し、等量の総和比も別に出す。
    pooled_block_paired_medianは全4process/pairsの64block比をpoolした中央値。
    64個の独立process観測ではなく記述統計であり、process反復数は4のまま。
    閾値選択に参照するとしても、新しい実機raw未取得段階の仕様である。

期待outputは各caseで8processのfull bytes一致を確認する。case間の一致は前提にしない。
保存bytesはmode0 expectedであり、candidateの最後のtimed outputとの一致はproducerが
メモリ内で検査する契約。全graphのcandidate出力dumpを独立解析したとは書かない。

STATUS/library/provider/hash、pins、NUMA、case/block/arm完全性、logical/actual mode、
begin/end/elapsed、ms/graph換算、process非重複、full expected bytes等を検査する。
STATUSはexit0/error=nullだけでなくpassed=true/state=completedを要求する。
metaのclock=CLOCK_MONOTONIC、hashes_verified_by_harness=falseを要求し、CPU/baseの
provider pathをlauncher identityと照合する。case spec/header/launcher SHAも有効かつ
8process共通を要求する。古い不完全なSTATUS schemaを互換受理しない。
shard_statとextra_library_dirsも必須とし、JSONのkey順を正規化して8process同一を検査する。
外れ値を除かず、bootstrap/CI/p-value/自動採否は行わない。
「passed=true」はschemaと証拠検査の通過であり、性能採用を意味しない。

synthetic tests:
  python3 test_analyze_mixed_run.py --output NEW_TEST_RESULT.json

このtestは架空の時計と生成byteを使い、engineを動かさない。TEST_RESULTには
SYNTHETIC OFFLINE ANALYZER TESTS; NOT PERFORMANCE EVIDENCE と明記する。
既知比の検証は、F4/E2の2倍誤りを検出する。外れ値保持、mode不整合、partial、
欠落、重複、非finite、clock不整合、出力差、library/hash差も拒否する。
