Xeon MoE Lab 周辺レビュー — 配布物の使い方
2026-11-16

最初に ASTRA_HANDOFF_JA_2026-11-16.txt を読んでください。
TXTだけでも指摘と修正方針を読めます。このZIPは再現に必要なコードと証拠を追加したものです。

対象:
  https://github.com/MangekyoMyan/xeon-moe-lab
  commit 4ff68eba2d8c480692511dc6d219d27a6a102866

この公開版へのレビューです。現行実機・11月12日以降の未公開実装への適用可否は、
実装担当が現行コードと照合してください。公開版の問題を再現するprobeの「成功」は、
その問題が残っていることの確認です。修正後の合格条件へは期待結果を変更する必要があります。

構成
====
ASTRA_HANDOFF_JA_2026-11-16.txt   統合した引き継ぎ本文
build/BUILD_REVIEW_JA.txt        再構築とmanifest照合の4項目
measurement/MEASUREMENT_REVIEW_JA.txt  計測・集計の4項目
operations/OPERATIONS_REVIEW_JA.txt    運用監視の3項目と保留・棄却候補
runtime/collect_runtime_snapshot.py   実行中PIDのbacking fileを読む採取ツール
SOURCE_PROVENANCE.json          repo commit、主要sourceのSHA256、試験範囲
CONTENTS_SHA256.json            この配布物の各fileのSHA256（このmanifest自身は除く）
LICENSE_SOURCE_REPO.txt         同梱した元repoコードへのライセンス表示

measurement/synthetic-fixtures-*/以下は人工入力です。
保存されたANALYSIS/SUMMARYは旧analyzerの実出力で、測定結果の架空例を実機値として扱うものではありません。
median_helper_only.cppも人工標本のための小さな単体コードです。コンパイル済みbinaryは同梱していません。
measurement/handoff_repeat_results.json は引き渡し用CLI整備後の再実行結果です。

runtime/collector_independent_review.json は採取ツール初版への指摘です。
runtime/collector_final_verification.json が修正版への独立確認です。対象SHAはそれぞれのJSONにあります。
この2つを、同じ版への相反する判定と解釈しないでください。

実施したこと・していないこと
============================
Linux/Python3.12/GCC14.2のCPU環境で、人工入力、一時sourceコピー、fake I/O、仮想時計を使用しました。
CMakeによるCPU-only configureでは、小さなcompiler feature probeがコンパイル・実行されます。
measurement probeは実median関数だけを含む小さなC++コードをコンパイルします。
operations probeは固定sourceの指定関数/classだけをASTから読み、外部I/Oをfakeへ置換します。
終了時のLinux process groupの意味を確かめる補助確認では、短命なPython子だけを起動しました。
その結果も linux_exit_semantics.json に残しています。

モデル/GPU/API/実監視サービスの起動・変更、ファン/電力/ECC設定の変更、実機への接続、
通知送信、公開GitHubへの更新は行っていません。repo追跡fileは変更していません。
実機性能、実センサー遅延、言語品質、2 token表示の原因を新たに実測した資料ではありません。

再現環境の目安
==============
Linux、Python3.10以降を推奨します。今回の確認はPython3.12です。
collector単体はPython3.9以降の標準ライブラリと/procを使います。
各probeは標準ライブラリを使い、buildのみCMakeとC/C++ compiler、measurementのみg++またはclang++が別途必要です。
review対象repoは別途用意してください。このZIPに全repo・モデル・DSOは含めていません。

以下はZIPを展開したrootからの例です。/path/to/...は実際の絶対pathへ置き換えます。
出力先は毎回新しい名前を使ってください。

1. ビルド手順の再現
-------------------
python3 build/reproduce_build_review.py \
  --repo /path/to/xeon-moe-lab \
  --cmake /absolute/path/to/cmake \
  --output /path/to/new-build-review.json

--cmakeには実行fileへの絶対pathを指定してください（例 /usr/bin/cmake）。
CPU-only configureを省略する場合は --skip-cmake-configure を追加できます。
UI DOWNLOAD primitiveは一時コピーの中だけで失敗stubに置き換えます。
prepared treeも一時directory内に作り、通常のwork/buildへ触れません。
HEADの固定commit一致と、追跡fileの変更がないことを要求します。
この事前checkは未追跡/ignored fileの完全inventory照合ではありません。
再現には余分なbuild入力を加えていない、凍結版の新しいcheckoutを使ってください。

2. 計測analyzerの再現
---------------------
python3 -B measurement/probe_measurement.py \
  --repo /path/to/xeon-moe-lab \
  --output /path/to/new-measurement-directory

--outputは未作成directoryを指定します。空でも既存directoryは拒否します。
固定HEADと実行対象5ソースのhashが違えば実行前に拒否します。
結果JSONとsynthetic-fixtures-*を新規directoryへ保存します。
保存traceを必ず作ってから元analyzerを実行し、perf実行分岐へ入りません。
元operator/model/GPUは動かしません。median helperのコンパイルだけを行います。

3. 監視のfake I/O試験
--------------------
python3 operations/operations_probes.py \
  --repo /path/to/xeon-moe-lab \
  --output /path/to/new-operations-review.json

対象8fileのhash照合後、指定関数/classだけをASTから読みます。
実startup codeをimportしません。I/O、signal、時計、子プロセスをfakeへ置換します。
想定表示は passed=12 / total=12。正の対照と確定指摘から外した候補も含む数です。
コードのpinを機械的に変更して現行版を通すのではなく、差分と期待結果を見てtestへ移植してください。

4. Runtime collectorの自己試験（実モデル不要）
---------------------------------------------
python3 runtime/test_runtime_snapshot.py

小さな人工fileを現在のPythonプロセスへdataとしてmmapし、13項目を確認します。
ELFをcodeとしてロードする試験ではありません。
削除済みmap_filesの権限がない環境では、正しくunavailableになることを確認します。

独立確認の再現:
python3 runtime/verify_collector_final.py \
  --collector runtime/collect_runtime_snapshot.py \
  --previous-review runtime/collector_independent_review.json \
  --output /path/to/new-collector-independent.json

対象の新旧reportを上書きせず、同inodeのhash後更新とmtime復元を確認します。
正常対照・名前判定を含む24項目が、保存した実行ではすべてpassでした。

5. 実行中の対象PIDへcollectorを使う場合
--------------------------------------
python3 runtime/collect_runtime_snapshot.py \
  --pid 12345 \
  --output /path/to/runtime-snapshot-new.json

12345はすでに動いている対象PIDへ置換します。PIDの自動探索やサービスの起動はしません。
読取権限がなければunavailable/失敗になります。自動で権限昇格や設定変更を試しません。
default512MiB/fileまでのhashです。必要な場合だけ --max-file-mib を明示します。
status=hashedだけでなくverification全体を確認してください。
exit0は記録範囲の観測が成立した意味です。モデル品質や全依存DSO、source/build対応の証明ではありません。
exit2は欠けた情報・途中変更・出力先既存などです。理由を見て次の観測を判断します。

このcollectorはatomic snapshotではありません。個々の最終確認後の変更は見逃し得ます。
backing file hashとsource/build manifestを別々に記録してrun IDへ結びつけてください。

添付bytesの照合
==============
CONTENTS_SHA256.jsonは配布内のpath、size_bytes、sha256を持ちます。
配布物が同じbytesかを確かめるためのもので、署名や外部の信頼rootではありません。
各レビュー結果にも、実際に読んだsource hashを記録しています。
