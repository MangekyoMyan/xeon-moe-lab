xeon-moe-lab route37 review return
2026-11-17 / pinned commit 688683c71803ed94ac7a8e41e8e916d4e0174e6d

最初に読むファイル
==================
1. ASTRA_ROUTE37_NEXT_ACTION_JA.txt
   実機担当向け。full link/provider、scheduler小fixture、専用driverの成功/失敗cleanup、
   4tokenから64tokenへ進む手順。外部提案であり、新しい実機起動許可ではない。
2. ROUTE37_REVIEW_JA.txt
   親の統合判断。相談5問への回答、証拠の範囲、GPU並列という目標との接続。
3. SOURCE_LINKS.txt
   指定commitに固定した主要sourceの完全URL。

判断
====
現collectorは指定された単一context/caller/CPU producerの範囲で維持してよい。
確定的な寿命・overflow・failure混入の欠陥は今回見つからなかった。
次の残件は実libllama/driverとの接続であり、現在の小型fixture成功は実モデルcoverageや
speedupを保証しない。fresh free2575MiBではpositive hot screenは通らず、GPU37–39と
既存guardを保った別memory計画が必要。

収録内容
========
docs/
  route37_api_contract_comments.patch: コメントだけの提案。元worktreeへ未適用。
  COMMENT_PATCH_VERIFICATION.json: git apply --check成功、非comment source同一の確認。
integration/
  実caller/scheduler/CPU経路の行番号付き報告と32sourceのhash。
lifetime/
  collector契約監査、追加12件の小型C++test、runner、結果、compile/run receipt。
  初回のLeakSanitizer環境失敗もinitial_lsan_*として残す。成功に上書きして隠していない。
evidence/
  保存handoff80ファイル、前回31ファイル、source/patch/build、6工程、capacityの独立監査。
  RESTORED_ARTIFICIAL_GOLDEN.binは17,760Bの人工F32 fixtureデータ。モデル重みや実行ELFではない。
tools/
  verify_review_bundle.py: 同梱manifestとファイル集合の検証だけを行う。
PARENT_VERIFICATION.json
  親が試験source/結果/主要sourceを読み、数量と配布状態を再確認した記録。
FILES_SHA256.json
  このファイル自身を除く全配布ファイルのsize/SHA256。

試験の範囲
==========
新しい実行は元route37.cpp + host-buffer stub + direct hookの12契約ケース。
実ggml arithmetic/scheduler/libllama/モデル/GPU/通常APIはここで実行していない。
ASan/UBSanは有効。初回LSanはptrace制約で失敗し、同じ12件をdetect_leaks=0で実行して成功。
TSan成功・leak検査成功・実機数値モデル試験とは扱わない。
元の13,320 scalar比較は保存source/receiptを照合したもので、この12件へ合算しない。

実機のCPU/provider/ELFのhashは今回資料に保存されたreceipt。こちらでremote実体を
再hashしたわけではない。詳細な証拠の限界はevidence報告を参照。

検証・再現（必要な場合だけ）
============================
bundleのファイル検証:
  python3 tools/verify_review_bundle.py <このdirectory>

小型collector試験を再現する場合:
  python3 lifetime/run_contract_test.py --repo <指定commitのworktree>
このrunnerは同梱sourceを読むが、同梱build.pyや実機binaryは使わない。
実行環境が同じLSan ptrace制約を持つ場合のみ--disable-leak-checkを指定する。
同じdirectoryへ新しいreceiptを書くため、元receiptを残すならcopy上で実行する。

保存証拠監査を再現する場合:
  python3 evidence/audit_route_evidence.py --repo <指定commitのworktree> \
    --previous <前回xeon_moe_mixed_review_2026-11-17の展開directory> \
    --output <repo外の新しい監査directory>
前回展開directoryと同じ親に、元のxeon_moe_mixed_review_2026-11-17.zipも必要。
INTAKEとのZIP byte/SHA照合を含むためである。今回bundle内へ前回ZIPは複製していない。

comments patchを採用する場合は実機担当が指定sourceに照合して適用する。
今回はgit apply --checkだけを行い、元repoも稼働APIも変更していない。

このbundleには実行ELF/object、モデル重み、起動自動化、production置換commandを含めない。
重要codeは親が直接確認し、独立担当の全報告と試験sourceも読んで統合した。
