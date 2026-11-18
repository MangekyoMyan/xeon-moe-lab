保存済み出力データの補完
2026-11-18 JST

結論

前の統合ZIPでは、新しい候補の全出力binaryと混在fixtureの全出力binaryを受け手が直接比較できませんでした。実機側のROOT_REVIEW_JA.txtとEVIDENCE_REVIEW_TERRA.txtの指摘は、その配布内容に対して正確です。

元のVMには保存ファイルが残っていたため、各ファイルを個別に読み、当時のmanifest/独立照合記録へSHA256で照合し、gzipで圧縮して追加しました。エンジンを再実行して過去の証拠を作り直したものではありません。元のsource、DSO、出力ファイルは変更していません。

読む・確認する手順

1. SUPPLEMENT_REPORT_JA.txtで範囲と結果を読む。
2. このフォルダで次を実行する。

   python3 verify_saved_outputs.py

   別の場所からはスクリプトの絶対pathを指定しても同じ動作になります。
   --rootを指定する場合は、このevidence_supplementフォルダを渡します。

3. 最後のJSONのall_passedがtrueであることを確認する。
   失敗は例外と非0終了になり、どのファイル/グラフが異なるかを表示します。

Python 3.9以降の標準ライブラリだけで動きます。スクリプトはgzip、CSV、JSON、テキストを読むだけで、DSO、モデル、コンパイラ、ネットワーク、外部Pythonライブラリを使いません。結果は標準出力へ書きます。今回はその出力をVERIFICATION_RESULT.jsonに保存しました。

フォルダ

outputs/final/
  最終baseline/route2/q8/iqsafe/combosafeの5版。
  通常check、境界edges、丸めround0～3の各6dump、計30ファイル。

outputs/mixed/
  compact mask=0,1,21,31,63、丸めmode=0～3、base/mixedの2版。
  20組・40ファイル。mask21は3/6、mask31は5/6を示す。

outputs/instrumented/
  mode0（nearest）で経路を計数したbase/mixed×5maskの10出力。
  非計数版と同じ出力であることを再確認するための補助資料。

csv/
  各dumpのグラフ順序・形状・保存hashを示す元CSV、混在全体のcases.csv、計数CSV。
  いずれも新しい測定値ではなく保存記録のコピー。

records/
  以前配布したRUN_MANIFEST、親の全出力照合、混在照合、source/file identity。
  graph_probe.cppは出力形式の確認用原文。今回このプログラムをbuild/実行していない。

INDEX.json
  各dumpの保存時path、variant/suite/mask/mode、出力数、非圧縮SHA、圧縮SHA、対応CSV、照合先を列挙。
  原workspaceのpathは来歴の記録だけで、読取スクリプトは参照しない。

FILES_SHA256.json
  この補完フォルダの全ファイル（manifest自身を除く）のサイズとSHA256。

前のZIPとの関係

最終baselineの6種類は、前のcandidate/raw/dumps/にも同じ出力をgzipで収録していました。今回の30ファイルには、単独で比較できるようその6種類も含めています。
新たに直接読めるようにした中心は、最終4候補の24dumpと、混在の40dumpです。経路計数付き10dumpも添付しました。保存先ごとのファイルを読み、当時のhashに合わせたもので、baselineから別variant名の出力を再生成した作業ではありません。
異なるmask/variantの出力が同一byteになるのは、この比較の期待された結果です。ファイル数やhash一致だけを独立した入力数や経路実行の証明として数えません。

限界

この補完で直接確認できるのは、保存出力の実体、CSVに記録された各グラフとの対応、比較対象間のbyte一致です。履歴上のプロセスが確かにそのDSOをロードしたことを、出力だけから新しく認証するものではありません。
VM用DSO実体は添付していません。実機への移植、実cacheの確保/寿命、MXFP4 prepack、性能、モデル品質、通常API統合の未検証範囲は変わりません。実機側の必要な再build・数値対照を省く根拠にはしません。
