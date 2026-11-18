6.2 Astra / max 独立探索・検証パッケージ
2026-11-18 JST

このZIPはそのまま実機側のメインエージェントへ渡せます。
実機への適用・API変更は行っていません。

読む順番

1. USER_REPORT_JA.pdf
   ユーザー向けの11ページの報告書。用語、7案、最終計時、見つかった問題、能力評価、次の実機作業を説明。
   USER_REPORT_JA.txt は同じ内容のテキスト版。

2. ASTRA_HANDOFF_JA.txt
   実機担当向け。現行実装との違い、patchの適用関係、DSO識別、混在compactの制約、実機での比較順。

3. candidate/REPORT_JA.md と candidate/REPRODUCE_JA.md
   試験対象の6.2自身が作成した報告と、別source/buildで再現する手順。
   親の評価と子の自己報告は別のファイルに保持。

4. evidence/
   親の独立再集計、数値確認、静的レビュー、初期案と過去案の照合、反例、build対応確認。

主なフォルダ

candidate/
  子が固定した成果物一式。patch、試験コード、再現スクリプト、raw全行、実行対応manifest、圧縮した代表出力dump。
  SHA256SUMSは子が固定した178ファイルの表。親が全ファイルを照合した。

protocol/
  モデル/effort指定、初期source範囲、追加情報と親の指摘の台帳、VM環境、初期sourceの全SHA。

evidence/parent/
  最終880行等の再集計、独立整数参照、符号式の列挙、直接dot比較、最終全出力の照合。
  親の検算scriptは当時のworkspace構造やlibdirを入力にするものを含む。原実行時の絶対pathは出典記録。

evidence/reviews/
  route、Q8、符号修正、旧/最終DSOの特徴監査。途中版の判定はその時点のsnapshotに限定。

evidence/history/
  初期案を固定してから行った、既知の後続実装との照合。子には初期情報として渡していない。

evidence/mixed_fixture_review/
  追加要求後の混在fixtureを親側が別途確認した記録。1,920実行行＝960 A/B組、282,240出力。

evidence/frozen/
  初期案と、旧/最終のhash記録。compiled DSOsや実行ファイルそのものは含めていない。

適用関係

・03、04、06は固定baselineに対する独立候補。
・07は03+04+06の合成。これ一つで3変更を含む。
・01はroute旧版。
・02/05は非nearestでの既知の数値不一致がある履歴比較用。
・08はroute37関数抜粋へのfixture用差分。現行runtime全体へのport完了を示すものではない。
・全patchを番号順に適用する形式ではない。

評価の範囲

初期source: commit 37f6390c9a5bc594658127651e204ab11420cd71の公開採用済み本体。
後続実験を含む最新実機全体を初期資料として渡したわけではない。
GPU、GGUFモデル重み、実機API、モデル全体のtoken/s、一般品質は今回未検証。
最終計時は修正後の別source/buildに対応。旧版の不利な結果や反例も履歴として保持。
6.0との同条件対照はなく、モデル順位や一般安全性の認定ではない。

再現の入口

GitHub固定commitのsourceを用意し、candidate/REPRODUCE_JA.mdに従ってcandidate/reproduce.shを新しい出力ディレクトリへ実行。
パッケージには巨大な上流sourceやモデル、VM用compiled DSOsを重複収録していない。
別のpath/compilerでbuildしたELFはhashが変わり得るため、同じpatch、機能、入力、出力を確認する。

SHA256SUMS.json はこの最終パッケージ全体のファイル表。
原ソースのライセンスはprotocol/LICENSE、protocol/THIRD_PARTY_NOTICES.mdを参照。
