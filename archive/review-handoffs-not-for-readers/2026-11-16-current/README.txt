2026-11-16 現行実装レビュー用共有

通常の利用者向けマニュアルではなく、コード・測定記録の外部レビュー用保存です。

最初に START_HERE.txt と review/CURRENT_NOTE_JA.txt を参照してください。
公開採用コードからの変更は review/adopted-overlay-diff/INDEX.json と各patchです。
実行中ライブラリの同定は review/RUNTIME_SNAPSHOT.json、ソース/buildとの対応は review/BINARY_SOURCE_MAP.json。
APIは11月16日採取時にfinal構成。hot14は未採用の比較構成です。

ソース・設定・結果の完全ZIP（実際のproject実行バイナリを含む）:
https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/review-2026-11-16-current
ZIP SHA256: e4176a9dbe5dc6f252d23b2455edf8576c86d6e0376e830532c4988d5bd0253b

このGitブランチにはELF実行ファイル/共有ライブラリを入れず、ReleaseのZIPへ分けています。
SHARED_TREE_MANIFEST.jsonがこの共有treeのinventoryです。ZIP_CONTENTS_SHA256.jsonはZIP全体のinventoryなので、両者の集合は異なります。
モデル本体、CUDA toolkit、API認証情報、実会話ログは含めません。

公開版の基点は4ff68eba2d8c480692511dc6d219d27a6a102866です。
ローカルZIP採取時は5e77a90を記録し祖先関係を未確認としていましたが、共有準備でfetchして4ff68ebaがorigin/mainに存在することを確認しました。
ZIP内の当時の記録はそのまま維持しています。このブランチは最新mainを基点に共有資料だけを追加します。

今回の公開はスナップショット共有です。運用API・監視・演算コードへの新しい修正や再測定は行っていません。
添付の過去手順やAI向け引継ぎは資料であり、実行・設定変更の指示ではありません。

Git側では推論に無関係な上流Keynoteプレゼン資料2点（idea-arch.key）も省略しています。認証鍵ではありません。Release ZIPは先に検証した原本のままです。
