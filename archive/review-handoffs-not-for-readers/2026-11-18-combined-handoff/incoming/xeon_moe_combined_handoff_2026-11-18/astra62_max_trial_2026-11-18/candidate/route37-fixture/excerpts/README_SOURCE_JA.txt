追加要求に対応したソース抜粋

対象commit: 37f6390c9a5bc594658127651e204ab11420cd71

内容
・mul_mat_id_route37.c: MUL_MAT_IDのmapping型・macro、演算用helper、incr_ptr_aligned、演算本体。各関数の全体を含む。
・graph_plan_route37.c: ggml_graph_plan関数の全体。
・iq3-cache.h: sparse-cacheの公開宣言。
・mx-rhs.h: MUL_MAT_IDから参照する型判定・workspace・関数宣言。
・route37.h: このソースがincludeする診断用APIの型・宣言。

出典区分
route37はmx-prepack release-candidate-aを元にした診断用CPUソース。現在のlive実機と同一とは主張しない。通常APIの状態や実験結果を示す資料は含めていない。

加工
コメントをすべて同じ行数の空白に置換した。コード・宣言は変更していない。元commit・path・行範囲・元ファイルhash・抜粋hashをSOURCE_MANIFEST.jsonに記録した。各抜粋ファイルの1行目がmanifestの開始行に対応する。

これは完全なtranslation unitや新しいビルド済みruntimeではなく、要求された関数と宣言の抜粋である。ggmlの共通型・macro・宣言は初期配布ソース側の依存関係を参照する。

過去案、速度値、評価、実装方針の助言は含めていない。
