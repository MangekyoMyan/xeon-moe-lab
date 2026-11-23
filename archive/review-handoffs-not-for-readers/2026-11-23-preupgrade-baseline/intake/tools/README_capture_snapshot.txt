capture_snapshot.py 使用説明

目的
  既存のruntime.jsonを参照し、通常APIとハードウェアの現在状態を保存する。
  Python標準ライブラリのみを使用。Python 3.9以降。
  import、--help、fixtureテストによってモデルをロードしたりAPIを呼んだりしない。
  実行時もサービス起動/停止、sudo、clock/電力設定変更は行わない。
  lscpu、dmidecode、nvidia-smi等の存在する読み取りコマンドと/proc、/sysを使う。

実機で指定するもの
  --profile：同梱 reference/current_runtime_633e.json へのパス。
  --lab-root：実機に既存の /home/codex/projects/llm-lab。
  --pid：現在確認した通常llama-serverのPID。過去のPIDを流用しない。
  --out：今回専用の、まだ存在しない出力ディレクトリ。
  --label：old-production-before等、1〜128文字の記録名。

コマンドの形
  以下の変数は実機メインが実在する絶対パスとPIDに解決してから使用する。

  python3 "$BENCH_KIT/tools/capture_snapshot.py" \
    --profile "$BENCH_KIT/reference/current_runtime_633e.json" \
    --lab-root /home/codex/projects/llm-lab \
    --pid "$BENCH_SERVER_PID" \
    --out "$BENCH_STUDY/old-production-before" \
    --label old-production-before

  復旧後はPIDを確認し直し、別の新規--outで再取得する。
  --pidを省略した場合、実行中プロセスを確認していないと記録する。
  benchmark driverはserver exeと異なるため、server PIDとして指定しない。
  OLD_P12のactual worker観測はハーネスで別途行う。

保存
  snapshot.json：期待値、観測値、hash、hardware、process、未取得理由。
  summary.txt：短い状態説明。
  config、exe、4DSO、plan、2receiptの9ファイルを小ファイルとしてSHA-256照合する。
  読取前後のstatやmapped device/inodeの違いを確認し、途中で変わったものを一致扱いしない。
  proc mapsの観測は実行中メモリの全バイトをhashした証明ではない。
  argvと環境変数は性能関係の許可項目に絞り、認証等は保存しない。
  実機でファイルの取得権限がなければ未観測とする。

巨大な重み
  既定ではGGUFとprojectorはsize/statのみ。過去receiptと期待SHAは参照として残す。
  content未再読はall_weight_contents_match=nullであり、false（不一致）ではない。
  size一致は別欄。size/stat一致から内容一致を証明しない。
  --full-weightsを明示したときのみ全内容を読み、今回SHAを計算する。
  参照で約347 GBの読み取りになるため、通常の交換前記録では既定動作を推奨する。

終了状態
  exit 0：収集処理が終わった。summary.statusはcollectedまたはpartialになり得る。
    optional未取得はpartial。exit 0を「全比較条件が確認できた」と読み替えない。
  exit 3：不一致、必須の未確認など。summary.findingsを確認する。
  その他：入力、保存等のエラー。既存出力は上書きしない。

  交換後に旧profileと比較すればCPU型番/core数の違いは意図された差である。
  新CPUでexit 3となっても、意図されたhardware変更と、意図しないbinary変更を分けて読む。
  差をなくすために観測値や旧profileを書き換えない。

限界
  TIDのaffinity/stat/last CPUは観測するが、decodeで実際に動いたteamを証明しない。
  cpu MHz等のsnapshot値は負荷中の持続周波数ではない。
  runtime参照のNUMA設定は、そのプロセスの全ページ配置の実測ではない。
  OS/権限/ツールの有無によりDMIやGPU等にunknownが生じる。
  これらは品質欄と未取得理由に残し、推測で埋めない。

機能検証
  kitディレクトリから：
    python3 -m unittest discover -s tests -p test_snapshot.py -v
  自作の小さなファイルとmockを使用。Xeon実機のハードウェア性能試験ではない。
