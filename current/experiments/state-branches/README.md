# x9ram の会話を保存して枝に戻る

テキスト会話の途中で名前を付け、同じ地点から複数の会話を進められます。枝を切り替えるときに、今の枝の未保存部分を保存します。保存してある枝に戻ると、次の質問から続けられます。

## 実機で使う

サーバーの一つ目の端末で次を実行し、`Branch CLI ready` の表示を待ちます。

```bash
cd /home/codex/projects/llm-lab/engine-state-phase1-20261128
sudo ./branch open
```

この端末は開いたままにします。1台のGPUを専有するため、通常APIが稼働中なら処理中の要求を待って一時停止します。CLIは専用のUnixソケットで動きます。終了時はこの端末で Ctrl+C を押してください。起動前に動いていた通常APIが復帰します。保存した枝は次回も使えます。

同じサーバーの二つ目の端末で、次を実行します。

```bash
cd /home/codex/projects/llm-lab/engine-state-phase1-20261128
./branch turn --text 'これから箱番号を一つずつ登録します。'
./branch mark 共通 --pin
./branch fork A --from 共通
./branch checkout A
./branch turn --text 'この枝の箱番号は731です。'
./branch checkout main
./branch fork B --from 共通
./branch checkout B
./branch turn --text 'この枝の箱番号は284です。'
./branch checkout A
./branch turn --text 'この枝の箱番号を数字だけで答えてください。'
```

共通地点にいる間に `mark` してください。各枝に同じ名前や同じ長さのテキストがあっても、内部の枝ID・保存点IDと、実際に処理したトークン列によって区別します。回答本文は受信中に表示します。集計用には `./branch --json turn --text '…'` で、追加処理したトークン数や最初の本文を受信した時刻も取得できます。

現在のheadがその名前の保存点IDを指している間にもう一度 `mark` すると、画像を確認して同じIDを返します。その画像が退避・欠損・破損していれば `missing_image`、同じ名前が別のheadを指していれば `mark_exists` を返します。どちらの拒否でも枝と保存点の参照を保ちます。現在地を改めて保存するには新しい名前を使ってください。

| 操作 | 確認・変更できること |
|---|---|
| `./branch list` | 枝名、親、現在の枝、世代、処理済み位置、未入力のサンプル数、最後の利用、再入力数、保持容量 |
| `./branch inspect branch A` | 枝のheadと、保存済みなら実ファイルのSHA-256・サイズ・実トークンとの一致 |
| `./branch inspect snapshot 共通` | 名前または不変IDで保存点を検査 |
| `./branch fork 別案 --from 共通` | 同じ保存点を参照する枝を作成。`--from` を省くと現在地を保存してから分岐 |
| `./branch pin branch A`、`./branch unpin mark 共通` | 枝のheadまたは名前付き保存点の保持指定 |
| `./branch release branch B`、`./branch release mark 共通` | 指定した参照を解除。別の枝・保持点に必要な画像は共有を継続。pinを解除してから実行 |
| `./branch budget 1073741824` | ディスク上のnative画像予算をバイトで指定。初期値は1 GiB |
| `./branch recover` | 物理スロットが失効したとき、記録してある論理headを再構築 |

予算を超えると、保持指定のない画像を最後の利用順に外します。トークンログ・親子関係は残るため、戻るときに祖先から再入力できます。保持指定と切替に必要な画像が収まらない場合は、その操作をエラーとして返します。SQLiteのメタデータ容量は別に表示します。画像を作る瞬間には、確定前の一時ファイル1件分のディスク空間も使います。

## 入力と復帰の意味

`turn --messages 会話.json` には、役割と本文を持つメッセージ配列を渡せます。テキストと記録済みのツール結果をテンプレートに通し、正規化したトークン列を照合します。復帰時には記録済みのツール結果を入力として再利用します。現在地の全トークン列が新しい入力の先頭と一致すれば、そのまま続きを処理します。編集した場合は、一致する祖先のうち最も長い保存点を選びます。適合する画像がなければ先頭から計算し、実際の再利用・再計算数を返します。

対象は完了したターンからの続行です。生成済みの末尾1トークンは、まだモデルに入力されていない場合があります。処理済み列とサンプル済みの末尾を別に保存し、次のターンのテンプレートが要求する分を一度だけ評価します。乱数状態を含む文字途中からの生成、画像入力、Open WebUIの分岐操作との連携は未検証です。

## 実装と回復

管理プロセスが専用スロットの唯一の利用者になります。すべての会話入力と切替を同じキューで順番に扱い、同一バックエンドや保存領域に二つの管理プロセスを起動すると拒否します。CLIの変更要求は、セッション・元の枝ID・head世代・全体の更新番号を照合します。自動処理側で取得したticketを固定したいときは `--at '保存したticketのJSON'` を使えます。接続先のプロセスはUnixソケットのpeer資格情報、起動時刻とboot IDで識別します。[CLIと単一ライター](branchctl.py)

切替は、出発した枝の確定保存、移動先の画像検証と復元、SQLite上のactive切替の順です。nativeが復元失敗を返した場合は元の枝を復元します。元の枝も復元できないときは論理headを保持し、物理スロットを失効扱いにします。要求の完了が不明な通信切断や管理プロセスの異常終了では、遅延した要求との衝突を避けるため、そのネイティブプロセスの世代を隔離します。この場合は最初の端末で閉じて専用エンジンを開き直すと、確定済みheadから復帰します。[枝の操作と世代照合](manager.py)

保存ファイルは既存の完全native形式です。ステージングファイルの処理済みトークン・サイズ・SHA-256を確認し、ファイルとディレクトリを同期してからSQLiteのトランザクションを確定します。枝と保存点に別のIDを割り当て、画像本体はSHA-256で共有します。利用者のデータはサーバーの `user-store/` に保存します。[画像・メタデータ・保持予算](storage.py)

専用ビルドは、nativeのストリーム最終応答で `branch_sampled_ids` を明示要求したときに、サンプルされたIDの全列を返します。UTF-8の途中で個別イベントが保留されても末尾の境界を照合できます。演算は通常版のCPU・CUDA・ggml-base・llamaのライブラリが担当します。独立環境の一般／MXしきい値は128/128、入力／生成は24/24です。[スロット形式とストリーミングの処理](native.py)・[HTTP側の変更](native-stream.patch)

## 開発用の再現

Pythonの標準ライブラリだけで回帰試験を実行できます。

```bash
python3 -B -m unittest discover -s current/experiments/state-branches -p 'test_*.py'
```

この実機でHTTPアダプターを再ビルドする際は、リポジトリのルートから次を実行します。出力先には新しいディレクトリを指定してください。配備済みサーバーのビルド記録とオブジェクトを再利用し、サーバーの翻訳単位2件をコンパイルします。

```bash
python3 scripts/prepare_source.py --profile current
python3 current/experiments/state-branches/build_adapter.py \
  --source work/current-source \
  --build /home/codex/projects/llm-lab/deepseek41-bringup-20261101/build \
  --output /tmp/xeon-branch-private-build
```

このサーバーの実験用ワークスペースはビルド済みです。別のワークスペースを用意する場合、ランチャーは `$XEON_BRANCH_WORKSPACE/build/libllama-server-impl.so` と同じ場所の `BUILD.json` を使用します。

独立試験で測定した往復、欠損時の再入力、全語彙の照合結果は実測記録にあります。[実機での検証](../../../archive/review-handoffs-not-for-readers/2026-11-28-state-branches/RESULTS_JA.md)

285トークンと約16kトークンの切替では、画像の書き込み・検証・SQLiteの確定を個別に計測しました。復元に成功した画像の最終利用時刻は、枝の切替が確定するトランザクションで記録します。[切替費用と編集したツール結果の検証](../../../archive/review-handoffs-not-for-readers/2026-11-28-branch-switch-cost/RESULTS_JA.md)
