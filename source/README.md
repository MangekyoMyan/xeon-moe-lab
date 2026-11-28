# x9ramで会話を保存し、前の発言からやり直す

同じ枝の履歴全体を対象にする `all_history` と、直近の指定トークン数を優先する `recent` を選べます。`checkpoint preview` は、対象範囲、共通部分へ戻る時間の推定、ディスク上の画像容量、履歴トークン毎秒に換算した復帰速度を表示します。計測が足りない項目には「未計測または部分計測」と表示します。

| 設定 | 実際に行うこと |
|---|---|
| `off`（既定） | 名前付き保存と枝切替時の保存を行う |
| `fixed` | テンプレートや特殊トークンを含む実入力の先頭から、N、2N、3N…を評価した時点で完全な状態を採取する。入力処理と生成の両方に適用 |
| `auto` | 復元・再入力の実測と容量から候補間隔を比較し、復帰時間の目標を満たす範囲で保持量を抑える。計測不足の間は指定した固定間隔を使用 |

## 独立CLIを起動する

x9ramの端末で実行してください。下の3例のうち1つを選びます。いずれも専用の1スロットを使い、`Branch CLI ready` が表示されたら別の端末から操作できます。最初の端末で Ctrl+C を押すと、起動前に稼働していた通常APIが復帰します。

```bash
cd /home/codex/projects/llm-lab/checkpoint-v2-20261128

# 実トークン位置を1,024ずつ区切る
sudo ./branch open --checkpoint-mode fixed --checkpoint-interval-tokens 1024 --checkpoint-budget-mib 1024
```

```bash
# 枝全体を対象に、共通部分への復帰を5秒以内にすることを目指す
sudo ./branch open --checkpoint-mode auto --checkpoint-interval-tokens 1024 --resume-target-ms 5000 --checkpoint-budget-mib 1024 --coverage-policy all_history
```

```bash
# 同じ容量で、現在の枝の末尾4,096トークンを優先する
sudo ./branch open --checkpoint-mode auto --checkpoint-interval-tokens 1024 --resume-target-ms 5000 --checkpoint-budget-mib 1024 --coverage-policy recent --recent-window-tokens 4096
```

二つ目の端末でも同じディレクトリに移り、以下を実行します。

```bash
cd /home/codex/projects/llm-lab/checkpoint-v2-20261128
./branch checkpoint preview --history-tokens 8192
./branch --json checkpoint preview --history-tokens 8192 --mode auto --coverage recent --recent 4096 --target-ms 5000
./branch turn --text '倉庫Aの棚番号は731です。'
./branch checkpoint status
./branch --json checkpoint status
```

`--history-tokens` は想定する履歴全長です。省略すると現在の長さを使います。現在の履歴上で、例えば17トークン目に戻る経路を調べるには `./branch checkpoint preview --return-position 17` を実行します。その位置で入力も終了する場合は `--empty-suffix` を付けます。プレビューは推論、保存、削除、設定変更を伴いません。

| 保持方針 | 評価する範囲 | 容量が厳しいとき |
|---|---|---|
| `all_history` | 選んだ枝の先頭から現在地まで | 最も長い再入力区間を広げにくい点を残す |
| `recent` | 現在地からWトークン前まで。左端に戻るための、それ以前の保存点も考慮 | 指定範囲の再入力を優先。窓外は目標の対象外 |

`currently_available` は現在の枝の実トークン列に合い、画像の存在とサイズを確認できた位置です。実際の復元時にSHA-256と画像内のID列を検証します。`after_plan_forecast` は今の位置から指定の終点まで進んだ場合の試算です。途中で有効化したとき、既に通過した位置には穴が残ります。別の内容に編集すると、その入力と一致する点で再計画します。

## 枝と保存点

```bash
./branch mark 共通 --pin
./branch fork A --from 共通
./branch checkout A
./branch turn --text 'この枝の検証番号は284です。'
./branch checkout main
./branch inspect snapshot 共通
./branch list
```

`turn --messages 会話.json` は、役割と本文を持つメッセージ配列をテンプレートに通して編集後の入力を作ります。途中の画像は復帰用の内部状態です。回答・新しい世代・末尾の未評価トークンは、ターンが完了した時点で確定します。途中画像から既知IDを再入力して新しい要求を始められます。中断時点の乱数、サンプラー、文法状態から同じランダム生成を再開する機能は対象外です。

`mark`、`fork`、`checkout`、`pin`、`unpin`、`release`、`inspect`、`recover` は引き続き利用できます。新しい名前と保存点の対応は不変です。同じ名前の画像が損傷した場合は新しい名前を使います。画像を外しても確定済みのトークン列と会話は保持します。操作中にネイティブ応答が途絶え、`uncertain` が出た場合は専用CLIを一度閉じ、同じ構成で開き直します。新しいプロセスで確定済みの枝を復元します。

## 見積もりの読み方

- 復帰時間は、保存候補の選択、画像検証、実機への復元、共通部分の再入力、その区間内の同期保存が済むまでです。その後に変わった入力を処理する時間、最初の本文までの時間、モデルロードは別に測ります。物理スロットに共通部分が既にあれば、そのまま進めます。
- 復元した位置で入力がちょうど終わるとき、状態画像には最後のlogitsがありません。一つ前までの利用可能な候補から既知IDを再評価し、当該位置の最初のlogitsを得ます。receiptの `old_replayed` で実数を確認できます。
- `target_met` は復帰時間に対する `yes / no / unknown` です。新しい保存に使える容量は別項目です。例えばpinで予算を使い切っていても、既存画像から目標時間で戻れる位置があります。画像が欠損すれば、実際の経路は他の一致点または先頭へ戻ります。
- 換算速度の分子は復帰した位置までの履歴長pです。最大待ち時間と最低換算速度にはそれぞれのpと使用点cを記録します。pが0、時間が0または不明なら対応する速度は空欄です。多数の位置を間引いた場合は、評価点での最大値と範囲全体の不明を別々に表示します。
- `auto` は同一ビルド・実行設定における直近192件の観測を使います。文脈長、再入力長、実機のオフロードしきい値による区分の近い2～6件から推定します。復元と再入力の合計を比較できるときは速い経路を、根拠が足りないときは最も近い一致点を選びます。候補間隔は基準の1/4〜16倍です。大きな範囲で全点列挙を省いたときは、評価した位置で間隔を選び、範囲全体の達成は未確認と表示します。将来の画像サイズは近い文脈長で観測した最大3件のうち最大の値を使います。対応する観測がなければ容量推定を空欄にします。通常の間隔変更には2回の確認を使い、画像書込み時には毎回、正確なサイズで予算を確認します。観測は管理プロセスを閉じるとリセットされます。

**容量の単位と範囲。** 1 MiB は1,048,576バイトです。有効時のディスク画像予算は、同じストアの名前付き画像、他枝・pin・共有画像、書込み途中の1画像を含みます。共有SHAは1回計上します。画像を書き始める前にネイティブ側で完全状態の長さを確定し、管理側は予約したバイト数まで受信します。公開後は同じファイルをrenameします。SQLite・WAL、OSのページキャッシュ、ネイティブの一時RAM、VRAMはこの上限の外です。`off` では従来の確定画像予算を使います。`./branch budget バイト数` で容量を変更できます。保護中の画像が収まらない変更は拒否し、以前の上限を保ちます。

**表示を検算するための仮想例。** 履歴8,192のうち4,096〜8,192を対象に、256ずつ離れた17点を置きます。各画像32 MiB、復元250 ms、再入力1,024 tokens/sと仮定すると、最大待ちは499.0234375 ms、最低換算速度はp=4,351、c=4,096で約8,719 tokens/sです。保持は544 MiB、さらに32 MiBを同時に書く場合のピークは576 MiBです。実機の値は実測報告にあります。[推定式と再現試験](test_checkpoints.py)

## 対応する専用ビルド

x9ramではDeepSeekのテキスト用スロット1個と、同期式のローカルディスク保存を使用します。専用コーデック `XKV1` はDSV4の圧縮状態に加え、raw KVの使用中セル、物理位置、次の配置位置を記録します。通常API、旧CLI、既存の会話ストアはそれぞれ元の実行条件で使えます。今回の実機用ディレクトリには独立したストアが作られます。

```bash
# このディレクトリのLinux向け回帰試験。モデルとGPUは不要
python3 -B -m unittest discover -q
```

同じx9ramビルドを再現する場合は、空の出力ディレクトリを選び、次を順に実行します。

```bash
lab=/home/codex/projects/llm-lab
engine="$lab/deepseek41-bringup-20261101"
out="$lab/checkpoint-v2-rebuild/build"
python3 build_adapter.py --source "$engine/source" --build "$engine/build" --output "$out"
python3 build_layout.py --source "$engine/source" --build "$engine/build" --link-receipt "$lab/exploration-20261112/sparse-cache/LOADER_BUILD.json" --output "$out/physical"
```

最初のコマンドは専用HTTPアダプターを、次のコマンドはKV保存・復元を含むモデルライブラリを作ります。後者は元のリンク入力から配備済みライブラリと同じSHA-256になることを先に検証します。LinuxのCLI・ストア・プランナーはPython標準ライブラリで動きます。x9ram用の起動・保護監視とDSV4コーデックは、この実機の構成に依存します。[ネイティブでの同期保存](prefix-capture.hpp)・[KV配置の保存形式](physical-layout.patch)・[実機の比較結果](../../../archive/review-handoffs-not-for-readers/2026-11-28-prefix-checkpoints/RESULTS_JA.md)

