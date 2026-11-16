直接 SSE 到着記録器 — 次回の「2 個ずつ表示」の切り分け用
2026-11-16 / stdlib のみ / 実 API・モデル未実行

1. このファイルの目的

record_sse_arrivals.py は、利用者が明示した URL に、明示した JSON を使って
stream=true の POST を 1 回送る、小さい診断用クライアントです。
HTTP body を read1(最大 4096 bytes) で受け取った時刻と、SSE event が完成した
時刻を別々に保存します。再試行、redirect 追従、サービス起動、設定変更はしません。

今回のレビューでは fake byte streams、fake clock、fake socket 上の本物の
http.client.HTTPResponse だけで検証しています。外部 API、実モデル、loopback
サーバーには接続していません。このツールの生成性能の実測値はありません。

この直接 HTTP リクエストは、Open WebUI の画面専用の処理経路を通りません。
Open WebUI の UI 専用 batching が原因かどうかを、この記録だけで確定することは
できません。backend の emit 時刻と、実際の browser の受信・表示時刻が別途必要です。
URL を Open WebUI の公開 API に向けても、画面の会話送信経路と同じとは限りません。

2. 実行例 — 利用者が実際に生成を要求するときだけ

Python 3.9 以降を想定し、Python 3.12.14 で合成試験済みです。追加 package は不要です。
以下の URL と port は例であり、現在の稼働構成を検出・推測した値ではありません。

native /completion 用の request.json の例:

  {
    "prompt": "1 から順に数えてください。",
    "n_predict": 32,
    "temperature": 0,
    "seed": 123,
    "stream": true
  }

  python3 record_sse_arrivals.py \
    --url 'http://127.0.0.1:8080/completion' \
    --request-json ./request.json \
    --output ./arrival-native-01.json

OpenAI 互換 /v1/chat/completions 用の request.json の例:

  {
    "model": "実際のモデルIDへ置換",
    "messages": [{"role": "user", "content": "1 から順に数えてください。"}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }

  python3 record_sse_arrivals.py \
    --url 'http://127.0.0.1:8080/v1/chat/completions' \
    --request-json ./request.json \
    --output ./arrival-openai-01.json

stream_options などの受理範囲は接続先に合わせてください。自動で別の body に変えて
再試行する機能はありません。stream だけは必ず true にし、変更の有無を記録します。
それ以外の生成条件は利用者の JSON を保持します。1 request / 1 generation を主な
対象とし、並列リクエストや native の複数 prompt を束ねる測定には使わないでください。
native event の index/id_slot が変わった場合は、複数 stream 非対応のエラーを記録し、
異なる stream の counter/text を集計に混ぜません。

認証が必要なら、任意の環境変数 XEON_REVIEW_API_KEY を設定します。値は Bearer
header にだけ使用し、引数・結果・エラーへ保存しません。URL の userinfo は拒否します。

--url / --request-json / --output はすべて必須です。出力の親ディレクトリは事前に
用意してください。出力を新規作成できることを確認してから送信し、既存ファイルを
上書きしません。新しい出力名で再実行する操作は、新しい生成要求を 1 回送ります。
自動 proxy 環境変数は使用せず、指定 host/port へ直接接続します。HTTPS は標準の
証明書検証を使います。

3. 保存する時刻

time.monotonic_ns を使用します。origin_monotonic_ns を起点として、stages_ns は
すべて offset ns です。body_reads と sse_events には絶対 monotonic ns と offset ns
を両方保存します。同じ記録内の差分に使用し、別マシンの時計とそのまま比較しません。

request_started_offset_ns:
  connection 作成前。DNS、接続、TLS、送信、待機を含む経過時間の起点。
request_sent_offset_ns:
  HTTPConnection.request が戻った時点。モデル受付・推論開始時刻ではありません。
response_headers_completed_offset_ns:
  getresponse が status/header の解析を終えた時点。真の最初の HTTP byte の時刻
  (wire TTFB) は http.client のこの API からは測定できません。この値は header
  完了の proxy です。header 用の内部 buffer が body を先読みする場合もあります。
first_body_observed_offset_ns:
  最初の非空 read1 が戻った時刻。HTTP transfer decoding 後の body の観測です。
first_content_text_observed_offset_ns:
  通常の content が初めて非空になった SSE event の完成時刻。
first_reasoning_text_observed_offset_ns:
  reasoning / reasoning_content が初めて非空になった event の完成時刻。
application_completion_offset_ns:
  最初の [DONE]、または native の stop:true を含む event が完成した時刻。
response_eof_offset_ns:
  HTTP body の終端を観測した時刻。chunked なら終端 chunk まで確認します。
capture_finished_offset_ns:
  後処理を終えた時刻。失敗時にも残します。

first_content は表示の proxy であり、本物の browser 描画時刻ではありません。
reasoning を画面に出す構成では利用者が先にそれを見る場合があります。そのため
通常 content と reasoning を一つの「first visible」に混ぜていません。

read1 は「最大 1 回の下層 read、またはすでにある buffer」を返す API ですが、
HTTP chunked の header 処理には追加の read があり得ます。1 read は TCP packet、
1 HTTP chunk、1 推論 token のいずれとも同一と見なしません。read1 上限、TLS、
socket buffer、client 自身の JSON 解析時間でも観測される分割が変わります。

4. 主要な JSON 項目と読み方

body_reads:
  各 read の時刻、byte 数、body 内の offset、SHA256。本文は保存しません。
sse_events:
  event の完成時刻、body_read_index、種別、data/content の文字数と hash。
  CRLF / CR / LF、分割 UTF-8、複数 data 行、コメント、同じ read 内の複数 event
  を処理します。id / retry は記録対象にせず、再接続もしません。
summary.body_read_intervals / sse_completion_intervals:
  隣接 read / 隣接 event の到着間隔の p50 / p95 / 最大値 (ms)。
summary.content_event_intervals / reasoning_event_intervals:
  その種類の非空 text を含む、隣接 SSE event 間の間隔。同一 event に複数 choices
  や reasoning 欄があっても、その event の時刻は各種類につき 1 回だけ数えます。
summary.body_reads_with_multiple_sse_events / multiple_content_events:
  1 回の body read に複数 event が含まれた回数。
summary.largest_sse_events_in_one_body_read:
  1 回の read に含まれた event 数の最大値。
summary.native_cumulative_observations:
  native の tokens_predicted を event ごとに記録し、前回値との差分を示します。
summary.native_cumulative_jumps_over_one:
  その差分が 2 以上になった観測。内部で本当に同時に 2 token 生成された証明では
  ありません。server の送信間隔、UTF-8 の保留、空文字 token などでも見え方が
  変わり得るため、必要なら server 側の token completion / emit 時刻を照合します。
summary.native_final_timings / openai_final_usage:
  server が報告した最終値。native の各 event の timings も保存します。
  tokens_predicted と timings.predicted_n は別々に記録し、差も残します。
  初 token の算入などは server 実装で確認し、ここでは一致を強制しません。

p50/p95 は、間隔の値を昇順に並べて (n-1)*p の位置で線形補間します。間隔が
0 個なら null です。これは小標本の記述統計で、性能保証や信頼区間ではありません。

SSE event 数、content の文字数、event の tokens 配列長を「生成 token 数」と
呼ばないでください。最終 usage/counter も server の報告であり、各 token の
内部完了時刻を復元しません。native の tokens 配列は要素数だけ保存し、合算して
総生成数にはしません。OpenAI は content、reasoning、tool_calls、usage を分け、
tool の id/name/arguments も長さと hash だけ記録します。OpenAI の複数 choices
では content の全体 hash は到着順に交互連結したものなので、回答 1 個の hash と
解釈しないでください。通常は n=1 にします。

5. 失敗と上限

success=true には、対応する生成 event、アプリケーション終端 marker、HTTP body
EOF、完全な SSE frame が必要です。非 JSON の data、server error、型不正、HTTP
失敗、読み取り timeout、途中切断、非対応の Content-Type / Content-Encoding、
サイズ・件数・時間の上限到達は success=false になり、errors に理由を残します。
[DONE] が見えた後でも HTTP chunked が途中で切れれば失敗です。終端のない SSE を
正常生成完了と見なしたり、finish_reason だけで [DONE] を補ったりはしません。

comments のみの heartbeat は受理します。data に独自の非 JSON heartbeat を入れる
接続先は、この診断器では非対応として失敗扱いです。native stop:true 後の別 JSON
event もエラーになります。長時間接続を閉じない独自 endpoint は対象外です。

標準上限:
  request file: 1 MiB
  response body: 8 MiB (--max-body-mib で 1～64 MiB)
  SSE event/line: 256 KiB
  SSE event 数: 4096
  非空 body read 数: 32768
  socket timeout: 30 秒 (--timeout-seconds)
  capture 経過時間: 300 秒 (--max-seconds)

経過時間の確認は I/O の前後で行い、socket timeout も残り時間に合わせます。
標準 resolver の DNS 解決などを強制終了する process watchdog ではありません。
接続を閉じても、遠隔 server の生成が直ちに停止する保証はありません。

exit code は、正常完了 0、失敗 2。途中切断等も、可能な範囲の metadata を新規
JSON に保存します。入力不正や新規出力作成失敗なら送信しません。disk full などで
出力自体を完了できない場合は成功にせず、部分 JSON が残る可能性があります。

raw prompt、response text、reasoning、tool arguments、API key、URL query 値を
出力しません。request body は byte 数と SHA256、response は文字数/hash/種別です。
endpoint の host/path は保存します。hash は匿名化や暗号化ではなく、同一性照合用
です。元の request.json は利用者が作ったファイルとして別に残ります。

6. 次回の切り分け方

まず同じ prompt と生成条件で native または OpenAI 互換 API を直接観測します。
各比較の入力 JSON、実行時 library/config、cache 状態、並行負荷を別途保存します。
短い試行を先に確認し、長さを増やす場合は上限に達していないことを確認します。

・body_read_index が同じ 2 content event は、client が一度に観測したことを示す。
  生成が二段ずつだったのか、server/ネットワーク/client に滞留したのかは別問題。
・直接 API では滑らか、Open WebUI 画面では 2 個ずつなら、次に UI 経路の emit、
  browser 受信、描画の各 timestamp を対応させる。直接 API 結果だけで原因確定はしない。
・native counter が 2 ずつ増えていても、その event 内に文字/token がどう積まれたか
  を確認し、server 側の token 完了と送信を分けて測る。
・平均 tok/s のみで判断せず、request 全体、first content、間隔の p95/最大値、
  同一 read での event grouping を一緒に見る。

7. 合成試験の再実行

  python3 -m unittest -v test_sse_recorder

このコマンドは生成要求を送信しません。socket.create_connection を失敗させる
guard を置き、fake socket の byte 列だけを HTTPResponse に渡しています。
SSE_TEST_RESULTS.json に今回の試験結果と実装・テストの SHA256 を記録しています。
