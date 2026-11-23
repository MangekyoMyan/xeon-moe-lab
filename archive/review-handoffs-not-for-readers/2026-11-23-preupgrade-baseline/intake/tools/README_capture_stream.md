# SSE受信の交換前・交換後比較

`capture_stream.py` は、明示実行したときだけ指定APIへ1回POSTする、Python標準ライブラリのみの観測クライアントです。Python 3.9以降で使用します。import・`--help`・テスト実行によって実機LLMへ接続することはありません。テストは127.0.0.1の疑似HTTPサーバーだけを使用します。

目的は「数文字ずつ届いて間が空く」ように見える現象について、**このクライアントが本文を受け取った時刻とSSEイベントが完成した時刻**を残すことです。OpenWebUI画面の描画時刻や、モデル内部の1トークン生成時刻を直接測るものではありません。

## 実行

エンジンPCで、実際に使うモデル名・固定の非機密prompt・生成条件を指定したJSONファイルを用意します。`--request` はJSON文字列ではなく**ファイルへのパス**です。以下は形を示す例です。

```json
{
  "model": "REPLACE_WITH_THE_ACTUAL_SERVED_MODEL_NAME",
  "messages": [
    {"role": "user", "content": "次の語を使って短い日本語の文を3つ書いてください。猫、窓、朝。"}
  ],
  "temperature": 0,
  "max_tokens": 128,
  "n": 1,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

`stream: true` と整数の `n: 1` は明示必須です。`stream_options.include_usage` は使用するエンジンが受け付ける場合に指定します。クライアントはpayloadを補正せず、元のバイト列をそのまま送信します。実際に使うJSONを確定したらCPU交換前後で同じファイルを使ってください。

```bash
python tools/capture_stream.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --request requests/stream.json \
  --out runs/pre-direct-01 \
  --label pre-direct-01
```

ポートと完全なAPIパスは実機の設定に合わせます。URLは `/chat/completions` で終わるものを明示します。認証が必要な場合は、すでに設定した環境変数の名前を `--api-key-env ENV_NAME` で渡してください。送信Authorizationヘッダーは記録しません。指定JSONと応答本文は生保存し、応答側に反射された値も改変しません。URLのuser情報・query・fragmentは受け付けません。prompt、応答本文、SSEのJSONは保存するため、固定の非機密データを使います。

交換後も同じクライアント・JSON・URL・観測位置で実行し、`--out` と `--label` を `post-direct-01` などに変えます。既存の出力ディレクトリは上書きしません。

`http.client` で指定ホストへ接続するため、`HTTP_PROXY`・`HTTPS_PROXY`・`ALL_PROXY` と小文字版の環境変数は使用しません。変数が存在する場合は名前だけをmetadataに記録します。URL自体がOpenWebUIや別のproxyを指す場合、そのproxy経由の観測になります。**エンジンへの直結とOpenWebUI/proxy経由は別の観測位置**として、別label・別出力で比較します。127.0.0.1はこのスクリプトを実行するPC自身です。

## 上限と終了状態

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `--timeout` | 60秒 | 各socket I/Oの待ち時間上限 |
| `--max-seconds` | 300秒 | 経過時間budget。read前等に残り時間を確認し、socket timeoutも残り時間以内に設定 |
| `--max-body-bytes` | 16 MiB | 保存するHTTP本文の上限。上限に到達した時点で保守的にpartialとして停止 |

DNS解決はOSの管理下にあり、`max-seconds` はDNSまで強制中断する厳密なwall-clock watchdogではありません。タイムアウト・本文上限・通常のCtrl-Cでも、受信済みのraw本文、イベント、summaryを残します。通常のファイルbufferに書き、readごとの明示flushは行いません。通常の終了処理でファイルを閉じてflushするため、部分保存はこの終了処理を通る場合に行われます。自動再試行・再接続・redirect追従はありません。

DONEを見てもHTTP本文のEOFまで読み続けます。DONEとEOFは別々に記録するため、DONEの後にHTTP本文が終了せずタイムアウトした場合も、その状態を残します。

| 終了コード | `status` | 意味 |
|---:|---|---|
| 0 | `complete` | 本文EOF・DONE・有効なcompletion_tokensを含むusage・finish reasonがあり、解析で問題を検出しなかった |
| 3 | `partial` | 欠損、未完成SSE、タイムアウト、上限等。受信済みデータは利用可能 |
| 1 | `error` | HTTP/APIエラー、受信前の通信失敗、保存失敗等 |
| 2 | — | URL・request・既存出力等の入力エラー。APIへ送る前に拒否 |
| 130 | `partial`等 | Ctrl-Cによる中断 |

usageを送らないエンジンでは、正常に文章を返してDONEが来ても `missing_usage` によりpartialになります。usageがobjectでも `completion_tokens` の存在と値を別途検査し、bool以外の非負整数だけを有効とします。項目がない場合は `missing_completion_tokens`、null・負値・bool等の場合は `invalid_completion_tokens` によりpartialとし、raw本文と報告されたusageはそのまま残します。これは実生成数を確認できない**観測記録の状態**で、生成失敗だと断定するものではありません。usage・任意のtoken内訳・finish reason・EOFは作り足しません。解析対象外のreasoning形式などが来た場合もraw/JSONを残し、問題をsummaryに記録します。

nonnullのfinish reasonは元の値を記録します。文字列でない値や空白だけの文字列は `invalid_finish_reason` としてpartialにします。

## 保存内容

| ファイル | 内容 |
|---|---|
| `metadata.json` | URL、観測label、Python版、clientとrequestのSHA-256、上限、変更していないpayloadの検査結果 |
| `request.requested.json` / `request.effective.json` | 同一の入力バイト列。送信payloadを記録 |
| `response_body.bin` | 受信したHTTP本文をそのまま保存。HTTP chunk framing等は `http.client` が除去しているためpacket captureではない |
| `reads.jsonl` | 各 `read1()` の開始・戻り時刻、戻ったバイト数、raw本文内offset。EOFの0-byte returnとread errorも記録 |
| `events.jsonl` | 各空行で完成したSSE blockの受信時刻、read番号、元の行、data、解析JSON。コメントだけ/空のblockも区別して記録 |
| `content.txt` | 非空の `delta.content` を順に連結 |
| `reasoning_content.txt` | `delta.reasoning_content` を順に連結 |
| `reasoning.txt` | `delta.reasoning` を順に連結 |
| `summary.json` | 最初/最後の本文・reasoning、headers ready、DONE、EOF、usage、finish reason、終了状態、イベント間隔 |

応答ヘッダーはcontent type/length/encodingとtransfer encodingだけを保存します。任意の例外メッセージは保存せず、例外の種類と発生phaseを記録します。標準出力は短いJSON summaryだけです。

## 時刻と解釈

`request_start` はPOST送信直前の `perf_counter_ns()`、`response_headers_ready` はHTTPヘッダー取得直後の時刻です。各受信時刻は `HTTPResponse.read1()` が戻った直後に採ります。**TCP/socketのread・HTTPのchunk・SSEイベント・モデルtokenは別の単位**です。

`read(65536)` の満杯待ちを避け、`read1(<=65536)` を使用します。すでにPython側でbuffer済みの本文が返る場合もあり、HTTP chunked転送の制御情報を読むための内部readもありえます。したがって、この記録からTCP packet数や正確なsocket-read回数を逆算しません。ファイル書込み・解析による後続readへの影響もありえます。

SSEイベントは区切りが完成したreadの時刻に対応付けます。UTF-8文字やCRLFがreadをまたいでも組み立て、EOF時の未完成イベントを完成扱いにしません。同じreadで複数イベントが完成した場合、それらは**同一timestamp、間隔0 ms**になります。

- `ttft_content_receive_ms`: request開始から最初の非空 `content` 到着まで。role-only/空文字deltaは含めません。
- `first_reasoning_receive_ms`: 最初の非空 `reasoning_content` または `reasoning` の到着まで。contentとは別です。
- `sse_data_event_gap_ms`: data付きSSEイベント間隔。空data、usage-only、DONEも含みます。コメントだけのblockは含めません。
- `sse_content_event_gap_ms` / `sse_reasoning_event_gap_ms`: その種類の非空textを含むイベント同士の間隔です。

間隔はmedian・p95・max・0 ms件数を記録します。p95はnearest-rankの `ceil(0.95*N)` 番目です。対象イベントが2つ未満なら間隔はnullです。**これらはtoken TPOTではありません。** `server_usage` はサーバーが返したusageをそのまま残し、イベント数や文字数をtoken数に換算しません。

内部エンジンのtoken-IDベンチマークとは観測位置も時計も異なります。このfixtureテストの時間を、旧CPU/新CPUや実際のエンジンの速度として使わないでください。

## 最小限の検証

kitディレクトリから実行します。使用するのは一時ディレクトリと127.0.0.1に立てる疑似HTTPサーバーのみです。

```bash
python -m unittest discover -s tests -p test_stream.py -v
```

検証するのは、任意byte境界・UTF-8・CR/LF/CRLF・空/複数行イベント・EOF末尾、raw保存、reasoning分離、usageとcompletion_tokensの欠損/null/負値/bool、最初の短いreadを満杯待ちしないこと、timeout時の部分保存、本文上限、HTTP本文の途中切断、redirect非追従、送信Authorizationヘッダー非記録、送信前検査です。認証のfixtureは応答に認証値を反射せず、送信ヘッダーを記録していないことだけを確認します。

## 一次資料

- [Python http.client](https://docs.python.org/3/library/http.client.html): HTTPConnection/HTTPResponse、timeout、TLSの標準動作。
- [Python io.BufferedIOBase.read1](https://docs.python.org/3/library/io.html#io.BufferedIOBase.read1): buffered read1の性質。
- [CPython HTTPResponse実装](https://github.com/python/cpython/blob/3.12/Lib/http/client.py): `read1` とchunked転送用 `_read1_chunked` の区別。fixture検証環境はPython 3.12.14。
- [WHATWG Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream): UTF-8/BOM、CR/LF/CRLF、data結合、空行dispatchとEOF時の未完成イベントの扱い。

このツールはSSEの本文framingを用いますが、ブラウザーEventSourceの自動再接続・redirect動作は実装しません。
