DeepSeek-V4.1-Flash-1017 Mix-256G 最小起動準備 / 2026-11-01

状態（2026-11-01 夜更新）：ユーザーがビルド・起動済み。稼働中モデルで合成会話のキャッシュ再利用を実測確認。
実測と制約: ../live-prefill-diagnosis-20261101/RESULTS_JA.txt と probes/ を参照。
通常の継続、全再計算、同じ質問の再送、別要求の割込みを区別した。Open WebUI固有の失効原因は未確定。
以下は将来の再ビルド・起動用手順。稼働中に実行する必要はない。長い処理は引き続きユーザーが実行する。

1. サーバーのSSHターミナルでビルド（sudo不要）
 python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/build.py

2. ビルド完了後、起動（読み込みはsystemd配下で継続）
 sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/start.py

3. 起動状態を一度確認（待機ループなし）
 sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/status.py

4. ログを見る場合
 sudo tail -f /home/codex/projects/llm-lab/deepseek41-bringup-20261101/server.stderr.log

5. 停止
 sudo systemctl stop llm-deepseek41-api.service

モデルをOS起動時に自動起動する設定にはしていない。
旧GLM用APIが稼働中なら、この起動スクリプトは停止する。旧サービスを勝手に終了しない。
必要な場合はユーザーが sudo systemctl stop llm-openai-api.service を実行してから起動する。
監視・異常通知は既存のxeon-watchサービスが担当。新APIの保護停止も通知対象に追加した。

API
 URL: http://10.10.10.2:8000/v1
 モデルID: deepseek-v4.1-flash-1017-mix256
 APIキー: 以前のGLM APIと同じもの
 コンテキスト枠: 131072（128k）
 初期モード: thinking_mode=chat、同時処理1件

Open WebUIではモデル一覧を更新して新しいモデルIDを選ぶ。
会話本文はOpen WebUI側の履歴保存を正本とする。サーバーのRAMだけでは再起動後の保存にはならない。
別チャット、自動タイトル生成、タグ生成などが同じ1枠に割り込めば、直前のKVは置き換わり得る。
継続会話のキャッシュを優先する間は、それらの自動タスクをオフにするか別モデルへ分ける。

今回の配置
 RAM: routed Expert重み、110.6GBのEngram表、現在の入力トークン列と管理情報。
 GPU: Attention等の非Expert重み、現在の1会話分のKV・圧縮状態、演算用作業領域。
 全346.4GBのモデルが入る512GB構成なので、HDDへのランダム参照を避けるためEngramもRAMに置く。
 --load-mode mmap+mlock と --lazy-mode off を使用。memlock上限は新サービス内だけ無制限。
 初回はHDDから大量に読むため長い。キャッシュによる省略対象は会話の再prefillであり、初回モデル読込ではない。
 VRAMはKVだけの専用領域ではなく、GPU演算に必要な重みと作業領域も占有する。
 既存の独自expertキャッシュ/分担プラグインは、このモデルへ移植していない。

会話キャッシュ
 --cache-prompt: 現在のslotに残るKVを再利用。前回と一致するトークン接頭部分を再計算せず、追加分を処理。
 --parallel 1 / --no-cont-batching: まず1会話ずつ処理する。
 --cache-ram 0: 他会話のKVスナップショットをRAMへ退避する機能は使わない。
 --ctx-checkpoints 0: RAMの中間状態チェックポイントとその復元を使わない。
 --no-context-shift: 対応が不完全な圧縮状態のスライドを要求しない。
 元のトークン列をRAMに持つことと、RAMへKV自体を退避することは別機能。
 APIリクエストには従来どおり履歴全体が必要。ただし送信し直すこととprefillし直すことは別。
 同じトークン接頭列、同じモデル/テンプレート/設定で、対象KVがslotに残っているときに再利用できる。
 会話の編集や分岐で一致しなくなった場合、圧縮状態が巻き戻せる範囲は限定される。
 対応範囲外ならエンジンが全状態をクリアして再prefillする。VRAMの生バッファだけを独自に消す実装はしない。

ユーザー実行のキャッシュ確認（読み込み完了後、他の会話をしていないとき）
 sudo python3 /home/codex/projects/llm-lab/deepseek41-bringup-20261101/check_cache.py

初回→継続キャッシュあり→同じ継続をキャッシュなし→元の会話へ戻る、の4リクエスト。
合言葉の回答とtimings.cache_nを記録。継続時cache_n>0を確認する。
巻き戻しでcache_n=0になることは、この初期方針では許容する。
これは機能確認であり、全ログイット同一性や回答品質を保証する試験ではない。
このcheck_cache.py自体は未実行。別の小規模probe_cache.pyで英語・日本語・stream・token ID継続を実測した。
日本語streamの同一入力98トークンでは、85トークン再利用時の初文字1.997秒、キャッシュ無効時10.885秒。出力文は一致。
これは全条件の品質保証ではなく、設定変更やエンジン改修による一般的な速度向上の主張でもない。

互換性の調査
 保持している旧エンジンはdeepseek4まで。ダウンロードしたGGUFのarchitectureはdeepseek41。
 対応済み上流masterを独立ディレクトリへ取得してビルドする。
 固定リビジョン: dde879bc4f954b809892fd0d660515f42536394a
 CUDA12.9.2、sm_70、Ivy Bridge AVX/F16C、AVX2/FMAなしを明示。
 CPUのweight repackを無効化。FP16 KVを使い、初期段階の量子化KVの追加変数を避ける。
 配布者の古いPR #30817/b72050d03起動例は採用しない。対応はPR #31086でmasterへ統合済み。
 キャッシュ復元の問題 #30944 と修正候補 #31292 は未解決報告がある。
 今回は標準版の1slot継続再利用を使い、checkpoint復元と並列会話の経路を避ける構成。
 これで全不具合が解決するという証明はしていない。今回の実測はcheckpoint復元を含まない。

根拠・記録
 https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF
 https://github.com/ggml-org/llama.cpp/pull/31086
 https://github.com/ggml-org/llama.cpp/issues/30944
 https://github.com/ggml-org/llama.cpp/pull/31292
 research/upstream: 関連ファイルだけを取得した読取用資料。ビルド用sourceとは別。
 gguf-metadata.json: 実GGUFの先頭約5.8MBから取得したメタデータ。重みはロードしていない。
 server-config.json / build-commands.json: 正確な起動・ビルド条件。
