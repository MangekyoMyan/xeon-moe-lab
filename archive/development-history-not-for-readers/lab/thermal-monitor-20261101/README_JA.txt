温度監視と異常時だけのCodex通知 — 2026-11-01

通常監視はローカルのPythonだけで行う。モデル/API呼び出し・トークン消費は発生しない。
異常時にだけcodex queueで既存会話へ短い通知を送る。その通知をAIが処理するときはトークンを使う。

稼働するサービス
  xeon-watch.service          root: センサー読取とローカル記録
  xeon-watch-notify.service   codex: 異常イベントの配送と配送記録
両方とも起動済み・OS起動時に自動起動。モデル起動、再起動、ファン変更、電力変更は行わない。
既存llm-openai-api.serviceの過熱/ECC保護機能は維持。APIが保護停止した場合も通知する。
監視ソフト自体は通知専用であり、手動起動した任意のモデルを自動停止する機能ではない。

読取間隔
  CPU/GPU/HBM/EDAC等: 10秒
  BMCのDIMM/PCH/周辺部/RPM/ファン出力: 30秒（非同期取得）
  通常履歴: 60秒。5MiB×最大4世代にローテーション。

通知上限（これまでの運用上限を使用。DIMM70℃は今回設定した運用上の通知目安）
  CPU60℃ / GPU60℃ / HBM65℃ / DIMM70℃ / PCH55℃ / 周辺部45℃
  ファン出力70%超、Standard以外のモード、使用中ファン停止
  ECC/CPU熱制限/PCIe replayのカウンター増加
これらはメーカー保証上限の説明ではない。config.jsonで変更できる。

通知量の制御
  同じ温度異常が続いても通知は1回。上限から3℃下がった状態を3回確認すると回復扱い。
  同じ項目の再通知には30分のクールダウン。新しい異常は別項目として保存する。
  センサーの連続取得失敗も検出。常駐監視の記録が120秒以上古い場合は通知係が検出する。
  通知はまとめて最大5分に1回。未送信分はローカル保管し、接続回復後にまとめて配送。
  正常復帰だけのAI呼び出しはしない。
  Codexのローカルデーモンが接続可能なときだけキューへ送る。停止中は起動させず保管する。
  この会話が作業中の場合はキューで待つ。キュー受理と実際のAI処理/画面通知は別である。
  アプリやネットワーク・利用枠の状況によってAI側の処理は遅れる。既存APIの保護停止はその返答を待たない。
  配送直後のプロセス異常などで受理確認を保存できない場合、再送が重複する可能性は残る。

状態を一度確認（AI不要）
  python3 /home/codex/projects/llm-lab/thermal-monitor-20261101/status.py

サービスを確認
  systemctl status xeon-watch.service xeon-watch-notify.service --no-pager

設定を変えた後の再起動（監視だけの軽量処理）
  sudo systemctl restart xeon-watch.service xeon-watch-notify.service

監視を止める場合は両方を止める
  sudo systemctl disable --now xeon-watch-notify.service xeon-watch.service

記録
  /var/lib/xeon-watch/latest.json             最新値、センサー取得失敗、継続アラーム
  /var/lib/xeon-watch/history.jsonl           間引いた履歴
  /var/lib/xeon-watch/detector.json           アラームの持続状態
  /var/lib/xeon-watch/outbox/                 通知イベント
  /var/lib/xeon-watch-notify/last-delivery.json キュー受理結果
  /var/lib/xeon-watch-notify/state.json         配送済みID

検証
  疑似データ14ケース: 温度境界、持続、再起動時の重複、ヒステリシス、欠落センサー、ECC増加、
  API保護停止、Codex停止時の保管/再送、まとめ送り、監視停止検出、起動時猶予。
  実機を加熱するテストはしていない。実センサーの単回確認は全16DIMMを取得し異常0。
  通知経路は明示的なSELFTESTで確認。SELFTESTは実際の温度超過ではない。

ユーザー運用方針
  長いダウンロード/モデル起動/プリロードはユーザーがコマンドを実行する。
  AIは通常監視や進捗待ちのために毎分会話を継続しない。
  rootのAGENTS.mdにも保存済み。過去の中断した比較試験は再開しない。

公式の通知コマンド資料
  https://learn.chatgpt.com/docs/developer-commands#codex-queue
