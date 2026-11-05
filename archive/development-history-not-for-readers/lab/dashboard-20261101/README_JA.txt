表示専用のライブ監視コマンド / 2026-11-01

起動やダウンロード用のターミナルとは別のSSHターミナルで開く:
 sudo python3 /home/codex/projects/llm-lab/xeon_dashboard.py

画面を2秒ごとに更新。Ctrl+Cで表示だけ終了する。
モデル、ダウンロード、常駐温度監視は継続する。表示を使うためのモデル再起動や追加インストールは不要。
幅110列程度・高さ30行以上にすると全項目を見やすい。狭い画面でもアラーム行を残す。

表示項目
 CPU/GPU/HBM/DIMM温度、PCH/周辺温度、ファン出力/回転数
 CPU/GPU使用率、GPU電力、ECC状態、VRAM使用量
 RAM利用可能量、モデルRSSと固定メモリ、HDD空き/読み書き速度、通信速度
 DRAM読み込み/書き込み/合計帯域（GB/s）、ソケットS0/S1の内訳
 GPUの直近30秒の採取値の平均/最大（既存API監視の記録がある場合）
 API起動状態、PID、起動経過、設定されたexpert配置、context
 処理中の入力進捗、生成済みtoken、会話cache再利用数、生成速度
 前回完了した入力処理/生成のtok/sと計測からの経過時間
 モデルダウンロードの容量、転送%、速度、残り時間目安、完了ファイル数、SHA256検証の進捗

更新の意味
 画面の更新は2秒。温度はxeon-watchが既に取得したデータを再表示するためCPU/GPUは約10秒、BMC/DIMM/ファンは約30秒間隔。
 センサー取得からの経過時間を表示し、古いデータを現在の測定値として扱わない。
 初回だけは差分がないのでCPU使用率や転送速度は--になる。次の更新から表示。
 APIはGET /health とGET /slotsだけを約5秒間隔で読む。応答が遅い場合は15秒に間隔を広げる。生成リクエスト、/metricsリセット、モデル起動操作は送らない。
 推論速度は既存ログの数値行とslotのtoken増分から得る。待機中の値は「前回完了」と表示し、現在の速度と混同しない。
 起動中はAPIの準備状態を表示する。APIから進捗率が提供されない場合、推測で起動%を作らない。HDD読み出し速度やモデルRSSも参考にできる。
 CPU/GPU/NUMA配置の表示は起動設定から読むもの。各tensorの実配置をこの画面だけで保証しない。

ダウンロード
 既定は現在のDeepSeek Mix-256G。manifestの予定容量と既存ファイル/.partの容量を比較する。大きな重み本体は読まない。
 転送100%とSHA256検証完了は別表示。検証済みmanifestが一致する場合のみ「検証記録あり」とする。ここで再hashはしない。
 今回、既存のdownload_deepseek41_orca_mix256.py / download_glm_air_q8.pyへ小さな進捗JSON出力を追加した。
 次回それらを実行するとdownload-progress.jsonに段階と検証率を記録する。転送速度は.partの増分、SHA検証率はその記録から表示する。
 ダウンロード、認証、再開、SHA照合、検証後renameの処理は維持。進捗JSONの書込み失敗だけでダウンロードを失敗させない。
 スクリプトを別フォルダーへコピーする場合は、同じフォルダーへxeon_download_progress.pyもコピーする。
 この容量ベースの転送率は、既存の順次書込みcurl/.part方式向け。HF/Xetや領域予約を行う任意のダウンローダーの正確な進捗を保証しない。疎ファイルや.aria2があれば割合を未確定にする。
 検証記録ありは過去の検証記録とサイズが一致する意味で、表示中にファイル全体を再検証しているわけではない。

GLMの表示に切り替える例:
 sudo python3 /home/codex/projects/llm-lab/xeon_dashboard.py --download-manifest /home/codex/projects/llm-lab/models/ggml-org--GLM-4.5-Air-GGUF/download-verified.json --download-dir /home/codex/projects/llm-lab/models/ggml-org--GLM-4.5-Air-GGUF

一度だけ表示:
 sudo python3 /home/codex/projects/llm-lab/xeon_dashboard.py --once

APIへ問い合わせず、ローカル情報だけで表示（IMCカウンターは使用）:
 sudo python3 /home/codex/projects/llm-lab/xeon_dashboard.py --no-api

既存の記録だけを読む場合は --no-api --no-imc を併用する。

2026-11-02追加: メモリ帯域の見方
 通常のコマンドを再実行すれば、DRAM欄を表示する。モデルや常駐サービスの再起動は不要。
 Linux perfが公開するuncore_imc_0〜3の各ソケット代表CPUで、CAS読み込み/書き込みを計16カウンター計測する。
 この実機のsysfs定義は1count=64bytes。一定時間の差分からGB/s（10^9bytes/s）へ換算する。
 S0/S1はそれぞれのCPUに接続されたメモリ側の通信量。他方のCPUやDMAからのアクセスも含み、実行したCPUの帰属を示さない。
 モデル専用の値ではなく、OS・他プロセス・転送を含むシステム全体の実測値。RAM容量やHDD速度の推定値ではない。
 初回の連続表示は差分待ち。--once/--jsonでは約1秒待って帯域を測る。
 カウンター権限不足や未対応の場合は--と表示する。perf_event_paranoid等の設定は変更しない。
 他のperf計測とカウンターを共有する場合は、各イベントの区間time_enabled/time_runningで補正し、多重化時の稼働率を表示する。低い稼働率の推定値は慎重に読む。
 --no-imcでこの計測だけ無効化できる。終了するとFDを閉じ、カウンターを解放する。デーモンの追加やモデル起動はしない。
 ダッシュボードを別フォルダーへコピーするときは、xeon_imc.pyも同じフォルダーへ置く。

GPUとボトルネックの見方
 現行gpu2はexpertをGPU末尾2層/CPU38層へ固定配置。動的expert cacheは使っていない。VRAM使用18.66GiBは2層の重み、通常の会話KV、演算作業領域などの合計。
 GPU使用率はcache hit率ではない。会話KVの再利用は別の機能であり、128kの通常コンテキストを維持している。
 従来のGPU使用率は約10秒間隔の取得値。GPU窓には既存API監視の約2秒間隔の値を使い、直近30秒のサンプル平均/最大/件数を表示する。
 GPU窓は連続測定による正確な30秒稼働時間ではなく、取得値の集計。短いGPU動作をすべて捕捉する保証はない。稼働中APIの監視記録がなければ表示しない。
 CPU使用率が高くても、CPU演算だけでなくメモリ待ちやbusy waitを含み得る。帯域が低いだけでCPU演算不足とは断定できず、メモリ遅延・NUMA偏り・同期も候補になる。
 以前の512GB構成の連続読み込みテストは約66〜70GB/sだったが、負荷や数え方が現在の推論とは異なるので「利用率」の分母には使わない。
 まずprefill中/生成中/待機中の帯域とソケット差を見比べる。自動でCPU/RAMどちらが限界かと判定する機能は付けていない。

根拠:
 https://man7.org/linux/man-pages/man2/perf_event_open.2.html
 実機 /sys/bus/event_source/devices/uncore_imc_*/events/cas_count_{read,write} と対応scale/unit。
 従来GPU取得仕様: https://docs.nvidia.com/deploy/nvidia-smi/index.html
 過去の別負荷データ: ../results/ram-upgrade-20261031/local/read-interleave-1/stdout.txt

その他: --interval 5 で画面更新を5秒、--duration 60 で表示を60秒後に終了、--jsonで一度だけ数値スナップショットを出力。
sudoは既存のAPIキー/health/logや/procの読取りに使う。サービス・設定・ファン・電力・ネットワークを変更するコマンドは含まない。
会話本文やAPIキーは画面・JSONへ出さない。ログは末尾/追加部分だけを読み、数値の性能情報だけを保持する。
表示を開いたままでもLLMへ生成を依頼しない。新しい常駐サービスやCodexへの周期通知も作成していない。

確認
 14テスト合格: 転送と検証の区別、rename二重計上防止、途中再開の速度、疎/予約ファイル、manifestパス検査、本文/キー非表示、権限不足、旧ログのリセット、SHA照合機能維持等。
 実機--onceとTTYの6秒連続更新・端末復元を確認。
 ダウンローダーは--checkのみ実行。モデル起動、ダウンロード、大容量SHA検証、比較推論は起動していない。
 2026-11-02: IMC/集計/表示の9テストと既存14テストが合格。実機で16イベント取得、多重化なし、6秒の表示更新を確認。確認時はほぼ待機中だったため、当該数値からprefillのボトルネックは判定していない。
