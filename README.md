# x9ramの途中チェックポイントを検算する

最初に `RESULTS_JA.md` を開くと、保存形式の修正前後と、モデルをロードした状態からの編集待ち時間を確認できます。`source/README.md` に独立CLIの起動方法と設定例があります。通常APIは元の設定で運用しています。

展開したディレクトリで、以下を実行してください。LinuxのPython 3.11以降で動作します。

```bash
python3 -B offline_replay.py
```

全ファイルのSHA-256、順序を回した実測の集計、全語彙FP32行、分割して収録した完全状態の復元とトークンヘッダー、CLIの回帰試験を確認します。この再計算にモデルやGPUは不要です。実機の推論そのものを再現するには、同じモデル、公開リポジトリの元ソースと実測時のアダプター、x9ramの起動・保護環境が必要です。

| 場所 | 内容 |
|---|---|
| `workflow-evidence/` | 正規トークンID、独立校正、編集要求ごとの本文・時刻・復帰経路・予算判定、CLI出力 |
| `long-evidence/` | 3,500入力の日本語台帳、実際に構築した3つの途中保存点、同じ状態からの6回の無分割off／固定比較 |
| `native-calls.jsonl`、`rows/` | バッチごとの実入力ID・位置・CPU/GPU割当と、SHA単位で重複を除いた全語彙行 |
| `fixtures/compact/`、`fixtures/physical/` | 元のKV保存と物理配置を保持する版の入力・実保存位置・復元時の評価数・数値比較 |
| `fixtures/ring/` | 実際の物理リングの周回、空の追加入力、途中キャンセル、別のエンジン世代での回復 |
| `image-index.json`、`images/` | 7位置ずつの元方式／修正版の完全画像、周回後1,536位置の修正版画像、長文編集で実際に読んだ3,072位置の画像。ほかの周回位置は元画像のハッシュと数値検証を収録 |
| `source/`、`build/`、`CODE.json` | 独立CLIのソース、nativeパッチ、元ビルドと配備・計測ライブラリの対応 |
| `hardware/`、`SERVICE.json` | 実機用の試験器、通常APIの復帰と保護条件の記録 |

公開パッケージの入力とスロット画像は、この実験用に生成した文章から作成しました。元のモデル重みは配布元から別途入手します。


[GitHubの実装・使い方](https://github.com/MangekyoMyan/xeon-moe-lab/blob/4d4628118f1b01df059d67a03e925d7d5f374f24/current/experiments/state-branches/README.md)・[実測報告](https://github.com/MangekyoMyan/xeon-moe-lab/blob/4d4628118f1b01df059d67a03e925d7d5f374f24/archive/review-handoffs-not-for-readers/2026-11-28-prefix-checkpoints/RESULTS_JA.md)
