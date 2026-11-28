# 途中保存の実機記録

数値を調べる入口は [日本語の報告](RESULTS_JA.md) です。各ファイルはx9ram上の独立CLIの合成入力を使っています。普段のAPIの変更は含みません。

このディレクトリで `python3 -B summarize.py` を実行すると、順序を回した比較の生記録から編集待ち時間を集計できます。全語彙行と完全なKV画像を含む再計算は、[データ共有用ブランチのZIP](https://codeload.github.com/MangekyoMyan/xeon-moe-lab/zip/refs/heads/handoff-prefix-checkpoints-20261128)で `python3 -B offline_replay.py` を実行してください。元ログとFP32全行の対応、画像本体からのIDとハッシュの再確認、CLIの回帰試験をまとめて行います。
