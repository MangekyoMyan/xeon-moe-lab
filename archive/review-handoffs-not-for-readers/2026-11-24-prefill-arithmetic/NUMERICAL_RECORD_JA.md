# 数値の検算

この記録は、2026年11月24日のx9ram上のDeepSeekテキスト推論で、CPU交換後の入力しきい値による差を特定した実験である。結果と通常APIの設定は [実測結果](RESULTS_JA.md) にまとめた。

公開ディレクトリで `python3 -B verify_record.py .` を実行すると、Pythonの標準ライブラリで、会話の採点、時刻、各コアへの担当行数、OpenMPの待ち、expert集合の変化、FP64参照値との誤差、固定履歴での第一候補の差を検算できる。PythonとSQLの回答は、既知の入力例に対して動作を検査した。

- `risk/` の各TSVは全129,280語彙を計算した後の位置別統計。AとBの第一・第二候補、順位、確率、参照トークンの負の対数尤度、総変動距離、Jensen–Shannon divergence、両方向のKLなどを保存している。温度1、語彙の切り捨てを行う前の分布を使う。
- `precision/*-dot.tsv` はGGUFの実ファイルから該当する行だけを読み出して再構成した内積である。CPU・CUDAの観測値、元入力との倍精度内積、CPU側のQ8_K内積、残差を補った場合、Voltaの計算経路に合わせた入力量子化を同じ行で比較できる。
- `runs/precision-*/` には内部観測で出力を変えていないことの照合、実行時ライブラリ、入力ID、対照の貪欲生成に使った全ID、第0層の計算に用いた活性値・expert ID、全40層のexpert IDがある。
- `runs/causal/` は128設定で第0層の指定箇所だけを変えた隔離試験。通常GPUの記録から置換した実験と、実際にCPUの内積を追加して残差を補正した実験を含む。受信・書き戻しのバイト一致を確認している。
- 単一生成を追跡した16本のワーカー時系列が `runs/worker-trace/` にある。同時計測したCPU実行標本は `cpu-PC.tsv`。観測器は隔離プロセスで使用した。通常APIのしきい値は64 / 64である。

実機には全語彙のF32原本と中間テンソルも保存した。公開記録には位置別の数値、選んだ内積の生データ、全語彙原本のSHA-256を含めた。実機固有のビルド・起動スクリプトは当時のパスを参照する。

基盤はllama.cpp・ggmlのMIT実装であり、CPUの並列処理はGNU OpenMP、GPUはNVIDIA CUDA、CPU標本はLinux perfを使用した。ソースの由来と権利表示はリポジトリ直下のLICENSE、THIRD_PARTY_NOTICES.mdおよびACKNOWLEDGMENTS.mdにある。元になった通常構成と旧測定は [24/24実装の前回記録](https://github.com/MangekyoMyan/xeon-moe-lab/tree/f6eafb2d4ffa55c348ac004c97ac6b08a96f6821/archive/review-handoffs-not-for-readers/2026-11-24-normal24-timeline) にある。
