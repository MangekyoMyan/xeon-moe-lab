# E5-2697 v2 ×2への移行

2026年11月19日の実測機はE5-2630L v2×2、12物理コア・24論理CPUです。512GiBを構成する32GB DIMM 16枚は、すべてBIOS上の設定値が1600 MT/sでした。DMIが報告する基板はSugon X9DR3-F、BIOSは3.4です。

E5-2697 v2×2の仕様は合計24物理コア・48論理CPUです。両CPUはIvy Bridge EPで、各ソケットに4個のメモリチャネルがあります。2697 v2のDDR3-1866はCPU側の対応上限です。組み込んだDIMMの速度は交換後にBIOSの設定値を読み取ってください。1個当たりのTDPは60Wから130Wになります。電源を見直す際は、基板・BIOSの130W品対応とCPU冷却を実装される個体に合わせて確認してください。[Intel E5-2630L v2](https://www.intel.com/content/www/us/en/products/sku/75791/intel-xeon-processor-e52630l-v2-15m-cache-2-40-ghz/specifications.html)・[Intel E5-2697 v2](https://www.intel.com/content/www/us/en/products/sku/75283/intel-xeon-processor-e52697-v2-30m-cache-2-70-ghz/specifications.html)

交換後の最初の確認は、APIを起動する前に実行できます。

```bash
python3 current/check.py topology
```

このコマンドはLinuxのCPU番号・ソケット・コアの対応と、そのプロセスから使える各コアの代表CPUを表示します。旧機の通常設定は生成12、入力処理24、strict maskはどちらも `0xffffff`、OpenMPの上限は24です。新機では実際の番号からマスクを決めます。

1. 同じモデル・量子化・配置計画を使い、両ソケットから同数ずつ選んだ12物理コアと24物理コアで、入力処理と生成を個別に測ります。メモリの動作速度と負荷中の持続クロックも記録します。
2. ワーカー数を決めてから、CPU/GPUへの割当て、転送を始める閾値、展開済みの重みに使うRAMを一項目ずつ比較します。モデルの頻度記録は再利用できます。各expertを展開する価値は実時間で再評価します。
3. 現行実装のキャッシュ生成ワーカー数12は、ロード時に渡す値です。生成時のワーカー数は別の引数です。5120行のdown射影では、1入力・ggml内部のNUMA分割が無効のとき、ワーカー数を12から24にすると80個の64行区間からワーカー数に応じた区間分割に変わります。OSのメモリ交互配置は別の設定です。

新CPUの実測値は、交換後のプロファイルに記録します。
