# Q1・Q3　実機と共有版の差分、過去の候補の採否

比較元は前回の精度相談commit `5090d9285bb2f769e8da9cc0f1bef5d84c267573`。通常APIに関係する実行物・設定を比較した範囲で、以降に配備された変更は0件だった。11月5日の保存版やGitHubのmainだけを見ると、後日採用した処理が抜ける。現行の配備を追う入口は11月24日の24/24構成である。[通常構成](../../../current/README.md)・[実ロード照合](CURRENT_STATE.json)

**ソースの比較方法。** ローカルのモデル用フォークは `dde879bc4f954b809892fd0d660515f42536394a` のクリーンなcheckout。追跡4,015エントリのうち公開 `vendor/llama.cpp` にある3,995件は、同じ相対パスでGitのモード・種別・オブジェクトIDが全件一致した。通常版はこのツリーに複数の外部ビルドのライブラリを重ねる。そこでCPU C/C++、compactカーネル、MXFP4補助、CUDA、scheduler、loader、build recipe、状態処理を個別に前回commitのGit blobともバイト照合した。比較表には各入力のパスとSHA-256、公開用vendorから省いた20パスがある。[ソース照合表](SOURCE_DIFF.json)・[ビルドと実ロードの対応記録](../2026-11-16-current/review/BINARY_SOURCE_MAP.json)

### 前回の公開資料に含まれなかった候補

| ローカルにあるもの | 実装・確認済みの範囲 | 今回の共有物 |
|---|---|---|
| `chunks32` | 24 worker・T=1・top6のMAT_IDの出力行割当てを32行単位にするCPU DSO。ビルド／ロード先確認まで。通常速度・正しさの実モデル比較は未実施 | [元ソースへの差分](unadopted/chunks32.patch) |
| `join` | x86 OpenMPでCPU graph末尾の明示barrierを省き、直後のOpenMPチーム終了に委ねる比較DSO。ビルド／ロード先確認まで。元のbarrierが他の読者やabort経路を守る可能性もあり、実モデルへの適用は未評価 | [元ソースへの差分](unadopted/join.patch) |
| `spin3000` | `GOMP_SPINCOUNT=3000` を試す比較設定。標準CPU DSOを使う。新CPUでの対照計時は未実施 | [当時の比較設計](unadopted/CPU_COMPARISON.json) |

2つのCソースの基準SHA-256は `e8fdf3d424463f9852fa774d9de6057dce35855a872936bda87d12d3d756e3a8`。各patch、適用結果、候補バイナリの完全なSHA-256を記録した。これらは前回相談より前に手元にあった試作である。今回の棚卸しは推論・計測に関係する差分が対象。以前と同様にモデル全量、全ビルド中間物、全活性値の再配布は対象に含めていない。[差分とハッシュ](SOURCE_DIFF.json)

### 過去の試験と今回の境界との関係

| 状態 | 対象 | 実測と判断 |
|---|---|---|
| 通常採用 | expert単位IQ compact＋元形式へのfallback、MXFP4のQ8入力準備結果を同じ射影内で再利用、CUDAのexpert ID安定整列、ホスト重み登録 | CPU事前展開の有無を射影・expert別に判定。row-localのコードは入っているが通常は無効。簡素化版 `2626bb83` は比較用 |
| 未採用・現行移植が必要 | **03** のT=1選択ID／対応表圧縮 | 実機通常ソースは各MAT_IDで384用の表を扱い、worker間でカウンタを初期化し、各workerがexpertを走査する。03は旧基準向けで直接適用できなかった。現行通常版で03単独の性能は不明 |
| 実機の局所試験で保留 | **04** のQ8_K AVX1変換、**08** のcompact向け入力folding | 旧CPUで04の正しさは確認。08の主要セルは時間比中央値0.957／0.978だったが、試行順によって逆転。08は元のQ8_Kと別のfolded入力を併用する単一射影内の案 |
| 未採用 | **09** の選択入力限定 | 08に追加する案。外部VMの実寸5/6条件で08比1.045、短縮1/6組。新CPUの全モデル測定はない |
| 未採用 | F/B/C、private CPU backend、追加の常駐expert | 旧CPUの23層で、リソースを保持したまま元graphを実行するBに比べ、全expertをCPUへ渡すcustom経路Cは日本語+1.168 ms/T1、Python+0.729 ms/T1。4,866→4,797ノードでも遅くなった。差には再dispatch・copy・同期・配置が含まれる。H=0でも余分なCPU経路を通る設計に注意 |
| 未採用 | shared expert前半の先行投入 | 旧CPUでは通常38→75 CUDA Graph launch／定常T1。37層で前半がCPUに重なった量の中央値1.376 ms。観測なし全体では日本語+2.09%・Python−0.94%の時間変化。新CPUの通常トレースでも現行のshared前半は中央値1.609 ms、重なり0だった |
| 未採用 | GPU常駐expertの分担 | 旧CPUのhot14は配置全体を変えた8課題で7.575→8.427 tok/s、128生成IDの完全一致は1/8。新CPUでの第7層だけの3 expert常駐は日本語10.148→10.063、Python9.810→9.936 tok/s。CPUで未展開のexpertを渡すほど局所利得は大きいが、頻度、H=0時の負担、合流、数値差も効いた |
| 精度・速度比較用 | 入力しきい値128、CPU/GPU残差補正 | 新しい8課題の本文は64/128で一致。固定履歴では3/130で第一候補が変わった。残差で実GGUFに対する局所内積の誤差は小さくできたが、モデル全体の正答率向上は未確定。通常は64/64、追加の残差計算は無効 |

03と08は同じMAT_ID内の仕事を減らす。今回の第一案は**別々のgate・up射影をまたいで**同じQ8入力を使う。旧CPUで遅くなった2出力行SIMDカーネルとも区別できる。初回は既存の1行内積と加算順をそのまま呼び、各射影のcompact/native選択を独立に維持する。[03・08の移植検討](../2026-11-18-combined-handoff/ROOT_REVIEW_JA.txt)・[04実機試験](../2026-11-18-q8-k-avx1/RESULTS_JA.txt)・[F/B/C・08](../2026-11-19-fbc-h1-iq08/RESULTS_JA.txt)・[shared-prefix](../2026-11-19-shared-prefix/RESULTS_JA.txt)・[新CPUのGPU分担](../2026-11-24-normal24-timeline/SUMMARY_JA.txt)・[精度試験](../2026-11-25-selective-margin/RESULTS_JA.md)
