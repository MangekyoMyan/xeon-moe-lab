# 再現手順と成果物

Linux x86-64、GCC/G++、CMake、make、Python 3、patch、util-linuxのtasksetで再現できる。CPUはAVX、SSSE3、SSE4.2、F16Cを使えること。固定commitは `37f6390c9a5bc594658127651e204ab11420cd71` 。GitHubの完全なcheckoutでも、今回提供された `source_repo` でも `scripts/prepare_source.py` と `accepted` と `vendor/llama.cpp` を使う。スクリプトは新規ディレクトリにのみ書き込む。原本のoverlayを変更しない。

## 差分の選択

| ファイル | 適用先と関係 | 対応する記録名 |
|---|---|---|
| 01-route-workspace.patch | 固定版からの初版。索引の疎化。小バッチでの退行を記録した比較用 | route |
| 02-compact-input-signs.patch | 固定版からの初版。FE_DOWNWARDで不一致があり履歴比較用 | iq |
| 03-route-selected.patch | 固定版へ直接適用する01の改版。01を先に当てる必要はない | route2 |
| 04-q8k-avx1.patch | 固定版へ直接適用するQ8_K量子化単独版 | q8 |
| 05-combined.patch | 03・04・旧02を統合した固定版向け履歴比較用。旧02の不一致を含む | combined |
| 06-compact-signed-byte.patch | 固定版へ直接適用する02の修正版。-128補正を含む | iqsafe |
| 07-combined-corrected.patch | 固定版へ直接適用する03・04・06の統合版。これ一つで3変更を含む | combosafe |
| route37-fixture/08-mixed-dispatch.patch | コメント除去済みの診断版関数抜粋2ファイル用。元のQ8_Kとfolded入力を併用 | fixtureのmixed |

03・06ともCPU演算の共通箇所を編集する。3変更を試す場合は07を選ぶ。実使用版に `xeon_iq3_cache_lookup_experts_v2` がある場合は、追加ソースの節を先に確認する。

## 固定版からビルドし、正しさを照合する

```bash
trial_files=/path/to/outputs
engine_repo=/path/to/xeon-moe-lab
repro_dir="$PWD/ivb-repro"
bash "$trial_files/reproduce.sh" "$engine_repo" "$repro_dir" 4
```

基準、03、04、06、07ごとに別のソースとbuildを作る。各ソースを再構成・SHA照合し、fuzzなしでpatchを当て、CPU版ggmlをビルドする。通常・境界・4丸めのCPUグラフを実行し、同一入力の全F32出力を `cmp` する。親の最小丸め反例、Q8_Kの参照バイト列、tiled無効・強制も照合する。成果物は `$repro_dir/raw` と実行生成の `.bin` ファイルに残る。CPUのISA、コンパイラ、build pathでELFハッシュは変わり得るので、パッチ・機能・出力の対応も確認する。

今回のVMでCMakeは次のようにローカル導入した。実際に使ったCMakeは4.4.5、GCCは14.2.0。ninjaも同時に導入したが、CMakeが生成したのはUnix Makefiles。

```bash
python3 -m pip install --target "$PWD/build-tools" cmake ninja
export CMAKE_BIN="$PWD/build-tools/cmake/data/bin/cmake"
```

旧版01/02/05まで同じ方法で再現するには、別の新規出力先を指定して `INCLUDE_ARCHIVED=1 bash "$trial_files/reproduce.sh" "$engine_repo" "$PWD/ivb-with-history" 4` を実行する。この場合、旧02/05の下向き丸めでバイト差が生じることも検査する。

## 最終の11条件をA/B計時する

```bash
python3 "$trial_files/time_followup.py" "$repro_dir" --rounds 8 --samples 5 \
  --cpu1 2 --cpuN 2-4 --cases route2-tiny route2-many route2-mxfp4 \
  q8-256 q8-7168 q8-wide-graph safe-iq2s-gate safe-xxs-gate \
  safe-3s-down safe-xxs-down-3t safe-combo-gate > "$repro_dir/raw/final-timing.csv"
python3 "$trial_files/summarize.py" "$repro_dir/raw/final-timing.csv" \
  > "$repro_dir/raw/final-summary.csv"
```

`--cpu1` と `--cpuN` はそのホストの許可CPUに合わせて変更する。上の条件は1スレッドと3スレッドのVM対照用。実機の12スレッド版は、同じprobeの引数を明示して別の条件として記録できる。例えば下記はK・出力行数・6選択を合わせ、bankは12個に抑えた合成のIQ3_XXS down試験。コマンドは1行を5サンプル出し、2つのDSOを同じseedで比較する。実運用のaffinityはその実機で改めて確認する。

```bash
for impl in baseline iqsafe; do
  OMP_DYNAMIC=false LD_LIBRARY_PATH="$repro_dir/lib-$impl" taskset -c 0-11 \
    "$repro_dir/graph_probe" bench iq3_xxs 2304 5120 12 6 1 6 12 1 5 8 5 90791
done
```

`graph_probe bench` の引数順は「重み型、K、出力行数M、総expert数E、選択枠S、トークン数T、活性列A、スレッド数、compact有無、route pattern、反復回数、サンプル数、seed」。pattern 3は順番に選ぶ固定集合、5は反復ごとに集合を交替する。実行時間だけを測り、重みの生成・cache追加・graph plan・ウォームアップは計時外。グラフ全体のバッファ、ロード時間、トークン生成速度は別指標である。実E384の作業領域は `LD_LIBRARY_PATH="$repro_dir/lib-route2" "$repro_dir/graph_probe" plans` で、巨大なモデル配列を実確保せずに計画値を取得できる。

## 部分compactの関数抜粋を照合する

```bash
bash "$trial_files/route37-fixture/reproduce.sh" "$engine_repo" "$PWD/route37-fixture-run" 4
```

`route37-fixture/excerpts/SOURCE_MANIFEST.json` には元のcommit・file path・行・hashがある。スクリプトは比較する二つの版をともに新規ソースから準備し、演算とplanの全関数を差し替える。追加ソースのMXFP4専用プリパックは無効にし、実装のない関数は呼ばれたらabortする。スパースcacheの代わりに、固定版で生成した真のcompactデータへのポインタを選択テーブルへ注入する。`XEON_TEST_MASK=31` は6選択のうち最初の5つをcompactにする。実際のチャンク呼出しは追加資料の関数コードが行う。診断ログのrecord関数のみ空の試験関数。計数用LD_PRELOADは別実行で使用し、計数の有無で結果が一致することも比較する。0・1・3・5・6ヒット、4丸めモードの出力は `raw/hashes.txt` に保存される。

実機の統合先では、対象バイナリの元ソース・IQ型・各expertのcache状況・量子化前後の入力型を確認する。必要な契約は、元形式とfolded形式のstrideを分ける、量子化済みの同一ブロックから作る、各workerが自分の範囲を書いて既存barrierで公開する、native側とtiled側は元形式を読むこと。通常の12/24スレッド、普段のNUMA方針、同一のモデル・ルート・プロンプトで、まず演算出力を照合し、次にdecodeとprefillを別々に計時する。全体検証には少なくともlogits・選択expert・生成tokenと、キャッシュ充填率別の時間が必要で、これらは今回のVMでは測っていない。

## 生データの見方

- `RUN_MANIFEST.json` の `timing_final_clean_builds` は最終5DSOのSHA、06のsymbol数、Q8_K関数内のAVX変換命令、route plan値を示す。`timings` が各phaseのrawとDSO群を結ぶ。1回目は360行、2回目は1,120行、最終は880行。親の指摘で除外した一時DSOのSHAも記録した。
- `raw/final-checks` は最終計時に用いた同じ5DSOから、通常・境界・4丸めを出力し直したCSV。`RUN_MANIFEST.json` の `ordinary_and_rounding_final` に各dumpとCSVのSHAがある。workspaceには `work/final-evidence/{variant}-{check|edges|round0|round1|round2|round3}.bin` に全出力を保存した。この配布物の `raw/dumps` には5版で同一と確認した基準dumpのgzipと、旧02で差が出た下向き丸めdumpを入れた。
- `raw/round-comparisons.csv` は修正前から修正後までの丸め差。`raw/review-rounding-*.csv` は親が作成した最小反例の直接実行結果。`raw/q8-byte-check-isolated.csv` はQ8_Kのスケール・バイト・bsums・guardの比較。
- `raw/fixture-comparison.csv` と `raw/route37-fixture/cases.csv` は混在fixtureの個別結果。`raw/route37-fixture/paths-*.csv` は試験用の実行経路計数。実行ファイル・二つのDSO・インターポーザのSHAもmanifestに収録した。
- `raw/environment.txt` と `raw/isolated-*.log` にCPU、コンパイラ、ビルドログがある。`raw/asan-*` は履歴版05について通常丸めの186グラフをAddressSanitizerで走らせた補足資料。最終版の安全性をこの履歴版のASan結果から推定していない。
