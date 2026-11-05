#!/usr/bin/env python3
"""Write an evidence-based observation report, without adopting any changes."""
import collections
import csv
import json
from pathlib import Path
import re
import shutil

lab = Path(__file__).resolve().parent
root = lab / 'results/bottleneck-observation-20261019'
rows = json.loads((root / 'summary.json').read_text())
by_name = {r['name']: r for r in rows}
memory = json.loads((root / 'memory-summary.json').read_text())
futex = json.loads((root / 'futex-summary.json').read_text())
reference = json.loads((root / 'streaming-reference.json').read_text())
audit = json.loads((root / 'completion-audit.json').read_text())
assert audit['frozen_artifacts_unchanged'] and audit['settings_unchanged'] and audit['same_boot']
assert not audit['active_measurement_processes']
assert sum(r['valid'] for r in rows) == 7
assert all(memory[p]['valid'] and memory[p]['lost_samples'] == 0 for p in ('decode', 'prefill'))
assert not futex['unmatched_entries'] and not futex['unmatched_exits']

symbol_summary = {}
for phase in ('decode', 'prefill'):
    totals = collections.Counter()
    for row in csv.reader((root / f'prior-{phase}-symbols.csv').read_text().splitlines(), delimiter=';'):
        if not row or row[0].startswith('#') or '%' not in row[0]:
            continue
        period, dso, symbol = int(row[1]), row[3].strip(), row[4].strip()
        totals['all'] += period
        if 'libgomp' in dso:
            totals['OpenMP'] += period
            address = re.search(r'0x([0-9a-f]+)', symbol)
            if address and 0x24720 <= int(address.group(1), 16) < 0x24732:
                totals['confirmed_pause_wait_loop'] += period
        if 'ggml_vec_dot_q8_0' in symbol:
            totals['Q8_dot'] += period
        if 'ggml_vec_dot_iq2_xxs' in symbol:
            totals['IQ2_dot'] += period
    symbol_summary[phase] = {'percent_sampled_cycles': {k: 100 * v / totals['all'] for k, v in totals.items() if k != 'all'},
                             'source': f'prior-{phase}-symbols.csv', 'note': 'Earlier same-boot same-build 12-second sample, not simultaneous with new counters.'}
(root / 'symbol-summary.json').write_text(json.dumps(symbol_summary, indent=2) + '\n')

full = by_name['prefill-full-stalls-futex-1']['metrics']
fullcap = json.loads((root / 'prefill-full-stalls-futex-1/capture.json').read_text())
dec = by_name['decode-locality-2']['metrics']
pre = by_name['prefill-locality-1']['metrics']
q8 = next(r for r in memory['decode']['categories'] if r['name'] == 'Q8_dot')
iq2 = next(r for r in memory['decode']['categories'] if r['name'] == 'IQ2_dot')
report = f'''Xeon LLM ボトルネック観察レポート — 2026-10-19

現時点の判断
生成にはメモリ待ちが大きく関わる。しかし「DRAM帯域だけで全体が決まる」
とするには証拠が足りない。量子化形式ごとのCPU処理と、スレッドの待ち合わせもある。
入力処理は生成と性質が異なり、今回の512トークン条件ではCPU側の処理が中心と見られる。

ユーザーの指示どおり最適化は停止。実装、通常の起動スクリプト、OS設定、GPUの扱いは
変更していない。33ファイルパスのハッシュ、27設定項目、同一bootを終了時に照合済み。
追加したのは観察用スクリプトと測定結果。モデルの追加・削除、ビルド、GPU実行なし。

対象と条件
実機: Sugon X9DR3-F、Xeon E5-2630L v2 ×2、12物理コア/24論理CPU。
32GB DDR3-1600 ×4、計128GB、swapなし。DMI上A1/B1/G1/H1に装着。
モデル: DeepSeek-V4-Flash IQ2_XXS-XL、約73.13 GiB。V4.1を測ったものではない。
現行 build-avx-deep、CPUのみ。生成12、入力24スレッド、mmap/interleave、batch512。
同じ合成トークン条件で生成128トークン、入力512トークンを別々に実行。
各々ウォームアップ後。日本語回答・冷間ロード・長コンテキスト全般へは一般化しない。

主要な実測（新しい観察）
項目                              生成                  入力512
DRAM読み出し                      22.47–22.55 GB/s      2.876–2.878 GB/s
DRAM書き込み                      1.35–1.38 GB/s        1.13–1.16 GB/s
L2ミスを伴う実行停止/CPUサイクル   35.65–35.90%          4.17–4.19%
全実行停止/CPUサイクル             45.36–45.44%          23.57–23.65%
CPU実行量（論理CPU換算）           約12.00/12            約21.76–21.80/24
CPU上の周波数指標                  約2.584 GHz           約2.597 GHz
major/minor page fault             0 / 0                 0 / 0
観察窓のプロセスHDD読み出し        0 bytes               0 bytes
スレッドのCPU移動                  0                     0

18秒窓では停止系を各2回、NUMA読み出しを各1回測定した。
さらに入力を56秒間観察（計測ループ{fullcap['bench_timed_seconds']:.2f}秒の約{56 / fullcap['bench_timed_seconds'] * 100:.0f}%）。
この長い窓でもDRAM読み出し{full['dram_read_GB_s']:.3f} GB/s、
L2ミスを伴う停止{full['cycle_activity.stalls_l2_pending:u_fraction_of_cycles'] * 100:.2f}%だった。
冒頭だけの特殊な状態で入力/生成の差が出たとは考えにくい。

注意: サイクル比はCPU内部の指標であり、ユーザーが待つ時間の内訳ではない。
入力はSMTあり、生成はSMTなし。停止率・関数別サンプル・DRAM転送量は重なる尺度。
足して100%にしたり、停止率をそのまま改善可能な速度に換算してはいけない。

1. メモリ帯域と読み出し待ち
同じ実機で既存の2GiB逐次読み出し診断を再実行し、各3回の中央値は
CPUごとにローカル配置 {reference['median_of_run_medians_GB_s']['local']:.2f} GB/s、
interleave配置 {reference['median_of_run_medians_GB_s']['interleave']:.2f} GB/sだった。
推論停止後の背景DRAM読み出しは0.0135 GB/sと小さかった。

生成の約22.5 GB/sは、逐次読み出しの最大転送を常時続けている状態ではない。
ただし両テストはアクセス形態が違う。差を「使える残り帯域」や高速化率とは呼べない。
行列演算ごとの短い帯域飽和、読み出し並列度、NUMA遅延、CPU側の処理を
平均帯域だけで一意に分離することはできない。
観察から確実に言えるのは、生成ではキャッシュの先からデータを待つ停止が大きいこと。

2. Q8とIQ2を同じボトルネックとして扱わない
同じboot/現行ビルドの既存サイクルサンプルでは、生成のQ8内積は
{symbol_summary['decode']['percent_sampled_cycles']['Q8_dot']:.2f}%、IQ2内積は{symbol_summary['decode']['percent_sampled_cycles']['IQ2_dot']:.2f}%だった。
今回、30サイクルを超える抽出ロードを別途採取したところ、生成の76,887サンプル中
Q8内積が{q8['sample_share'] * 100:.2f}%、IQ2内積が{iq2['sample_share'] * 100:.2f}%を占めた。
抽出use-latencyの合計に対する比率は、それぞれ{q8['use_latency_weight_share'] * 100:.2f}%と{iq2['use_latency_weight_share'] * 100:.2f}%。

これらは「Q8はメモリ待ち寄り、IQ2は展開・演算側の比重が大きい」という仮説を支持する。
ただし別窓の異なる種類のサンプルであり、命令数や選択バイアスも違う。
Q8の何%が帯域律速、IQ2の何%が演算律速、という確定値にはできない。

生成の遅延サンプルの81.75%はLFB経由だった。既に進行している読み出しの
データを待つ経路で観測され、最終的なローカル/リモートDRAMの内訳は分からない。
Intelのuse-latencyはCPU内のキュー待ちも含むため、純粋なDRAM応答時間ではない。
全抽出ロードの中央値/95百分位は生成92/803、入力49/267サイクル。
これは閾値を超えた抽出ロード内の統計であり、全ロードの平均遅延ではない。

3. 入力処理はDRAM帯域よりCPU側が中心
入力の主なサイクル消費はQ8の行列演算32.85%、IQ2のAVXタイル演算30.58%など。
DRAM転送量とL2ミスを伴う停止が生成より小さいことと整合する。
ただしCPU側の細分化（数値演算、量子化の展開、L1/L2アクセス、命令供給など）は
まだ完全には分離していない。「浮動小数点演算器だけが上限」とは断定しない。

4. NUMAとメモリチャネル
重みの常駐ページはN0約39.26 GB、N1約39.26 GBで、ほぼ正確に半分ずつだった。
生成の読み出し帯域はCPU0側約11.84–11.90 GB/s、CPU1側約10.61–10.65 GB/s。
片方のCPU側メモリだけに転送が偏る状態ではない。
IMCではCPU0側チャネル0/1、CPU1側チャネル2/3が動作し、残りはゼロだった。
DMIの4枚構成とも整合する。容量と使用チャネル数は別の観察項目である。

数えたDRAM由来retired load uopsのリモート比率は生成{dec['remote_share_of_local_plus_remote_retired_dram_load_uops'] * 100:.2f}%、
入力{pre['remote_share_of_local_plus_remote_retired_dram_load_uops'] * 100:.2f}%。
均等に配置されていても、各CPUがローカルだけを読むわけではない。
この比率は転送バイト数でも全アクセス比でもない。プリフェッチやLFBは扱いが異なる。
QPIリンクの飽和度、リモートアクセスによる正味の遅延増は今回未確定。

5. 待ち合わせをCPU上の待ちとOS上の待機に分ける
生成の既存CPUサンプルはOpenMP全体{symbol_summary['decode']['percent_sampled_cycles']['OpenMP']:.2f}%。
うち{symbol_summary['decode']['percent_sampled_cycles']['confirmed_pause_wait_loop']:.2f}%は逆アセンブルで確認したpause待ちループ内だった。
生成の12スレッドは新しい窓でもほぼ常時CPU上にいた。CPU使用率100%でも有効演算だけではない。
これはサンプル上のサイクル比であり、その割合だけ壁時計時間を削れる意味ではない。

入力では24スレッドを{futex['telemetry_seconds']:.3f}秒観察し、CPU上で走っていない時間は
全スレッド合計{futex['thread_offcpu_seconds']:.3f}スレッド秒だった。
同じ窓に切り詰めたfutex待機呼び出しは合計{futex['successful_wait_calls']['sum_syscall_seconds_clipped_to_telemetry']:.3f}スレッド秒。
OSの実行待ち行列での待ちは合計{futex['thread_runqueue_seconds']:.3f}スレッド秒と小さい。
入退出11,032組はすべて対応し、欠落イベントは報告されていない。
入力のCPU非実行時間の大半は待ち合わせと対応し、他プロセスとのCPU取り合いが主因ではなかった。
futex時間にはカーネル処理と復帰時の待ちも含まれる。スレッド間で時間が重なるため、
これらを処理全体の56秒へ足したり、改善可能な秒数とは解釈しない。

6. HDD、容量、温度
有効な全観察窓でHDD読み出し・major/minor page fault・CPU移動はゼロ。
モデル常駐後の推論について、HDDや容量不足を主因とする証拠はない。
観察中のRSSは生成約73.29 GiB、入力約73.52 GiBだった。
CPU温度の1秒間隔サンプルの最高は生成46℃、入力50℃。熱抑制カウンタ増分ゼロ。
周波数指標も安定していた。電力制限の専用カウンタは今回採取していない。
冷間ロードやダウンロード速度、将来の大きいモデルについての評価ではない。

測定の採否と限界
・生成/入力の冒頭18秒窓を各3回採用（停止系2回、NUMA系1回）。
・入力56秒窓は主カウンタとIMCを採用。補助のcycles_l2_pendingだけ稼働率表示99%で除外。
  このため収集ラッパーは当初の全イベント判定で失敗終了したが、フェーズ完了・主カウンタ100%・
  57点のテレメトリを確認し、補助イベントを含まない結果として個別に採用した。元ログは変更しない。
・最初のdecode-stalls-1とdecode-locality-1は主カウンタが交代制になったため除外。
・報告に使った主カウンタとIMCの稼働率表示は100%。
・別の遅延サンプル2回はlost samplesゼロ。入力futexトレースは全入退出が対応。
・同一条件の再測定は一致したが、異なるプロンプト・長コンテキスト・他モデルは対象外。
・NUMA、キャッシュ、同期、演算は互いに影響する。足して100%になる原因内訳は作っていない。

観察後の状態
最適化は引き続き停止。通常ランチャーと現行ビルドはそのまま。測定プロセスは終了済み。
この結果から新しい設定の採用や、メモリ増設後の速度の断定は行っていない。

再現・確認するファイル
summary.json: 新規カウンタ、チャネル別帯域、スレッド別実行量、温度、採否。
memory-summary.json: 閾値付きロードサンプルの統計と関数別内訳。
futex-summary.json: 入退出対応、同一窓に切り詰めた待機時間。
freeze-before.json / freeze-after.json: 実装と設定が変わっていないことの照合。
METHODS_JA.txt: 方法、単位、イベントの解釈。
各capture.json / observations.json / CSV / ログ: 生データと実行コマンド。
tools/: 観察と集計のスクリプト。出力先は既存ディレクトリを上書きしない。
raw-capture-hashes.json: サーバーに残すperf.dataのハッシュ。ZIPには生perf.dataを含めない。

定義の参照
Intel IvyTown PMU: https://perfmon-events.intel.com/platforms/ivytown/core-events/core/
Linux perf-mem: https://man7.org/linux/man-pages/man1/perf-mem.1.html
'''
(root / 'REPORT_JA.txt').write_text(report)
(root / 'REPORT.json').write_text(json.dumps({'status': 'observation_complete_optimization_paused',
    'audit': audit, 'new_counters': rows, 'symbol_summary': symbol_summary,
    'memory_summary_file': 'memory-summary.json', 'futex_summary': futex,
    'reference_bandwidth_GB_s': reference['median_of_run_medians_GB_s'],
    'claims': {'generation': 'Substantial cache-miss-related waiting, quantized kernels, and synchronization; pure bandwidth saturation not established.',
               'prefill': 'CPU-side kernels dominate under the measured 512-token workload; DRAM bandwidth saturation is not supported.',
               'unknown': ['Per-kernel bandwidth saturation versus latency', 'QPI saturation', 'Detailed CPU execution-resource limits', 'Long-context/other-model behavior', 'Cold loading and future RAM upgrades']}}, indent=2) + '\n')
toolsdir = root / 'tools'; toolsdir.mkdir(exist_ok=True)
for name in ('observe_bottlenecks.py', 'observe_phase.py', 'run_bottleneck_observations.py',
             'run_bottleneck_memory_samples.py', 'observe_memory_reference.py', 'read_bandwidth.c',
             'summarize_bottleneck_observations.py', 'summarize_memory_observations.py',
             'summarize_futex_observation.py', 'finalize_bottleneck_observations.py'):
    shutil.copy2(lab / name, toolsdir / name)
prior = root / 'prior-profile-evidence'; prior.mkdir(exist_ok=True)
for phase in ('decode', 'prefill'):
    shutil.copy2(lab / f'results/deep-review-profile-final-{phase}.json', prior / f'{phase}-summary.json')
    for name in ('core.csv', 'imc.csv', 'capture.json', 'sidecar-counters.json'):
        shutil.copy2(lab / f'results/deep-profile-final-{phase}' / name, prior / f'{phase}-{name}')
p = lab / 'experiment-preferences.json'; preferences = json.loads(p.read_text())
preferences['current_work'] = 'Observation completed 2026-10-19. Active build, launcher and OS settings verified unchanged. Optimization remains paused at user request. Read the bottleneck observation report before proposing or applying further tuning.'
preferences['bottleneck_observation_report'] = str(root / 'REPORT_JA.txt')
preferences['bottleneck_observation_archive'] = '/home/codex/projects/xeon-bottleneck-observation-20261019.zip'
p.write_text(json.dumps(preferences, indent=2) + '\n')
print(json.dumps({'report': str(root / 'REPORT_JA.txt'), 'valid_counter_observations': 7, 'optimization_paused': True}, ensure_ascii=False))
