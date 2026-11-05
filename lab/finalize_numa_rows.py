#!/usr/bin/env python3
"""Write the reviewed decision and preserve the normal runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('--moe-decision-reason', required=True)
args = ap.parse_args()
lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-numa-rows-20261020'
path = out / 'REPORT.json'
report = json.loads(path.read_text())
assert report['integrity']['active_artifacts_unchanged_before_metadata_update']
if 'decision' in report:
    raise SystemExit('Preserve existing final decision')
report['decision'] = {
    'adopted': False, 'active_build': str(lab / 'build-avx-prefetch'),
    'numa_rows': 'Not adopted: extra copy and startup cost, small variable synthetic gain, slower fixed Japanese generation.',
    'moe_groups': args.moe_decision_reason,
    'experimental_sources_and_builds_retained': True}
comparisons = {c['name']: c for c in report['comparisons']}
rows = report['full_model_validation']['model-validation']
moe = report['full_model_validation']['moe-model-validation']
a = report['locality_observations']['A']['metrics']
b = report['locality_observations']['B']['metrics']
lines = [
    'Xeon: NUMA配置とMoEの仕事配分の実験結果 — 2026-10-20',
    '',
    '今回は2案とも通常版への採用を見送った。通常起動は検証済みのbuild-avx-prefetchを維持。',
    '実装・ビルド・単体検証・実モデル比較まで実施。実験用ソースと実行ファイルは保存。',
    '',
    '対象と測定',
    'Xeon E5-2630L v2 x2、12物理コア/24論理CPU、128GB。GPUは使用していない。',
    'DeepSeek-V4-Flash IQ2_XXS-XL、約73.13GiB。V4.1の測定ではない。',
    '再起動後に既存モデルを読み込み、同一boot内で比較。計測のHDD読み出しは全比較で0。',
    '生成ベンチは入力0・生成32・深さ0、12スレッド、各プロセス3反復。batch/ubatch512、mmap/interleave。',
    'ABBA/BAABの各順序で各構成2プロセス。比較単位はプロセス平均。重いページ配置観察は別実行。',
    '',
    '1. Q8重みの行をCPUごとに分けて配置',
    '対象107個、合計3,253,207,040 bytesのQ8重みを1コピーだけ確保。',
    '前半の行をnode0、後半の行をnode1へmbindし、12ワーカーの前半・後半の担当と一致させる。',
    '1列の入力だけ16行単位の固定担当にし、既存の入力量子化・バリア・内積を再利用。複数列は一般処理。',
    '214領域すべてで物理配置を確認。ワーカー0-5/node0、6-11/node1の対応も確認。',
    '対照は同じ1コピー・同じ固定担当のまま、コピーをinterleaveに置いた版。',
    'この比較は配置方針の総合効果であり、THP形成などを独立に固定した純粋なDRAM遅延比較ではない。',
]
for key in ('compare-shard-current-abba', 'compare-shard-current-baab', 'compare-locality-control-abba'):
    c = comparisons[key]
    lines.append(f"  {key}: {c['medians']['A']:.6f} -> {c['medians']['B']:.6f} tok/s、{c['gain_percent']:+.3f}%")
resource = comparisons['compare-shard-current-abba']['resources']
rss = {v: statistics.median(r['max_rss_gib'] for r in resource if r['variant'] == v) for v in 'AB'}
lines += [
    f"  観測最大RSSのプロセス間中央値: {rss['A']:.3f} -> {rss['B']:.3f} GiB。",
    f"  固定日本語応答の生成速度差: {rows['generation_gain_percent']:+.3f}%。",
    f"  読み込み時間中央値: {rows['median_ms']['A']['load']:.2f} -> {rows['median_ms']['B']['load']:.2f} ms。",
    '追加メモリと起動時のコピー負担があり、今回の実応答でも遅くなったため不採用。',
    '',
    '2. MoEの専門家ごとにワーカーをまとめる',
    'IQ2_XXS・1入力トークン・重複のない専門家IDだけが対象。重みの追加コピーなし。',
    '6専門家・12ワーカーなら各専門家を2ワーカーで担当。既存のgate/upペアと内積を再利用。',
    '複数入力、重複ID、他形式、参照実行、強制tiledでは従来経路。既存バリアは維持。',
]
for key in ('compare-moe-current-abba', 'compare-moe-current-baab', 'moe-control'):
    c = comparisons[key]
    lines.append(f"  {key}: {c['medians']['A']:.6f} -> {c['medians']['B']:.6f} tok/s、{c['gain_percent']:+.3f}%")
lines += [
    f"  固定日本語応答の生成速度差: {moe['generation_gain_percent']:+.3f}%。",
    args.moe_decision_reason,
    '',
    '別実行のNUMAカウンター観察',
    f"  DRAM由来retired load uopsのリモート割合: {100*a['remote_share_of_local_plus_remote_retired_dram_load_uops']:.2f}% -> {100*b['remote_share_of_local_plus_remote_retired_dram_load_uops']:.2f}%。",
    f"  DRAM読み出し帯域: {a['dram_read_GB_s']:.3f} -> {b['dram_read_GB_s']:.3f} GB/s。",
    f"  CPU温度最高: {a['cpu_temperature_max_C']:.0f} / {b['cpu_temperature_max_C']:.0f}℃。熱抑制増分0。",
    '各18秒の別観察窓。主要カウンターの稼働率100%。HDD読み出し・page fault・CPU移動は0。',
    'リモート割合は転送バイト数でも全ロードの割合でもない。プリフェッチとLFBの扱いも異なる。',
    '観察実行の速度は採用判断に使っていない。今回の変更で遠い側の読み出しを減らせても、速度改善は限定的だった。',
    '',
    '正しさ',
    'NUMA候補: 配置・入力列数・スレッド数・部分書き込みを含む63ケースを2配置モードで確認。',
    '一般行列80ケース、既存MoE96ケース、既存量子化テストも通過。',
    'MoE候補: 576ケースを通常版・実験機能無効・有効で比較し、すべてビット一致。',
    '1/2/3/4/5/6/12/24スレッド、端数行、重複ID、専門家別入力、ビュー、複数トークンを含む。',
    '両候補とも固定160トークン中34位置 x 129280語彙 = 4,395,520個の予測スコアが通常版とビット一致。',
    '固定59トークンの日本語プロンプト、同じ24評価トークンの回答も全回一致。BLUE-731/東京/42を確認。',
    '一行形式への厳密な追従や、幅広いモデル品質は評価していない。',
    '',
    '状態と限界',
    f"通常ソース{report['integrity']['source_files_checked']}ファイル、主要バイナリ・ランチャー・再ビルド手順のハッシュを確認。",
    f"OS設定{report['integrity']['settings_count']}項目も変更なし。常用は前回のQ8先読み版を維持。",
    '追加の512トークン入力性能試験は生成の採用スクリーニングで止めたため、この候補の入力512性能は未評価。',
    '結果は今回の短い生成と固定日本語応答の範囲。長文脈、同時処理、メモリ増設、他モデル、GPU、電力へ一般化しない。',
    'ハードウェアの絶対上限を証明した結果ではない。今回の2つの配分・配置変更では、大幅な高速化に結びつかなかった。',
    '',
    '通常起動: ' + str(lab / 'run_deepseek_flash.sh'),
    '構造化結果・実行コマンド・生ログ参照: ' + str(out / 'REPORT.json'),
]
(out / 'REPORT_JA.txt').write_text('\n'.join(lines) + '\n')
preferences_path = lab / 'experiment-preferences.json'
raw = preferences_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == report['experiment']['active_sha256'][str(preferences_path)]
(out / 'preferences-before-final-update.json').write_bytes(raw)
preferences = json.loads(raw)
preferences['current_work'] = (
    'Completed row-local Q8 placement and grouped-expert MoE experiments. Neither adopted; '
    'the active build-avx-prefetch runtime, launcher and source are unchanged. '
    'Q8 sharding reduced observed remote DRAM-load-uop share but added about 3 GiB and slowed the fixed Japanese response. '
    'Expert grouping produced sub-percent synthetic differences; see the report for practical-response results and decision.')
preferences['numa_work_distribution_results'] = str(out / 'REPORT_JA.txt')
preferences['numa_work_distribution_archive'] = str(lab.parent / 'xeon-numa-work-distribution-20261020.zip')
preferences_path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2) + '\n')
report['metadata_update'] = {'path': str(preferences_path),
    'before_sha256': hashlib.sha256(raw).hexdigest(),
    'after_sha256': hashlib.sha256(preferences_path.read_bytes()).hexdigest(),
    'runtime_configuration_changed': False}
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report': str(out / 'REPORT_JA.txt'), 'adopted': False, 'runtime_unchanged': True}))
