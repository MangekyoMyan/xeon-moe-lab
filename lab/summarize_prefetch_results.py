#!/usr/bin/env python3
"""Collect completed experiments without mixing observer protocols."""
import hashlib
import json
from pathlib import Path
import statistics

lab = Path(__file__).resolve().parent
out = lab / 'results/optimization-memory-sync-20261019'
experiment = json.loads((out / 'experiment.json').read_text())
paths = dict(experiment['comparisons'])
for r in json.loads((out / 'light-followups.json').read_text()):
    paths['light_' + r['name']] = r['saved']
final = json.loads((out / 'final-checks.json').read_text())
for r in final:
    if r['name'].startswith('canonical-'):
        if r['exit_code'] or 'saved' not in r: raise SystemExit('Final comparison incomplete')
        paths[r['name']] = r['saved']
if not all(k in paths for k in ('canonical-decode-abba', 'canonical-decode-baab', 'canonical-prefill-abba')):
    raise SystemExit('Final comparisons incomplete')
rows = []
for name, path in paths.items():
    data = json.loads(Path(path).read_text())
    observations = [json.loads(Path(r['summary']).read_text()) for r in data['runs']]
    b = observations[0]['benchmarks'][0]
    medians = {v: statistics.median(r['avg_ts'] for r in data['runs'] if r['variant'] == v) for v in 'AB'}
    rows.append({'name': name, 'path': path, 'order': data['order'], 'prompt': b['n_prompt'], 'tokens': b['n_gen'],
        'threads': b['n_threads'], 'depth': b['n_depth'], 'repetitions_per_process': len(b['samples_ts']),
        'observer': observations[0].get('telemetry_mode', 'full: live numa_maps scan'),
        'median_tokens_per_second': medians, 'gain_percent': data['candidate_gain_percent'],
        'same_boot': data['same_boot'], 'warm_cache': data['warm_cache_observed'],
        'process_means': [{'variant': r['variant'], 'tokens_per_second': r['avg_ts']} for r in data['runs']],
        'observer_cpu_seconds': [r.get('observer_cpu_seconds') for r in observations],
        'max_rss_kib': [max(s['rss_kib'] for s in r['observations']) for r in observations]})
model = json.loads((out / 'permanent-model-validation/comparison.json').read_text())
reverse = json.loads((out / 'response-reverse.json').read_text())
if not model['complete'] or not reverse['complete']: raise SystemExit('Response comparisons incomplete')
response_rows = [{'variant': r['variant'], 'eval_ms': r['timings_ms']['eval']} for r in model['responses'] + reverse['runs']]
response_medians = {v: statistics.median(r['eval_ms'] for r in response_rows if r['variant'] == v) for v in 'AB'}
all_responses = [r['response'] for r in model['responses'] + reverse['runs']]
activation = json.loads((out / 'activation.json').read_text()) if (out / 'activation.json').exists() else {'activated': False}
baseline_check = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    for p, digest in experiment['baseline_sha256'].items() if not p.endswith('/run_deepseek_flash.sh')}
old_freeze = json.loads((lab / 'results/bottleneck-observation-20261019/freeze-before.json').read_text())
settings = {}
for path, value in old_freeze['settings'].items():
    settings[path] = {'before': value, 'after': Path(path).read_text().strip()}
report = {'hardware': {'board': 'Sugon X9DR3-F', 'cpu': '2 x Xeon E5-2630L v2', 'physical_cores': 12,
    'logical_cpus': 24, 'ram': '4 x 32 GiB DDR3-1600, two populated channels per socket', 'gpu_used': False},
    'model': 'DeepSeek-V4-Flash IQ2_XXS-XL, 73.126 GiB (not V4.1)',
    'changes': 'Bounded T0 weight prefetch eight Q8 blocks ahead, AVX single-row dot only. No floating-point arithmetic, row4 kernel, MoE or wait-policy changes.',
    'comparisons': rows, 'model_validation': model, 'response_reverse': reverse,
    'combined_response': {'runs': response_rows, 'median_eval_ms': response_medians,
        'gain_percent': 100 * (response_medians['A'] / response_medians['B'] - 1),
        'responses_identical': len(set(all_responses)) == 1,
        'scope': 'One fixed 59-token Japanese prompt, same 24 evaluated tokens, four independent processes per variant.'},
    'micro': json.loads((out / 'q8-micro/comparison.json').read_text()),
    'activation': activation, 'baseline_source_and_binaries_unchanged': baseline_check,
    'settings': settings, 'settings_unchanged': all(v['before'] == v['after'] for v in settings.values()),
    'same_boot_as_observation': Path('/proc/sys/kernel/random/boot_id').read_text().strip() == old_freeze['boot_id'],
    'limitations': ['Small synthetic workload and one fixed Japanese prompt; no universal speedup claim.',
        'No cold HDD loading, RAM upgrade, GPU, other model, long-context or power measurement.',
        'Observer protocol changed; old-to-light baseline throughput difference is not an optimization gain.',
        'CPU function samples and memory-stall fractions were not remeasured after this small kernel change.']}
(out / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
lookup = {r['name']: r for r in rows}
decode = [lookup[k] for k in ('canonical-decode-abba', 'canonical-decode-baab')]
prefill = lookup['canonical-prefill-abba']
response = report['combined_response']
status = '通常ランチャーへ採用済み。' if activation['activated'] else '通常ランチャーは旧版のまま。採用判断はactivation.jsonを参照。'
lines = [
    'Xeon / DeepSeek メモリ待ち・同期の最適化結果 — 2026-10-19', '', status,
    f"生成ベンチは{min(r['gain_percent'] for r in decode):+.2f}〜{max(r['gain_percent'] for r in decode):+.2f}%。入力処理は実質同等。",
    '短い日本語実応答は反復で差の向きが変わり、明確な速度改善とは見なさない。',
    '対象: Xeon E5-2630L v2 x2、12コア/24論理CPU、128GB。GPUは使用していない。',
    'モデル: DeepSeek-V4-Flash IQ2_XXS-XL、約73.13GiB。V4.1の測定ではない。', '',
    '今回の変更',
    'AVX用Q8の1行内積で、8ブロック先の重みをT0ヒントで先読みする3行を追加した。',
    '先読み先は行内に制限。演算順序・重みの精度・既存の4行処理・MoE処理は変更しない。',
    '追加の重み複製や大きい作業バッファはない。OpenMPの同期方式・待機設定も現行のまま。', '',
    '通常利用向けビルドの速度比較（軽い監視、mmap/interleave、同一boot、ウォーム状態）',
]
for r in decode + [prefill]:
    a, b = (r['median_tokens_per_second'][v] for v in 'AB')
    lines.append(f"  {r['name']}: {a:.6f} -> {b:.6f} tok/s、{r['gain_percent']:+.3f}%")
lines += [
    '生成: 入力0・生成32・深さ0、12スレッド、各プロセス3回。ABBAとBAABを実施。',
    '入力: 512トークン、24スレッド、各プロセス1回。ABBAを実施。batch/ubatchは512。',
    '比較単位は独立したプロセスの平均。各順序で各構成2プロセス。全てHDD読み出し0。', '',
    '日本語の実応答（合成ベンチとは別の結果）',
    f"  59入力トークン・同じ24評価トークン。ABBA {model['generation_gain_percent']:+.3f}%、BAAB {reverse['generation_throughput_gain_percent']:+.3f}%。",
    f"  8プロセスをまとめた生成時間の中央値: {response_medians['A']:.2f} -> {response_medians['B']:.2f} ms、速度差 {response['gain_percent']:+.3f}%。",
    '  回答内容は全回同一。BLUE-731、東京、42を確認。一行形式への厳密な追従は判定していない。',
    '  応答速度の差は順序によって正負が変わるため、まとめた中央値だけから改善を断定しない。',
    '  一つの固定プロンプトの結果であり、全ての会話が合成ベンチと同じ率で速くなるとは言えない。', '',
    '正しさ',
    '  通常利用向け版の行列80ケースで、現行版と出力がビット単位で一致。量子化の既存テストも通過。',
    '  固定160入力トークン、位置126-159の34箇所 x 129280語彙 = 4,395,520個のfloatが全て一致。',
    '  logitsファイルSHA256: ' + model['logits'][0]['sha256'],
    '  先読み単体の境界確認は1560個の異なる候補・入力組合せ。短い行・奇数ブロック・guard pageを含む。',
    '  guard pageは通常のロード境界の確認。先読みポインタ自体の境界はソースでも確認した。', '',
    '採用しなかった同期の案',
    f"  既存C11バリアをOpenMPチーム内で使う案: 軽い監視では {lookup['light_native-barrier']['gain_percent']:+.3f}%。",
    '  従来の監視では-0.170%。安定した改善を確認できず不採用。同期16ケース、MoE96ケースは通過。',
    f"  GOMP_SPINCOUNT=10000: 軽い監視の生成は {lookup['light_spin10000']['gain_percent']:+.3f}%。通常設定には採用しない。",
    '  従来の監視では生成-0.072%、入力+0.012%。観察区間のCPU使用量は生成約7.5%、入力約3.6%減少。',
    '  CPU使用時間は電力ではない。速度向上とCPU上で待つ時間の減少は別に扱う。', '',
    '測定方法の修正',
    '  従来のbench_deepseek.pyは/proc/PID/numa_mapsを毎秒読み取っていた。',
    '  Q8の逆順比較の点検時、監視プロセスに35.9%の累積平均CPU使用率が見えた。',
    '  新しいbench_deepseek_light.pyではこの読み取りを外し、CPU・RSS・HDD・fault・affinityの記録を維持した。',
    '  新監視のCPU時間は約43-44秒の初回生成測定あたり約0.05秒。',
    '  同じ通常版の速度も従来の約2.4台から約2.58 tok/sへ上がった。これは観測負荷の減少であり、LLM本体の改善率に数えない。',
    '  本レポートの主要比較はA/B双方を軽い監視で測定している。古い監視の結果・スクリプトも保存した。',
    '  NUMA配置の詳細観察と速度比較は今後も分ける。新しい速度比較はライブのページ配置を採取していない。', '',
    '単体スクリーニングと解釈',
    '  Q8先読み8ブロック: K1024とK4096では1コアで約8.5%/13.7%、12コアで約1.8%/1.3%改善。',
    '  行列行数はキャッシュより大きくするため65536/16384を使用。単体値をモデル全体の改善率には換算しない。',
    '  他の先読み距離16/32も単体で比較。距離8は両方の主要Kで改善を示したため実モデルへ進めた。',
    '  同期の置換だけでは改善が安定しなかった。待ち合わせ関数の費用と仕事の終わる時刻の差は同じではない。',
    '  Q8変更後のPEBS/PMU内訳は再計測していないため、メモリ待ちの削減割合は断定しない。', '',
    '保存と再実行',
    '  通常起動: /home/codex/projects/llm-lab/run_deepseek_flash.sh',
    '  旧版起動: /home/codex/projects/llm-lab/run_deepseek_flash_before_prefetch.sh',
    '  再ビルド: /home/codex/projects/llm-lab/rebuild_deepseek.sh',
    '  新しい比較: run_cpu_ab_light.py。全コマンド、ログ、ハッシュ、出力比較範囲はREPORT.jsonを参照。',
    '  元のbuild-avx-deepとllama-privateは保存。新しい版はbuild-avx-prefetchとllama-private-prefetch。',
    f"  終了時に旧ソース/バイナリのハッシュを照合。OSの{len(settings)}設定は{'変更なし' if report['settings_unchanged'] else '差分あり。REPORT.jsonを確認'}。",
    '  測定前に拡張したテストの16MiB arenaが不足した失敗は、128MiBに修正して再検証。失敗ログも保存した。', '',
    '限界',
    '  小さい合成生成と一つの固定日本語応答の結果。HDDコールドロード、RAM増設、GPU、他モデル、長文脈全般、消費電力は未評価。',
    '  普遍的な速度向上やメモリ帯域ボトルネックの解消を示す結果ではない。',
    '  OpenMP待機設定の仕様: https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html',
]
(out / 'REPORT_JA.txt').write_text('\n'.join(lines) + '\n')
print(json.dumps({'saved': str(out / 'REPORT.json'), 'combined_response_gain_percent': report['combined_response']['gain_percent'],
    'baseline_unchanged': all(baseline_check.values())}, indent=2))
