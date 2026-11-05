#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import statistics as st

R=Path(__file__).resolve().parent;LAB=R.parent;OLD=LAB/'strata-tc-study-20261031'
def load(p):return json.loads(p.read_text())
def metrics(root,label):
    j=load(root/'results'/label/'result.json');assert j['success'];s=j['stats']
    return {'label':label,'prefill_s':s['XV_PREFILL'][0]['seconds'],'prefill_tps':s['XV_PREFILL'][0]['tps'],
            'steady_decode_tps':s['XV_TIMING'][0]['decode_tps'],
            'steady_decode_records':s['XV_TIMING'][0]['records']-s['XV_TIMING'][0]['skip_records'],
            'first_decode_s':s['XV_RECORD'][1]['seconds'],
            'all_decode_s':s['XV_ALL_TIMING'][0]['decode_seconds'],
            'all_decode_tps':s['XV_ALL_TIMING'][0]['decode_steps']/s['XV_ALL_TIMING'][0]['decode_seconds'],
            'io':s['XV_PREFILL_IO'][0],'total_process_wall_s':j['finished']-j['started'],
            'resources':j['resources']}
state=load(R/'STATE.json');assert state['phase']=='measurements_complete'
result={'scope':'GLM4.5Air Q8, existing frozen runtimes, before128GiB vs after512GiB; GPU150W/ECC; context131072; cache9216MiB',
        'long':{},'individual':{},'correctness':{},'maxima':{},'api':{},'preload':load(R/'preload-state.json')}
groups={'baseline':(['final-a0','final-a1'],['long-a0','long-a1']),
        'phase-v3':(['v3-long'],['long-b0','long-b1'])}
for kind,(a,b) in groups.items():
    before=[metrics(OLD,label) for label in a];after=[metrics(R,label) for label in b]
    row={'before':before,'after':after}
    for k in ('prefill_s','prefill_tps','steady_decode_tps','all_decode_tps','first_decode_s'):
        row[k]={'before_mean':st.mean(v[k] for v in before),'after_mean':st.mean(v[k] for v in after),
                'after_min':min(v[k] for v in after),'after_max':max(v[k] for v in after)}
    row['prefill_speedup']=row['prefill_s']['before_mean']/row['prefill_s']['after_mean']
    row['steady_decode_speedup']=row['steady_decode_tps']['after_mean']/row['steady_decode_tps']['before_mean']
    result['long'][kind]=row
for label,old in [('ja1024-a','ja1024-a'),('ja1024-b','ja1024-b'),('short-b','v3-short')]:
    result['individual'][label]={'before':metrics(OLD,old),'after':metrics(R,label)}
samples=[]
for row in state['runs']:
    label=row['label'];q=load(R/'results'/label/'historical-logit-comparison.json');assert q['all_logits_bitexact']
    result['correctness'][label]=q
    health=load(LAB/'results'/R.name/label/'run.json');assert health['returncode']==0 and not health.get('error')
    samples.extend(health['samples'])
    assert health['gpu_after']['ecc_single']==0 and health['gpu_after']['ecc_double']==0
result['maxima']={'gpu_C':max(s['gpu']['gpu_C'] for s in samples),'hbm_C':max(s['gpu']['hbm_C'] for s in samples),
    'cpu_C':max(v for s in samples for v in s['cpu_C'].values()),
    'fan_percent':max(v for s in samples if 'fan_duty' in s for v in s['fan_duty']['percent']),
    'vram_MiB':max(s['gpu']['vram_used_MiB'] for s in samples)}
result['stream_prefill_breakdown']={}
for root,label in [(OLD,'v3-long'),(R,'long-b0'),(R,'long-b1')]:
    s=load(root/'results'/label/'result.json')['stats']
    result['stream_prefill_breakdown'][label]={
        'stream':next(x for x in s['XV_STREAM'] if x['phase']=='prefill' and x['operations']),
        'worker_quantize_ms_including_decode':sum(x['quantize_ms'] for x in s['XV_EXACT'])}
api=load(R/'API_SMOKE.json');assert api['success']
oldapi=load(OLD/'API_SMOKE.json')
result['api']['requests']=[{k:q.get(k) for k in ['name','first_content_s','total_s','content','timings','usage','majflt','disk_read_bytes']} for q in api['requests']]
result['api']['historical_requests']=[{k:q.get(k) for k in ['name','first_content_s','total_s','content','timings','majflt','disk_read_bytes']} for q in oldapi['requests']]
result['api']['production_config_changed']=False
result['api']['temporary_supervisor_returncode']=api['supervisor_returncode']
result['api']['temporary_child_returncode']=api['health'].get('child_returncode')
result['limitations']=['Hardware before/after comparison, not a DIMM swap A/B in one boot.',
    'Same fixed input tokens and 31 teacher-forced decode steps for main comparison; steady metric uses last16.',
    'Context allocation131072, actual main input2048. This is not a128k prompt benchmark.',
    'V3 historical main comparison has one run, present has two; baseline has two in each.',
    'All logits compare each runtime to itself across RAM upgrade, not TC rounding to exact baseline.',
    'Local API timings exclude main-PC network and OpenWebUI processing.',
    'HDD preload separate from inference. Model initialization/startup separate from prompt timings.',
    'Expert cache stays9GiB in both arms to isolate RAM effects; productionAPI uses10GiB and remains unchanged.']
(R/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['512GB増設後：既存GLM-4.5-Air Q8の実モデル測定（2026-10-31）','',
       '同じモデル・同じ実行バイナリ・同じプロンプト・同じメモリ配置を使用。GPU150W/ECC有効。',
       '従来版と、前回検証した入力処理高速化版phase-v3を、それぞれ128GB時と比較した。',
       'コンテキスト枠128k、Q8 KV、expertキャッシュ9GiB、入力24/生成12スレッド。',
       '主試験の実際の入力は2048トークン。128kトークンの長文処理速度は未測定。','',
       '主比較：入力2048トークン＋固定された後続31トークン（teacher forcing）',
       '生成速度は最後の16ステップの平均。ルーティングが同じになるよう、モデルが選んだ文ではなく同じ後続トークンを与えた。',
       '                         入力処理秒 128→512GB     入力速度比    安定生成tok/s 128→512GB    生成速度比']
for kind,row in result['long'].items():
    p=row['prefill_s'];t=row['steady_decode_tps']
    lines.append(f"{kind:<16} {p['before_mean']:.3f} → {p['after_mean']:.3f}     {row['prefill_speedup']:.3f}x       {t['before_mean']:.3f} → {t['after_mean']:.3f}       {row['steady_decode_speedup']:.3f}x")
    lines.append('  今回の個別値: '+str([{k:v[k] for k in ['label','prefill_s','steady_decode_tps','first_decode_s']} for v in row['after']]))
lines+=['','日本語1024トークンと短文64トークン（各1回、各自の増設前と比較）']
for label,row in result['individual'].items():
    o=row['before'];n=row['after']
    lines.append(f"{label}: 入力 {o['prefill_s']:.3f} → {n['prefill_s']:.3f}秒、生成指標 {o['steady_decode_tps']:.3f} → {n['steady_decode_tps']:.3f} tok/s")
lines+=['短文64の生成指標は元ハーネスの仕様上最後の1ステップのみ。安定速度の結論には主試験を使う。','',
        '実際のAPI応答（サーバー内loopback、cache_prompt=false、思考表示off、温度0）']
oldrequests={q['name']:q for q in result['api']['historical_requests']}
for q in result['api']['requests']:
    old=oldrequests.get(q['name']);t=q['timings']
    lines.append(f"{q['name']}: 入力{t['prompt_n']}トークン、最初の文字まで{q['first_content_s']:.3f}秒、入力{t['prompt_per_second']:.3f}tok/s、生成{t['predicted_per_second']:.3f}tok/s")
    if old:lines.append(f"  増設前の最初の文字まで: {old['first_content_s']:.3f}秒")
    lines.append('  応答: '+q['content'])
lines+=['初めの3リクエストは前回と同じ。最後の日本語説明は追加の自由生成確認（最大128トークン）。',
        '挨拶や計算の回答は生成数が少ないため、その生成tok/sを長い応答の代表値にはしない。','',
        'HDDと出力検証',
        f"初回の順次読み込み {result['preload']['bytes_read']/1e9:.3f}GB / {result['preload']['elapsed_s']:.1f}秒。推論計時には含まない。",
        'RAM常駐とNUMA配置を確認後に計時。全prefill観測窓とAPIリクエストでHDD読出し/major fault=0を確認。',
        f"7件の固定トークン比較で合計{sum(q['records'] for q in result['correctness'].values())}位置の全語彙logitsが、それぞれの増設前出力とbit単位で一致。",
        '高速化版と従来版の相互一致を主張するものではない。高速化版のTensorCore計算には従来から丸め差がある。','',
        '監視値と終了状態',str(result['maxima']),
        'GPU150W/ECC有効、Standardファンモードと70%上限を維持。GPU/ECC/PCIe/CPU熱制限の変化はガードで監視。',
        '元のAPIサービスは開始時inactive。公開設定は変更せず、試験用loopbackサーバーは終了時に停止。','',
        '比較の限界',*result['limitations'],'',
        '詳細: SUMMARY.json、STATE.json、binary-baselines.json、results/*/result.json、各logits.bin。',
        f'温度等の連続記録: {LAB/"results"/R.name}',
        '実行ドライバー: execute.py、run_model.py、watch.py。']
(R/'REPORT_JA.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ['long','maxima']},indent=2))
