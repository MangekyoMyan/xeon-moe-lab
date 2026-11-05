#!/usr/bin/env python3
"""Compare all decode tokens with a prespecified per-turn startup exclusion.

The first N decode tokens still execute and update the cache; they are excluded
only from this reporting view. This is not an independent workload or proof of
stationarity. Per-token simulator logs have reduced decimal precision.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
from collections import defaultdict

RECORD = struct.Struct('<II240H40B')
FIELDS = ['id','card','trace_name','selection','turn','steps','excluded_decode_steps',
          'ms_token','tok_s','ready_fraction','gpu_fraction','cpu_experts_token',
          'copy_MB_token','copies_token','input_wait_ms','return_wait_ms','p50_ms','p95_ms']


def expected_layout(config, cache):
    identity=(config['trace'],config['phases'])
    if identity not in cache:
        raw=Path(config['trace']).read_bytes()
        if len(raw)%RECORD.size:
            raise ValueError('partial routing record')
        route=list(RECORD.iter_unpack(raw))
        phases=[tuple(map(int,line.split())) for line in Path(config['phases']).read_text().splitlines() if line.strip()]
        if len(route)!=len(phases):
            raise ValueError('route/phase length mismatch')
        if any(len(phase)!=2 or phase[0] not in (0,1) or not 0<=phase[1]<100 for phase in phases):
            raise ValueError('invalid phase entry')
        if any(b[0]!=a[0]+1 for a,b in zip(route,route[1:])):
            raise ValueError('non-contiguous routing step ids')
        cache[identity]=[(r[0],p[1],p[0]) for r,p in zip(route,phases)]
    return cache[identity]


def read_detail(path, expected):
    with path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(expected):
        raise ValueError(f'incomplete detail: {len(rows)} rows, expected {len(expected)}')
    converted=[]
    for row,want in zip(rows,expected):
        out={k:int(row[k]) for k in ['step','turn','decode','ready','gpu','cpu','copies']}
        out.update({k:float(row[k]) for k in ['ms','weight_MB','input_wait_ms','return_wait_ms']})
        if (out['step'],out['turn'],out['decode'])!=want:
            raise ValueError(f'phase/step mismatch at {out["step"]}')
        if not 0<=out['gpu']<=out['ready']<=240 or out['cpu']+out['gpu']!=240:
            raise ValueError('expert accounting mismatch')
        if any(not math.isfinite(v) or v<0 for v in out.values()) or out['ms']<=0:
            raise ValueError('invalid detail metric')
        converted.append(out)
    return converted


def aggregate(config, selection, turn, rows, original_count):
    if not rows:
        return None
    n=len(rows);time=sum(r['ms'] for r in rows);lat=sorted(r['ms'] for r in rows)
    q=lambda p:lat[int(p*(n-1))]
    return {'id':config['id'],'card':config['card'],'trace_name':Path(config['trace']).parent.name,
            'selection':selection,'turn':turn,'steps':n,'excluded_decode_steps':original_count-n,
            'ms_token':time/n,'tok_s':1000*n/time,
            'ready_fraction':sum(r['ready'] for r in rows)/(240*n),
            'gpu_fraction':sum(r['gpu'] for r in rows)/(240*n),
            'cpu_experts_token':sum(r['cpu'] for r in rows)/n,
            'copy_MB_token':sum(r['weight_MB'] for r in rows)/n,
            'copies_token':sum(r['copies'] for r in rows)/n,
            'input_wait_ms':sum(r['input_wait_ms'] for r in rows)/n,
            'return_wait_ms':sum(r['return_wait_ms'] for r in rows)/n,
            'p50_ms':q(.5),'p95_ms':q(.95)}


def selections(rows, discard):
    by_turn=defaultdict(list)
    for r in rows:
        if r['decode']:
            by_turn[r['turn']].append(r)
    views={'all_decode':{},f'first_{discard}_per_turn':{},f'after_{discard}_per_turn':{}}
    for turn,items in sorted(by_turn.items()):
        views['all_decode'][turn]=items
        views[f'first_{discard}_per_turn'][turn]=items[:discard]
        views[f'after_{discard}_per_turn'][turn]=items[discard:]
    return by_turn,views


def assert_full_summary(rows, reference, config_id):
    diffs=[]
    lookup={(r['id'],int(r['turn'])):r for r in reference}
    keys=['ms_token','ready_fraction','gpu_fraction','cpu_experts_token','copy_MB_token',
          'copies_token','input_wait_ms','return_wait_ms']
    for row in rows:
        if row['selection']!='all_decode':
            continue
        original=lookup.get((config_id,row['turn']))
        if original is None:
            raise ValueError(f'no finished summary row for {config_id}, turn {row["turn"]}')
        if row['steps']!=int(original['steps']):
            raise ValueError('detail and summary token count mismatch')
        for key in keys:
            expected=float(original[key]);diff=abs(row[key]-expected)
            # Six-significant-digit token logs vs twelve-digit final totals.
            tolerance=max(0.0002,abs(expected)*0.00002)
            if diff>tolerance:
                raise ValueError(f'detail and summary disagree: {config_id}/{row["turn"]}/{key}: {diff}')
            diffs.append({'turn':row['turn'],'metric':key,'absolute_difference':diff})
    return diffs


def selftest():
    cfg={'id':'fixture_P40','card':'P40','trace':'/fixture/test/route.bin'}
    rows=[{'step':i,'turn':t,'decode':d,'ms':ms,'ready':120,'gpu':90,'cpu':150,'copies':2,
           'weight_MB':10,'input_wait_ms':1,'return_wait_ms':2}
          for i,(t,d,ms) in enumerate([(0,0,100),(0,1,10),(0,1,20),(0,1,30),(1,0,100),(1,1,40),(1,1,50)])]
    by_turn,views=selections(rows,1)
    after=[r for group in views['after_1_per_turn'].values() for r in group]
    assert [r['ms'] for r in after]==[20,30,50]
    out=aggregate(cfg,'after_1_per_turn',-1,after,5)
    assert out['steps']==3 and out['excluded_decode_steps']==2
    assert out['tok_s']==30 and out['ms_token']==100/3
    assert out['ready_fraction']==.5 and out['gpu_fraction']==.375
    assert out['cpu_experts_token']==150
    assert aggregate(cfg,'empty',0,[],3) is None
    assert not selections(rows,99)[1]['after_99_per_turn'][0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',nargs='+',type=Path,required=True)
    parser.add_argument('--details-dir',type=Path)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--discard',type=int,default=32)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    if args.discard<0:
        parser.error('--discard must be nonnegative')
    selftest()
    outputs=[];validated=[];unavailable=[];layout_cache={};seen=set();sources=[]
    for config_path in args.configs:
        configs=json.loads(config_path.read_text())
        summary_path=config_path.with_name(config_path.name.replace('-configs.json','-results.tsv'))
        reference=list(csv.DictReader(summary_path.open(),delimiter='\t')) if summary_path.exists() else []
        sources.append({'config_path':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        'summary_path':str(summary_path),
                        'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.exists() else None})
        for config in configs:
            if not config.get('detail'):
                continue
            identity=config['id']
            if identity in seen:
                raise ValueError(f'duplicate detailed scenario: {identity}')
            seen.add(identity)
            path=(args.details_dir or config_path.parent/'details')/(identity+'.csv')
            try:
                expected=expected_layout(config,layout_cache)
                rows=read_detail(path,expected)
                by_turn,views=selections(rows,args.discard)
                original=sum(map(len,by_turn.values()))
                local=[]
                for label,view in views.items():
                    all_rows=[r for group in view.values() for r in group]
                    summary=aggregate(config,label,-1,all_rows,original)
                    if summary:local.append(summary)
                    for turn,items in view.items():
                        summary=aggregate(config,label,turn,items,len(by_turn[turn]))
                        if summary:local.append(summary)
                differences=assert_full_summary(local,reference,identity)
                validated.append({'id':identity,'route_records':len(rows),'decode_steps':original,
                                  'detail_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'max_full_summary_difference':max(d['absolute_difference'] for d in differences),
                                  'turns_without_remaining_decode':[t for t,items in by_turn.items() if len(items)<=args.discard]})
                outputs.extend(local)
            except (ValueError,KeyError,FileNotFoundError) as error:
                if not args.allow_incomplete:
                    raise
                unavailable.append({'id':identity,'reason':str(error)})
    pairs=[]
    groups=defaultdict(dict)
    for row in outputs:
        if row['turn']==-1:
            family=row['id'].rsplit('_',1)[0]
            groups[(family,row['selection'])][row['card']]=row
    for (family,label),cards in sorted(groups.items()):
        if 'P40' in cards and 'V100' in cards:
            p,v=cards['P40'],cards['V100']
            pairs.append({'family':family,'selection':label,'P40_tok_s':p['tok_s'],'V100_tok_s':v['tok_s'],
                          'P40_over_V100_speed':p['tok_s']/v['tok_s'],
                          'P40_minus_V100_ms':p['ms_token']-v['ms_token']})
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    csv_path=Path(str(args.output_prefix)+'.tsv')
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,delimiter='\t');writer.writeheader();writer.writerows(outputs)
    json_path=Path(str(args.output_prefix)+'.json')
    json_path.write_text(json.dumps({'discard_first_decode_per_turn':args.discard,'selftest':'passed',
        'method':'Keep original cache evolution; exclude first N decode tokens of EACH turn only from metric aggregation. All prefill tokens remain executed but unscored.',
        'limitations':['This is a prespecified startup-sensitivity view, not a proof that the retained tokens form a stationary process.',
                       'Show all-decode and exclusion views together; users pay startup latency too.',
                       'Three routing runs share the same eight prompts; different outputs do not create three independent workloads.',
                       'Long-context scenarios reuse short-context routes; exclusion does not validate long-context routing.',
                       'Per-token CSV precision is lower than final simulator totals; full-view agreement is tolerance-checked.'],
        'sources':sources,'validated_details':validated,'unavailable':unavailable,'comparisons':pairs,'results':outputs},indent=2)+'\n')
    print(json.dumps({'validated_scenarios':len(validated),'unavailable':len(unavailable),'rows':len(outputs),
                      'tsv':str(csv_path),'json':str(json_path)},indent=2))
    for p in pairs:
        if p['selection'] in ['all_decode',f'after_{args.discard}_per_turn']:
            print(p['family'],p['selection'],'P40',round(p['P40_tok_s'],3),'V100',round(p['V100_tok_s'],3),
                  'ratio',round(p['P40_over_V100_speed'],4))


if __name__=='__main__':
    main()
