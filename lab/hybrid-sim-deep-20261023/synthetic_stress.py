#!/usr/bin/env python3
"""Synthetic locality stress, not a predicted distribution of real model routes.

Clone completed central smoke configurations, preserve all hardware assumptions,
and vary only routed expert identifiers. Never compile or edit the simulator.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import runpy
import shutil
import struct
import subprocess
import sys

SRC=Path(__file__).resolve().parent
OUT=SRC.parent/'results/hybrid-sim-deep-20261023'
DEST=OUT/'synthetic'
RECORD=struct.Struct('<II240H40B')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_uniform(seed,working_set,prefill,decode):
    folder=DEST/'routes'/f'uniform_w{working_set:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    subsets=[];generators=[]
    for layer in range(40):
        rng=random.Random(seed+working_set*65537+layer*257)
        subsets.append(sorted(rng.sample(range(384),working_set)))
        generators.append(random.Random(seed*1000003+working_set*997+layer))
    blob=bytearray()
    for step in range(prefill+decode):
        ids=[expert for layer in range(40) for expert in generators[layer].sample(subsets[layer],6)]
        assert all(len(set(ids[6*l:6*(l+1)]))==6 for l in range(40))
        blob.extend(RECORD.pack(step,step,*ids,*([0]*40)))
    route=folder/'route.bin';phase=folder/'phases.tsv'
    route.write_bytes(blob);phase.write_text('0\t0\n'*prefill+'1\t0\n'*decode)
    meta={'name':f'uniform_w{working_set:03d}','kind':'synthetic_uniform_fixed_working_set',
          'seed':seed,'working_set_per_layer':working_set,'total_working_experts':working_set*40,
          'prefill_steps':prefill,'decode_steps':decode,'working_expert_ids':subsets,
          'construction':'Each layer has a fixed random subset of W of its 384 experts. Each step independently selects 6 unique IDs uniformly from that subset. Layer RNG streams are separate.',
          'original_logged_hits':'zero placeholders; not hardware observations',
          'route':str(route),'phases':str(phase),'route_sha256':sha(route),'phases_sha256':sha(phase)}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def make_permuted(seed,base):
    folder=DEST/'routes'/'turn_permuted_base';folder.mkdir(parents=True,exist_ok=True)
    original=Path(base['trace']).read_bytes();records=list(RECORD.iter_unpack(original))
    phases=[tuple(map(int,line.split())) for line in Path(base['phases']).read_text().splitlines() if line.strip()]
    assert len(records)==len(phases)
    permutations={}
    for turn in sorted({t for d,t in phases}):
        permutations[turn]=[]
        for layer in range(40):
            order=list(range(384))
            if turn:
                random.Random(seed*1000003+turn*1009+layer).shuffle(order)
            permutations[turn].append(order)
    blob=bytearray();identical_first_turn=0
    for record,(decode,turn) in zip(records,phases):
        ids=[permutations[turn][layer][record[2+6*layer+j]] for layer in range(40) for j in range(6)]
        if turn==0:
            assert ids==list(record[2:242]);identical_first_turn+=1
        assert all(len(set(ids[6*l:6*(l+1)]))==6 for l in range(40))
        blob.extend(RECORD.pack(record[0],record[1],*ids,*([0]*40)))
    route=folder/'route.bin';phase=folder/'phases.tsv'
    route.write_bytes(blob);shutil.copyfile(base['phases'],phase)
    meta={'name':'turn_permuted_base','kind':'synthetic_turnwise_id_permutation','seed':seed,
          'construction':'Identity mapping in turn 0. Every later turn and layer uses an independent fixed permutation of 384 expert IDs; within-turn access locality and per-layer request sets are preserved. Cross-turn ID reuse is randomized, not forced to zero.',
          'original_trace':base['trace'],'original_trace_sha256':sha(base['trace']),
          'prefill_steps':sum(not d for d,t in phases),'decode_steps':sum(d for d,t in phases),
          'identical_first_turn_steps':identical_first_turn,'route':str(route),'phases':str(phase),
          'route_sha256':sha(route),'phases_sha256':sha(phase),
          'original_logged_hits':'zero placeholders; old cache hit counters cannot apply to relabeled experts'}
    (folder/'permutations.json').write_text(json.dumps(permutations)+'\n')
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def prepare(args):
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'details').mkdir(exist_ok=True)
    central=json.loads((OUT/'smoke-configs.json').read_text())
    cards={c['card']:c for c in central};assert set(cards)=={'P40','V100'}
    assert all(c['policy']==3 for c in cards.values())
    meta=[make_uniform(args.seed,w,args.prefill,args.decode) for w in [6,12,14,18,32,384]]
    meta.append(make_permuted(args.seed,cards['P40']))
    cases=[]
    for trace in meta:
        for card in ['P40','V100']:
            c=copy.deepcopy(cards[card]);c.update(id=trace['name']+'_'+card,trace=trace['route'],phases=trace['phases'],detail=1)
            cases.append(c)
    order=runpy.run_path(str(SRC/'run_deep.py'))['ORDER']
    assert all(set(c)==set(order) for c in cases)
    (DEST/'synthetic-configs.json').write_text(json.dumps(cases,indent=2)+'\n')
    (DEST/'synthetic-configs.tsv').write_text(''.join('\t'.join(str(c[k]) for k in order)+'\n' for c in cases))
    manifest={'seed':args.seed,'python_version':sys.version,'central_config_sha256':sha(OUT/'smoke-configs.json'),
              'source_generator_sha256':sha(__file__),'traces':meta,'central_slots':{k:c['slots'] for k,c in cards.items()},
              'scope':'Deliberately constructed locality regimes. These are not samples from the real DeepSeek routing distribution and cannot be assigned occurrence probabilities.'}
    (DEST/'synthetic-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return cases,manifest


def snapshot_completed_binary():
    smoke=json.loads((OUT/'smoke-status.json').read_text())
    assert smoke['exit_code']==0,'smoke must finish successfully first'
    sources={p.name:sha(p) for p in [SRC/'deep-sim.cpp',SRC/'resources.hpp',SRC/'cache.hpp',SRC/'run_deep.py']}
    assert all(smoke['source_sha256'].get(name)==digest for name,digest in sources.items()),'source changed since completed smoke; wait for current smoke to finish'
    original=SRC/'deep-sim';digest=sha(original);binary=DEST/'deep-sim-used'
    shutil.copyfile(original,binary);binary.chmod(0o755)
    assert sha(binary)==digest==sha(original),'simulator executable changed during snapshot'
    return binary,{'binary_sha256':digest,'source_sha256':sources,'smoke_status_sha256':sha(OUT/'smoke-status.json')}


def check_first_turn_prefix():
    checks=[]
    for card in ['P40','V100']:
        original=list(csv.DictReader((OUT/'details'/f'smoke_{card}.csv').open()))
        changed=list(csv.DictReader((DEST/'details'/f'turn_permuted_base_{card}.csv').open()))
        a=[r for r in original if int(r['turn'])==0]
        b=[r for r in changed if int(r['turn'])==0]
        assert a==b,'future turn relabeling changed an identical first-turn prefix'
        checks.append({'card':card,'first_turn_steps':len(a),'all_detail_columns_identical':True})
    return checks


def report(manifest):
    steady=json.loads((DEST/'steady-synthetic.json').read_text())
    table={}
    for row in steady['results']:
        if row['turn']==-1:
            table[(row['id'],row['selection'])]=row
    lines=['合成ルーティングの局所性ストレステスト','2026-10-23 日本時間','',
           'これは実モデルのroute分布の予測ではない。局所性が変わった時に中央ハードウェア仮定で勝者が変わるかを調べる構成実験。',
           '中央smoke設定のGPU/CPU/PCIe/容量/量子化/200W時の性能仮定は全て固定し、routeだけを変更。GPU実測ではない。',
           f"実際のslot数: P40={manifest['central_slots']['P40']}、V100={manifest['central_slots']['V100']}。",'',
           '構成','各層384expertのうち固定されたW個から毎token6個を重複なし一様抽出。層別RNG、固定seed。全40層。',
           'prefill256+decode2048。集合はrun中変わらず、長期に持続する局所性を意図的に仮定。',
           'W=6は両GPUが全working setを保持、W=12/14はP40だけ全保持、W=18はP40容量境界付近、W=32/384は両方missを生む領域。',
           '現在/過去のrouteだけでEMAを更新。future routeを用いた配置はしない。',
           '別ケースはbase traceのturn0を保持し、以降はturn毎・layer毎にexpertIDを固定random permutation。',
           'これによりturn内の局所性を保ってturn間の再利用をランダム化する。再利用を厳密にゼロにはしていない。','',
           '結果: 各turn最初32decode除外、ただしcache履歴は保持','case P40_tok/s V100_tok/s P40/V100 P40_hit% V100_hit% P40_copyMB/token V100_copyMB/token']
    for trace in manifest['traces']:
        a=table[(trace['name']+'_P40','after_32_per_turn')];b=table[(trace['name']+'_V100','after_32_per_turn')]
        lines.append(f"{trace['name']:20} {a['tok_s']:8.3f} {b['tok_s']:8.3f} {a['tok_s']/b['tok_s']:7.3f} {a['ready_fraction']*100:7.2f} {b['ready_fraction']*100:7.2f} {a['copy_MB_token']:8.2f} {b['copy_MB_token']:8.2f}")
    lines+=['','全decodeもsteady-synthetic.tsv/jsonに保存。先頭除外だけを都合よく採用しない。',
            'Wはその会話中に使われる集合の大きさであり、実モデルのexpertを削除したモデルサイズではない。',
            '実routeでこれらの小さなworking set状態が何割発生するかはこの実験から分からない。',
            '同じseedの各Wは異なる構成条件であり統計的なGPU勝率の推定には使わない。',
            '元base traceは9月版であり1017の実測ではない。permutation後のtraceはさらに合成である。',
            '最初のturnが元smokeと全detail列で一致することを確認し、後のturnの変更を先読みしていないことを検証。',
            '再実行: python3 synthetic_stress.py（中央smoke完了とsource一致が必要）。--prepare-onlyで入力だけ生成。']
    (DEST/'REPORT_JA.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[14:26]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--prefill',type=int,default=256)
    parser.add_argument('--decode',type=int,default=2048)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    assert args.prefill>=0 and args.decode>32
    cases,manifest=prepare(args)
    if args.prepare_only:
        print('Prepared',len(cases),'cases in',DEST);return
    binary,provenance=snapshot_completed_binary()
    command=[str(binary),str(DEST/'synthetic-configs.tsv'),str(DEST/'synthetic-results.tsv'),str(DEST/'details')]
    with (DEST/'synthetic-log.txt').open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    status={'command':command,'exit_code':code,'cases':len(cases),**provenance,
            'config_sha256':sha(DEST/'synthetic-configs.json')}
    (DEST/'synthetic-status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert code==0
    checks=check_first_turn_prefix()
    subprocess.run([sys.executable,str(SRC/'analyze_steady.py'),'--configs',str(DEST/'synthetic-configs.json'),
                    '--output-prefix',str(DEST/'steady-synthetic')],check=True)
    (DEST/'prefix-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    report(manifest)


if __name__=='__main__':main()
