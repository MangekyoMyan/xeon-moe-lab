#!/usr/bin/env python3
"""Prepare calibrated inputs and explicit hypothetical GPU scenarios."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

SRC = Path(__file__).resolve().parent
LAB = SRC.parent
OUT = LAB / 'results/hybrid-sim-20261023'
EVIDENCE = LAB / 'results/expert-cache-evidence-20261022'
TRACE = EVIDENCE / 'cachedmoe/bench/results/hitrate/base_auto/route.bin'
ORDER = ['id','card','policy','slots','budget','queue','decay','margin','min_heat','cpu_scale',
         'host_GBs','gpu_GBs','common_ms','pcie_GBs','contention','gpu_contention','sync_us',
         'launch_us','parallel_common','instant','detail']


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/'details').mkdir(exist_ok=True)
    meta = json.loads((EVIDENCE/'cachedmoe/bench/results/hitrate/base_auto/turns.json').read_text())
    phases = []
    for i,t in enumerate(meta['turns']):
        phases += [(0,i)]*t['prefill_tokens'] + [(1,i)]*t['decode_steps']
    assert len(phases)*528 == TRACE.stat().st_size
    (OUT/'phases.tsv').write_text(''.join(f'{p}\t{t}\n' for p,t in phases))
    samples = [json.loads(l) for l in (OUT/'cpu-calibration/stdout.txt').read_text().splitlines()]
    assert samples[0]['validation']
    calibration = {}
    for threads in [12,24]:
        timings = [0.0] + [statistics.median(r['ms_per_layer'] for r in samples
                    if r.get('threads') == threads and r.get('experts') == n) for n in range(1,7)]
        gops = {}
        for k in [5120,2304]:
            rows = [json.loads(l) for l in (OUT/f'compute-k{k}-t{threads}/stdout.txt').read_text().splitlines()]
            gops[k] = statistics.median(2*r['Gweights_s'] for r in rows if 'Gweights_s' in r)
        one = (2*2*5120*2304/gops[5120] + 2*5120*2304/gops[2304])/1e6
        floor = [n*one for n in range(7)]
        assert all(f <= t for f,t in zip(floor,timings))
        calibration[threads] = {'ms':timings,'cache_GOPS':gops,'compute_floor_ms':floor}
    (OUT/'cpu-times.tsv').write_text(' '.join(map(str,calibration[24]['ms']))+'\n'+
                                   ' '.join(map(str,calibration[24]['compute_floor_ms']))+'\n')
    (OUT/'calibration-summary.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (OUT/'inputs.json').write_text(json.dumps({
        'trace':str(TRACE),'trace_sha256':hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        'source_commit':'84ba1ab723b1397e04e5532bdd50e8e0668594b2',
        'client_date':'2026-10-23','trace_release':'September V4.1, not 1017',
        'total_steps':len(phases),'decode_steps':sum(p for p,t in phases),
        'turn_labels':[t['label'] for t in meta['turns']],
        'CPU_calibration':calibration,
        'measured':['CPU 1..6 MXFP4 expert dot-product times at actual H5120/F2304 geometry',
                    'CPU cache-resident kernel rates used as a compute-ceiling proxy',
                    'Public real route sequence and original 4500-slot hits'],
        'assumed':['Effective GPU expert throughput and non-expert GPU milliseconds at 200W are not measured',
                   '8/10/12GiB total non-expert VRAM reserves',
                   'PCIe 6/12GB/s plus 40us per CPU/GPU activation handoff',
                   'DRAM memory-side scaling 1..2, constrained by a measured compute-ceiling proxy',
                   'DMA contention is a sensitivity parameter, not a hardware measurement'],
        'scope':['One sequence, decode focus; all host weights resident, no HDD misses',
                 'Prefill routes warm the same cache token by token; no batched prefill throughput prediction',
                 'CPU and GPU experts join at each layer; no deferred experts or altered routing',
                 'No future route accesses are used for cache admission',
                 'Power/thermal effects are represented only by effective GPU-rate inputs, not an electrical model']
    },indent=2)+'\n')


def config(ident, card, **kw):
    size = 24 if card == 'P40' else 16
    reserve = kw.pop('reserve',10)
    c = dict(id=ident,card=card,policy=3,slots=int((size-reserve)*2**30//18808832),budget=4,
             queue=32,decay=.97,margin=1.15,min_heat=1.5,cpu_scale=1,host_GBs=45,
             gpu_GBs=220 if card=='P40' else 550, common_ms=35 if card=='P40' else 15,
             pcie_GBs=12,contention=.5,gpu_contention=1,sync_us=40,launch_us=10,
             parallel_common=.2,instant=0,detail=0)
    c.update(kw)
    assert set(c)==set(ORDER)
    return c


def screen():
    configs = [config('control_ideal_lru','V100',policy=2,slots=4500,instant=1,detail=0)]
    for card in ['P40','V100']:
        for name,policy,budget in [('none',0,0),('prefill',1,32),('async_lru',2,0),('sync_lru',4,0)]:
            configs.append(config(f'{card}_{name}',card,policy=policy,budget=budget,detail=1))
        for budget in [1,2,4,8,16,32]:
            configs.append(config(f'{card}_adaptive{budget}',card,budget=budget,detail=1))
    return configs


def sweep():
    configs=[]
    for reserve in [8,10,12]:
        for scale in [1,1.5,2]:
            for pcie in [6,12]:
                for gap in [0,10,20,30,40]:
                    for card in ['P40','V100']:
                        ident=f'grid_r{reserve}_b{scale}_p{pcie}_g{gap}_{card}'
                        configs.append(config(ident,card,reserve=reserve,cpu_scale=scale,
                            host_GBs=45*scale,pcie_GBs=pcie,common_ms=15+(gap if card=='P40' else 0),budget=8))
    for card in ['P40','V100']:
        for alpha in [0,1]:
            configs.append(config(f'contention_{alpha}_{card}',card,contention=alpha,budget=8))
        for overlap in [0,.4]:
            configs.append(config(f'overlap_{overlap}_{card}',card,parallel_common=overlap,budget=8))
        for gpu in ([70,150,280] if card=='P40' else [180,400,700]):
            configs.append(config(f'gpu_{gpu}_{card}',card,gpu_GBs=gpu,budget=8))
        for decay in [.90,.99]:
            configs.append(config(f'decay_{decay}_{card}',card,decay=decay,budget=8))
        for budget in [2,4,16]:
            for scale in [1.5,2]:
                configs.append(config(f'budget_{budget}_b{scale}_{card}',card,budget=budget,
                    cpu_scale=scale,host_GBs=45*scale))
    for reserve in [12,14]:
        configs.append(config(f'P40_extra_workspace_r{reserve}','P40',reserve=reserve,budget=8))
    return configs


def run(mode):
    prepare()
    build = ['c++','-O3','-std=c++17',str(SRC/'hybrid-sim.cpp'),'-o',str(SRC/'hybrid-sim')]
    subprocess.run(build,check=True)
    test = subprocess.run([str(SRC/'hybrid-sim'),'--selftest'],capture_output=True,text=True,check=True)
    (OUT/'sim-build.json').write_text(json.dumps({'command':build,'selftest':test.stdout,
        'source_sha256':hashlib.sha256((SRC/'hybrid-sim.cpp').read_bytes()).hexdigest()},indent=2)+'\n')
    configs = screen() if mode=='screen' else sweep()
    config_path=OUT/f'{mode}-configs.tsv'
    config_path.write_text(''.join('\t'.join(str(c[k]) for k in ORDER)+'\n' for c in configs))
    (OUT/f'{mode}-configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    cmd=[str(SRC/'hybrid-sim'),str(TRACE),str(OUT/'phases.tsv'),str(OUT/'cpu-times.tsv'),
         str(config_path),str(OUT/f'{mode}-results.tsv'),str(OUT/'details')]
    with (OUT/f'{mode}-log.txt').open('w') as log:
        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=child.wait()
    (OUT/f'{mode}-status.json').write_text(json.dumps({'command':cmd,'exit_code':code,
        'scenarios':len(configs),'source_sha256':hashlib.sha256((SRC/'hybrid-sim.cpp').read_bytes()).hexdigest()},indent=2)+'\n')
    assert code==0
    rows=list(csv.DictReader((OUT/f'{mode}-results.tsv').open(),delimiter='\t'))
    assert len([r for r in rows if r['turn']=='-1'])==len(configs)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['screen','sweep']);args=ap.parse_args();run(args.mode)
