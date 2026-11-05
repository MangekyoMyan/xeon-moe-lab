#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

SRC=Path(__file__).resolve().parent
LAB=SRC.parent
OUT=LAB/'results/hybrid-sim-deep-20261023'
OLD=LAB/'results/hybrid-sim-20261023'
EVID=LAB/'results/expert-cache-evidence-20261022'
ORDER=['id','card','trace','phases','shapes','cpu_file','slots','policy','budget','queue','balance','context_base','candidate_first','detail',
       'decay','margin','min_heat','usable_gib','workspace_gib','mem_scale','cpu_fixed_scale','cpu_compute_scale','common_bw','expert_bw',
       'gpu_peak_bw','gpu_GF','attention_GF','host_bw','h2d_bw','d2h_bw','chunk','copy_setup_us','transfer_us','common_us','expert_us','merge_us','occupancy']


def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'details').mkdir(exist_ok=True)
    groups=list(csv.DictReader((OUT/'gpu-evidence/stage-group-accounting.csv').open()))
    matrices={}
    for mode in ['native','wo_a_fp16','all_fp8_to_fp16']:
        rows=[[0.0]*6 for _ in range(41)]
        for r in groups:
            if r['mode']!=mode:continue
            index=int(r['layer']);group=r['group'];b=int(r['decode_weight_read_bytes']);f=int(r['matrix_dot_flops'])
            offset={'pre_attention':0,'post_attention':2,'shared_parallel':4,'tail':0,'attention_math':0}[group]
            rows[index][offset]+=b;rows[index][offset+1]+=f
        path=OUT/f'layers-{mode}.tsv';path.write_text(''.join('\t'.join(map(str,r))+'\n' for r in rows));matrices[mode]=path
    gops={}
    for k in [5120,2304]:
        r=[json.loads(l) for l in (OUT/f'compute22-k{k}/stdout.txt').read_text().splitlines()]
        gops[k]=statistics.median(2*x['Gweights_s'] for x in r if 'Gweights_s' in x)
    floor_one=(2*2*5120*2304/gops[5120]+2*5120*2304/gops[2304])/1e6
    raw=[json.loads(l) for l in (OUT/'full-cpu-control-core/stdout.txt').read_text().splitlines()]
    full=[0.0]+[statistics.median(r['ms_per_layer'] for r in raw if r.get('threads')==22 and r.get('selected')==n and 'ms_per_layer' in r) for n in range(1,7)]
    compute=[n*floor_one for n in range(7)]
    fixed=[full[n]-max(n*18800640/(45*1e6),compute[n]) for n in range(7)]
    assert min(fixed)>=0
    cpufile=OUT/'cpu-full22.tsv';cpufile.write_text(' '.join(map(str,fixed))+'\n'+' '.join(map(str,compute))+'\n')
    calibration={'full22_ms':full,'fixed_ms':fixed,'compute_proxy_ms':compute,'cache_GOPS':gops,'uncontended_CPU_read_GBs':45,
                 'reconstruction_error':[fixed[n]+max(n*18800640/(45*1e6),compute[n])-full[n] for n in range(7)]}
    (OUT/'cpu-model.json').write_text(json.dumps(calibration,indent=2)+'\n')
    traces={'base':(EVID/'cachedmoe/bench/results/hitrate/base_auto/route.bin',OLD/'phases.tsv'),
            'cap5500':(OUT/'trace-policy/sources/bench/results/hitrate/cap5500/route.bin',OUT/'trace-policy/cap5500-phases.tsv'),
            'conv8':(OUT/'trace-policy/sources/bench/results/linux/conv8_auto/route.bin',OUT/'trace-policy/conv8_auto-phases.tsv')}
    accounting=json.loads((OUT/'gpu-evidence/header-accounting.json').read_text())
    return matrices,cpufile,traces,accounting


def kv_bytes(N):
    return int(40*128*512*(1+1/32)+(3*(N//2)+N)*(512*(.5+1/16)+128*(.5+1/32)))


def configurations(mode):
    shapes,cpu,traces,accounting=prepare();rows=[];rejected=[]
    def add(name,card,trace='base',dtype='native',**kw):
        c=dict(id=name+'_'+card,card=card,trace=str(traces[trace][0]),phases=str(traces[trace][1]),shapes=str(shapes[dtype]),cpu_file=str(cpu),
               slots=0,policy=3,budget=8,queue=32,balance=1,context_base=0,candidate_first=1,detail=0,
               decay=.97,margin=1.15,min_heat=1.5,usable_gib=22.375 if card=='P40' else 16,workspace_gib=2,
               mem_scale=1,cpu_fixed_scale=1,cpu_compute_scale=1,
               common_bw=210 if card=='P40' else 475,expert_bw=210 if card=='P40' else 475,
               gpu_peak_bw=346 if card=='P40' else 900,gpu_GF=5000 if card=='P40' else 10000,
               attention_GF=1000,host_bw=48,h2d_bw=12,d2h_bw=12,chunk=1048576,copy_setup_us=3,
               transfer_us=20,common_us=100,expert_us=20,merge_us=5,occupancy=1)
        c.update(kw)
        spare=c['usable_gib']*2**30-accounting['decode_common_variants'][dtype]['weights_bytes']-c['workspace_gib']*2**30-kv_bytes(c['context_base']+4096)
        c['slots']=int(spare//18808832)
        if c['slots']<0:
            rejected.append({'id':c['id'],'reason':'common weights + workspace + KV exceed usable VRAM','excess_GiB':-spare/2**30});return
        assert set(c)==set(ORDER)
        rows.append(c)
    if mode=='screen':
        for card in ['P40','V100']:
            add('none',card,policy=0,detail=1)
            add('prefill',card,policy=1,budget=32)
            for chunk in [0,1048576,262144]:
                add(f'lru_chunk{chunk}',card,policy=2,chunk=chunk,detail=1)
                add(f'ema8_chunk{chunk}',card,chunk=chunk,detail=1)
            for budget in [2,4,16,32]:add(f'ema{budget}',card,budget=budget)
            add('ema8_all_gpu_hits',card,balance=0)
        for trace in ['cap5500','conv8']:
            for card in ['P40','V100']:add(f'trace_{trace}',card,trace=trace,detail=1)
    elif mode=='sweep':
        for p in [150,210,275]:
            for v in [350,475,650]:
                for card,bw in [('P40',p),('V100',v)]:add(f'band_P{p}_V{v}',card,common_bw=bw,expert_bw=bw)
        for trace in ['cap5500','conv8']:
            for p,v in [(150,350),(210,475),(275,650),(275,350)]:
                for card,bw in [('P40',p),('V100',v)]:add(f'band_{trace}_P{p}_V{v}',card,trace=trace,common_bw=bw,expert_bw=bw)
        for card in ['P40','V100']:
            for workspace in [1,3,4]:add(f'workspace{workspace}',card,workspace_gib=workspace)
            for mem in [1.5,2]:add(f'memory{mem}',card,mem_scale=mem)
            for fixed in [0,.5]:add(f'CPU_fixed{fixed}',card,cpu_fixed_scale=fixed)
            for pcie in [6,8]:
                for chunk in [0,1048576,262144]:add(f'pcie{pcie}_chunk{chunk}',card,h2d_bw=pcie,d2h_bw=pcie,chunk=chunk)
            for ctx in [32768,131072,1048576]:
                for opt in [0,1]:add(f'context{ctx}_opt{opt}',card,context_base=ctx,candidate_first=opt)
            for attention in [500,2000]:add(f'attention{attention}',card,attention_GF=attention)
            for common in [50,250]:add(f'launch{common}',card,common_us=common)
            for factor in [.6,.8,1.2]:
                anchor=210 if card=='P40' else 475
                add(f'common_eff{factor}',card,common_bw=anchor*factor)
            for decay in [.9,.99,.995]:add(f'decay{decay}',card,decay=decay)
            for dtype in ['wo_a_fp16','all_fp8_to_fp16']:add('dtype_'+dtype,card,dtype=dtype)
            add('dtype_allfp16_lowworkspace',card,dtype='all_fp8_to_fp16',workspace_gib=1)
            for setup in [0,10]:add(f'copy_setup{setup}',card,copy_setup_us=setup,chunk=262144)
            add('uniform_expert_bandwidth',card,occupancy=0)
            add('longer_activation_latency',card,transfer_us=60)
        add('P40_nominal24','P40',usable_gib=24)
        add('V100_usable15_75','V100',usable_gib=15.75)
    elif mode=='diagnostics':
        for card in ['P40','V100']:
            add('central',card,detail=1)
            for bw in [100,150,210,275,350,475,650]:
                add(f'expert_bw{bw}',card,expert_bw=bw)
            for compute in [.8,.5]:
                add(f'CPU_compute{compute}',card,cpu_compute_scale=compute)
                add(f'memory2_CPU_compute{compute}',card,mem_scale=2,cpu_compute_scale=compute)
            add('memory2_CPU_compute0.5_fixed0.5',card,mem_scale=2,cpu_compute_scale=.5,cpu_fixed_scale=.5)
            add('pcie3_slow_link',card,h2d_bw=3,d2h_bw=3)
            add('queue8',card,queue=8)
            add('queue64',card,queue=64)
            add('aggressive_budget32_chunk256K',card,budget=32,chunk=262144)
            add('prefill_chunk256K',card,policy=1,budget=32,chunk=262144)
        for bw in [230,240,245,250,255,260]:add(f'P40_rate{bw}','P40',common_bw=bw,expert_bw=bw)
        for bw in [325,340,345,350,355,360,375]:add(f'V100_rate{bw}','V100',common_bw=bw,expert_bw=bw)
        add('P40_with_V100_cache','P40',workspace_gib=8.375)
    elif mode=='cpu_sensitivity':
        calibration=json.loads((OUT/'cpu-model.json').read_text())
        for ref in [40,45,50,55]:
            fixed=[calibration['full22_ms'][n]-max(n*18800640/(ref*1e6),calibration['compute_proxy_ms'][n]) for n in range(7)]
            assert min(fixed)>=0
            cpufile=OUT/f'cpu-ref{ref}.tsv'
            cpufile.write_text(' '.join(map(str,fixed))+'\n'+' '.join(map(str,calibration['compute_proxy_ms']))+'\n')
            for card in ['P40','V100']:
                for memory in [1,2]:
                    for compute in [1,.5]:
                        add(f'cpu_ref{ref}_RAM{memory}_compute{compute}',card,cpu_file=str(cpufile),mem_scale=memory*ref/45,host_bw=48*45/ref,cpu_compute_scale=compute)
    else:
        add('smoke','P40',detail=1);add('smoke','V100',detail=1)
    assert len({c['id'] for c in rows})==len(rows)
    (OUT/f'{mode}-infeasible.json').write_text(json.dumps(rejected,indent=2)+'\n')
    return rows


def main(mode):
    rows=configurations(mode)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'deep-sim.cpp',SRC/'resources.hpp',SRC/'cache.hpp',SRC/'run_deep.py']}
    input_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted({c[k] for c in rows for k in ['trace','phases','shapes','cpu_file']})}
    (OUT/f'{mode}-configs.json').write_text(json.dumps(rows,indent=2)+'\n')
    configfile=OUT/f'{mode}-configs.tsv';configfile.write_text(''.join('\t'.join(str(c[k]) for k in ORDER)+'\n' for c in rows))
    temp_binary=SRC/f'.deep-sim-{os.getpid()}'
    build=['c++','-O3','-std=c++17','-Wall','-Wextra','-Wpedantic',str(SRC/'deep-sim.cpp'),'-o',str(temp_binary)]
    subprocess.run(build,check=True)
    temp_binary.replace(SRC/'deep-sim')
    cmd=[str(SRC/'deep-sim'),str(configfile),str(OUT/f'{mode}-results.tsv'),str(OUT/'details')]
    with (OUT/f'{mode}-log.txt').open('w') as log:
        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        code=child.wait()
    (OUT/f'{mode}-status.json').write_text(json.dumps({'command':cmd,'exit_code':code,'cases':len(rows),
        'source_sha256':source_hashes,'input_sha256':input_hashes},indent=2)+'\n')
    assert code==0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','screen','sweep','diagnostics','cpu_sensitivity']);args=ap.parse_args();main(args.mode)
