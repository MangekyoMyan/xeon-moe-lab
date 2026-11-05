#!/usr/bin/env python3
"""Standalone scientific figures. Requires matplotlib and numpy."""
import csv
import json
import os
from pathlib import Path

OUT=Path(__file__).resolve().parent.parent/'results/hybrid-sim-deep-20261023'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150})
COLORS={'P40':'#cb6b26','V100':'#236ab4'}
data={r['id']:{k:v if k in ['mode','id','card','trace'] else float(v) for k,v in r.items()} for r in csv.DictReader((OUT/'all-overall-results.tsv').open(),delimiter='\t')}
summary=json.loads((OUT/'summary.json').read_text())

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

# Mutually exclusive intervals, not an additive CPU/GPU utilization plot.
fig,ax=plt.subplots(figsize=(10.5,4.3))
parts=[('common_path_ms','GPU projections / router / head','#3d74a7'),('attention_path_ms','Attention / indexer','#92bbd6'),
       ('parallel_ms','CPU and GPU branches overlap','#74a383'),('cpu_tail_ms','CPU-side tail, incl. result transfer','#df9a49'),
       ('gpu_tail_ms','GPU-side tail','#b05b6a')]
for y,card in enumerate(['P40','V100']):
    r=data['central_'+card];left=0
    for key,label,color in parts:
        value=r[key];ax.barh(y,value,left=left,height=.5,color=color,label=label if y==0 else None)
        if value>6:ax.text(left+value/2,y,f'{value:.1f}',ha='center',va='center',color='white',weight='bold')
        left+=value
    rest=r['ms_token']-left;ax.barh(y,rest,left=left,height=.5,color='#bebebe',label='Other transfers / merge / control' if y==0 else None)
    ax.text(r['ms_token']+1,y,f"{r['ms_token']:.1f} ms  |  {r['tok_s']:.2f} tok/s",va='center')
ax.set_yticks([0,1],['P40: 709 expert slots','V100: 345 expert slots']);ax.invert_yaxis();ax.set_xlim(0,132)
ax.set_xlabel('Simulated decode time per token (ms; smaller is faster)')
ax.set_title('Why 24GB and 16GB remain close on the old Xeon')
ax.legend(ncols=2,loc='upper left',bbox_to_anchor=(0,-.22),frameon=False,fontsize=9)
fig.text(.125,.99,'Base public trace; CPU calibrated locally. GPU rates are assumptions, not 200W measurements.',fontsize=9,va='bottom')
save(fig,'critical-path')

fig,ax=plt.subplots(figsize=(9,4.8))
ws=[6,12,14,18,32,384];pool=[w*40 for w in ws]
syn={(r['id'],r['selection']):r for r in summary['synthetic']}
for card in ['P40','V100']:
    values=[syn[(f'uniform_w{w:03}_{card}','after_32_per_turn')]['tok_s'] for w in ws]
    ax.plot(range(len(ws)),values,'o-',color=COLORS[card],lw=2,label=card)
    other='V100' if card=='P40' else 'P40'
    for x,v in enumerate(values):
        peer=syn[(f'uniform_w{ws[x]:03}_{other}','after_32_per_turn')]['tok_s']
        ax.annotate(f'{v:.1f}',(x,v),xytext=(0,9 if v>=peer else -16),textcoords='offset points',ha='center',color=COLORS[card])
ax.set_xticks(range(len(ws)),[str(p) for p in pool]);ax.set_ylim(0,31);ax.set_ylabel('Simulated decode tokens/s');ax.set_xlabel('Total experts in the persistent working set (40 layers; categorical spacing)')
ax.set_title('Synthetic locality stress: extra VRAM can reverse the result')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
fig.subplots_adjust(bottom=.23)
fig.text(.125,.01,'Same GPU/CPU assumptions throughout. Uniform requests within fixed layer-local sets.\n256 warmup + 2048 decode; first 32 decode excluded only from this view. Not a real-route forecast.',fontsize=9)
save(fig,'locality-stress')

fig,axes=plt.subplots(1,2,figsize=(11.4,5),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
cases=[('none','No expert cache'),('lru_chunk0','LRU / whole expert copies'),('lru_chunk262144','LRU / 256 KiB chunks'),
       ('ema8_chunk0','EMA8 / whole expert copies'),('ema8_chunk1048576','EMA8 / 1 MiB chunks'),('aggressive_budget32_chunk256K','EMA32 / 256 KiB chunks')]
y=np.arange(len(cases))
for card,offset in [('P40',-.17),('V100',.17)]:
    rr=[data[name+'_'+card] for name,label in cases]
    axes[0].barh(y+offset,[r['tok_s'] for r in rr],height=.31,color=COLORS[card],label=card)
    axes[1].barh(y+offset,[r['copy_MB_token']/1000 for r in rr],height=.31,color=COLORS[card])
axes[0].set_yticks(y,[label for name,label in cases]);axes[0].invert_yaxis();axes[0].set_xlim(0,12)
axes[0].set_xlabel('Simulated decode tokens/s');axes[1].set_xlabel('Weight transfers (GB/token)')
axes[0].legend(frameon=False,ncols=2,loc='lower left',bbox_to_anchor=(0,1.01))
for a in axes:a.grid(axis='x',alpha=.2)
fig.suptitle('Cache policy matters more than simply copying every miss',fontweight='bold')
fig.subplots_adjust(bottom=.22,top=.84)
fig.text(.125,.01,'Common weights stay on GPU in every case. EMA8/32: at most 8/32 new experts per token.\nChunks prioritize small activation/result transfers. Rates are conditional simulation outputs.',fontsize=9)
save(fig,'transfer-policy')

fig,ax=plt.subplots(figsize=(7.4,5.2))
ps=[150,210,275];vs=[350,475,650]
ratios=np.array([[data[f'band_P{p}_V{v}_P40']['tok_s']/data[f'band_P{p}_V{v}_V100']['tok_s'] for v in vs] for p in ps])
im=ax.imshow(ratios,cmap='RdBu_r',vmin=.75,vmax=1.25)
for i,p in enumerate(ps):
    for j,v in enumerate(vs):
        winner='P40' if ratios[i,j]>1 else 'V100';adv=(ratios[i,j]-1)*100 if ratios[i,j]>1 else (1/ratios[i,j]-1)*100
        ax.text(j,i,f'{winner} +{adv:.1f}%\n{data[f"band_P{p}_V{v}_P40"]["tok_s"]:.2f} / {data[f"band_P{p}_V{v}_V100"]["tok_s"]:.2f}',ha='center',va='center',fontsize=10)
ax.set_xticks(range(3),vs);ax.set_yticks(range(3),ps);ax.set_xlabel('V100 assumed effective weight-read rate (GB/s)');ax.set_ylabel('P40 assumed effective weight-read rate (GB/s)');ax.set_title('GPU kernel efficiency can change the winner')
fig.colorbar(im,ax=ax,label='P40 throughput / V100 throughput')
fig.subplots_adjust(bottom=.22)
fig.text(.125,.01,'Cell values: P40 / V100 tokens/s. Base real route, native weight storage.\nA parameter grid, not probabilities or confidence intervals.',fontsize=9)
save(fig,'bandwidth-grid')

fig,ax=plt.subplots(figsize=(8.5,4.5))
for card,offset in [('P40',-.17),('V100',.17)]:
    vals=[r['speedup_percent'] for r in summary['RAM_sensitivity'] if r['card']==card]
    ax.bar(np.arange(4)+offset,vals,width=.32,color=COLORS[card],label=card)
    for i,v in enumerate(vals):ax.text(i+offset,v+.2,f'{v:.1f}%',ha='center',fontsize=9)
ax.set_xticks(range(4),[40,45,50,55]);ax.set_ylim(0,14);ax.set_xlabel('CPU read-rate parameter used when refitting (GB/s)');ax.set_ylabel('Simulated gain from 2x RAM bandwidth (%)');ax.legend(frameon=False)
ax.set_title('CPU measurements do not uniquely identify all bottlenecks')
fig.subplots_adjust(bottom=.23)
fig.text(.125,.01,'All four fits reproduce the same six measured CPU graph times.\nCompute and fixed costs held unchanged; RAM bandwidth alone doubled. No measured RAM-upgrade claim.',fontsize=9)
save(fig,'cpu-identifiability')
print('Saved 5 PNG and PDF figures to',OUT)
