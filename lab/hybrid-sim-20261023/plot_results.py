#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC=Path(__file__).resolve().parent
OUT=SRC.parent/'results/hybrid-sim-20261023'


def table(mode):
    return {r['id']:r for r in csv.DictReader((OUT/f'{mode}-results.tsv').open(),delimiter='\t') if r['turn']=='-1'}


def main():
    screen,sweep=table('screen'),table('sweep')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.20,'figure.facecolor':'white','savefig.facecolor':'white'})
    colors={'P40':'#127b9e','V100':'#b96321'}
    fig,ax=plt.subplots(figsize=(10.5,6.3))
    gaps=[0,10,20,30,40]
    for scale,style,label in [(1,'-','measured CPU baseline'),(1.5,'--','1.5x memory-side bandwidth, compute capped')]:
        for card in ['P40','V100']:
            y=[float(sweep[f'grid_r10_b{scale}_p12_g{g}_{card}']['tok_s']) for g in gaps]
            ax.plot(gaps,y,style,marker='o',color=colors[card],linewidth=2,label=f'{card}: {label}')
    ax.axvspan(12.7,15.1,color='#666666',alpha=.10)
    ax.text(14,19.0,'Crossovers\n~13-15 ms',ha='center',va='top',color='#444444')
    ax.set(xlabel='Assumed extra non-expert GPU time on P40 vs V100 (ms / token)',
           ylabel='Simulated decode throughput (tokens / s)',ylim=(8,20),xlim=(-.5,40.5))
    ax.legend(loc='lower left',fontsize=9,framealpha=.95)
    fig.suptitle('P40 24 GiB vs V100 16 GiB: conditional simulation',fontsize=16,fontweight='bold')
    ax.set_title('Not a GPU benchmark; GPU timings at 200 W have not been measured',fontsize=11,color='#8b3a25',pad=14)
    fig.text(.11,.045,'Inputs: 10 GiB reserved per GPU; 6 / 14 GiB expert cache; PCIe 12 GB/s; up to 8 updates/token.\nV100 non-expert time = 15 ms/token. Expert throughput assumptions: P40 220, V100 550 GB/s.\nReal September V4.1 routing trace: 8 turns, 2,256 decoded tokens. CPU times measured on dual E5-2630L v2.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.93))
    for ext in ['png','svg']:fig.savefig(OUT/f'crossover.{ext}',dpi=170)
    plt.close(fig)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5))
    labels=['Wait for\nevery refill','Async LRU\nqueue <= 32','Recent frequency\n<= 8 updates/token']
    suffix=['sync_lru','async_lru','adaptive8'];x=np.arange(3);width=.34
    for card,offset in [('P40',-width/2),('V100',width/2)]:
        rates=[float(screen[f'{card}_{s}']['tok_s']) for s in suffix]
        traffic=[float(screen[f'{card}_{s}']['DMA_MB_per_token'])/1000 for s in suffix]
        bars=ax1.bar(x+offset,rates,width,color=colors[card],label=card)
        ax1.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        bars=ax2.bar(x+offset,traffic,width,color=colors[card],label=card)
        ax2.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax1.set(ylabel='Simulated tokens / s',ylim=(0,17));ax2.set(ylabel='Completed weight copies: GB / token',ylim=(0,3.3))
    for ax in [ax1,ax2]:ax.set_xticks(x,labels);ax.legend(frameon=False);ax.set_axisbelow(True)
    fig.suptitle('Selective asynchronous updates reduce transfer cost',fontsize=15,fontweight='bold')
    fig.text(.08,.035,'Illustrative inputs only: P40 non-expert time 35 ms/token, V100 15 ms/token; measured CPU baseline.\nThe full-model P40/V100 speeds are not measured. All cache misses are computed by the CPU.',fontsize=10,color='#8b3a25')
    fig.tight_layout(rect=(0,.13,1,.93))
    for ext in ['png','svg']:fig.savefig(OUT/f'policy-and-traffic.{ext}',dpi=170)
    plt.close(fig)

    versions={'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    (OUT/'plot-versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    print('Saved crossover and policy-and-traffic PNG/SVG figures')


if __name__=='__main__':main()
