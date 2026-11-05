#!/usr/bin/env python3
import json,os
from pathlib import Path
OUT=Path(__file__).resolve().parent.parent/'results/server-study-20261023'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=json.loads((OUT/'summary.json').read_text())['cases']
def vals(names,key):
    m=np.array([s[n][key]['median'] for n in names])
    return m,np.array([m-[s[n][key]['min'] for n in names],[s[n][key]['max'] for n in names]-m])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13.5,4.6))
for ax,prefix,key,title,ylabel in [(axes[0],'read','logical_GBs','Sequential reads: 6 cores/socket','Payload GB/s (higher is better)'),(axes[1],'latency','dependent_load_ns','Dependent random reads: 1 core','ns per dependent load (lower is better)')]:
    for kind,offset,color in [('local',-.18,'#3678ad'),('remote',.18,'#cb873b')]:
        names=[f'{prefix}_n{n}_{kind}' for n in [0,1]]
        v,e=vals(names,key);ax.bar(np.arange(2)+offset,v,width=.34,color=color,yerr=e,capsize=3,label=kind)
        for i,value in enumerate(v):ax.text(i+offset,value+max(v)*.03,f'{value:.1f}',ha='center',fontsize=9)
    ax.set_xticks([0,1],['CPU socket 1','CPU socket 2']);ax.set_title(title,fontweight='bold');ax.set_ylabel(ylabel);ax.set_ylim(0,30 if prefix=='read' else 240);ax.grid(axis='y',alpha=.15)
    ax.legend(frameon=False,loc='upper center',ncols=2,fontsize=8)
names=['copy_n0_local1','copy_n0_from_remote1','copy_n0_to_remote1'];v,e=vals(names,'logical_GBs')
ax=axes[2];ax.bar(np.arange(3),v,color=['#3678ad','#74a9cd','#cb873b'],yerr=e,capsize=3)
for i,value in enumerate(v):ax.text(i,value+.12,f'{value:.2f}',ha='center')
ax.set_xticks(range(3),['Local → local','Remote → local','Local → remote'],rotation=18,ha='right')
ax.set_ylim(0,4.3);ax.set_title('Memory copy: 1 core on socket 1',fontweight='bold');ax.set_ylabel('Copied payload GB/s');ax.grid(axis='y',alpha=.15)
fig.suptitle('Direct measurements on the current dual Xeon / 128GB server',fontweight='bold')
fig.subplots_adjust(top=.83,bottom=.23,wspace=.35)
fig.text(.08,.015,'Three runs per condition; bars show medians, whiskers show min–max. CPU and memory placement verified.\n4 KiB pages. Random-read latency includes cache/TLB effects. No GPU work or GPU simulation.',fontsize=9)
fig.savefig(OUT/'memory-paths.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/'memory-paths.pdf',bbox_inches='tight')
