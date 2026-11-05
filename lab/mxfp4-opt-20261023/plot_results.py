#!/usr/bin/env python3
"""Standalone plots of measured throughput and warm-cache loading time."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mxfp4-matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / 'results/mxfp4-opt-20261023'
summary = json.loads((OUT / 'SUMMARY.json').read_text())
colors = ['#456d91', '#258c78']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, phase, title in zip(axes, ['pp128', 'pp512', 'tg256'], ['Prompt: 128 tokens', 'Prompt: 512 tokens', 'Generation: 256 tokens']):
    values = summary['primary']['phases'][phase]
    heights = [values[label]['pooled_tokens_s'] for label in ('A', 'B')]
    ax.bar([0, 1], heights, width=.56, color=colors, zorder=2)
    for x, label in enumerate(('A', 'B')):
        samples = [r['tokens'] * 1e9 / ns for r in summary['primary']['runs'] if r['variant'] == label and r['phase'] == phase for ns in r['samples_ns']]
        offsets = [(.12 * (i / max(1, len(samples)-1) - .5)) for i in range(len(samples))]
        ax.scatter([x + offset for offset in offsets], samples, s=14, color='#172c3b', zorder=3)
        ax.text(x, heights[x] + max(heights) * .035, f'{heights[x]:.2f}', ha='center')
    ax.set_xticks([0, 1], ['Baseline', 'Optimized'])
    ax.set_ylabel('tokens / second')
    ax.set_title(title + f"\n+{values['speed_gain_percent']:.1f}% throughput")
    ax.set_ylim(0, max(heights) * 1.19)
    ax.grid(axis='y', alpha=.2, zorder=0)
fig.suptitle('gpt-oss-20b MXFP4 | dual Xeon E5-2630L v2 | 24 CPU threads', fontsize=12)
fig.text(.5, .015, 'Same GGUF; 4 runs per variant, 2 repetitions. Points show individual samples. No GPU.', ha='center', fontsize=9)
fig.tight_layout(rect=[0, .06, 1, .92])
fig.savefig(OUT / 'model-performance.png', dpi=180)
fig.savefig(OUT / 'model-performance.pdf')
plt.close(fig)

labels = ['Baseline', 'Parallel packing\nwithout buffer reuse', 'Final optimized']
values = [summary['startup'][mode]['load_ms'] / 1000 for mode in ('baseline', 'parallel', 'reuse')]
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(labels, values, color=[colors[0], '#9ba5aa', colors[1]], width=.55)
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, value+.4, f'{value:.2f} s', ha='center')
ax.set_ylim(0, max(values) * 1.18)
ax.set_ylabel('reported model load time (seconds)')
ax.set_title('Loading the same model with the file already cached by the OS')
ax.grid(axis='y', alpha=.2)
fig.text(.5, .015, 'Mean of two runs in reversed order. This is not a cold HDD loading test.', ha='center', fontsize=9)
fig.tight_layout(rect=[0, .06, 1, 1])
fig.savefig(OUT / 'startup-performance.png', dpi=180)
fig.savefig(OUT / 'startup-performance.pdf')
