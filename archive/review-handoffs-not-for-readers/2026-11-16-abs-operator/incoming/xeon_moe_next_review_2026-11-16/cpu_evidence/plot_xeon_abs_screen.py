#!/usr/bin/env python3
"""Render saved Xeon input-abs screen data; does not run a benchmark.

Points are medians of four within-block paired speedup ratios; whiskers are
their observed minimum and maximum, not confidence intervals. Only wall-time
results for experimental_input_abs are shown. The separate bias62 blocks are
intentionally not compared here.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "XEON_SCREEN_REANALYSIS.json")
    parser.add_argument("--output", type=Path, default=here / "XEON_ABS_SCREEN.png")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    types = ("IQ3_XXS", "IQ2_S", "IQ3_S")
    shapes = ((2304, 256), (2304, 8192), (5120, 256), (5120, 8192))
    records = [r for r in data["recomputed_rows"] if r["variant"] == "experimental_input_abs"]
    expected = {(t, k, m, socket) for t in types for k, m in shapes for socket in (0, 1)}
    keys = [(r["type"], r["K"], r["rows"], r["socket"]) for r in records]
    if len(keys) != 24 or set(keys) != expected:
        raise ValueError("Expected exactly 12 input-abs conditions x 2 sockets")
    indexed = dict(zip(keys, records))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.8), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.73, bottom=0.265, wspace=0.13)
    colors = ("#1763A6", "#D36514")
    markers = ("o", "^")
    offsets = (-0.12, 0.12)
    all_gains = []
    for axis, typ in zip(axes, types):
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E3E7EC", linewidth=0.8)
        axis.axhline(3, color="#707780", linestyle=(0, (5, 4)), linewidth=1.4, zorder=1)
        for socket in (0, 1):
            x = []; medians = []; lower = []; upper = []
            for i, (k, m) in enumerate(shapes):
                row = indexed[(typ, k, m, socket)]
                ratios = row["wall"]["paired_ratios_from_arm_times"]
                if len(ratios) != 4 or not all(math.isfinite(v) and v > 0 for v in ratios):
                    raise ValueError("Expected four finite positive paired ratios")
                median = statistics.median(ratios)
                if not math.isclose(median, row["wall"]["paired_median_ratio"], abs_tol=1e-12):
                    raise ValueError("Saved median does not match its paired ratios")
                gains = [100 * (r - 1) for r in ratios]
                point = 100 * (median - 1)
                x.append(i + offsets[socket]); medians.append(point)
                lower.append(point - min(gains)); upper.append(max(gains) - point)
                all_gains.extend(gains)
            axis.errorbar(x, medians, yerr=[lower, upper], fmt=markers[socket],
                          color=colors[socket], ecolor=colors[socket], markersize=7,
                          markeredgecolor="white", markeredgewidth=0.65,
                          elinewidth=1.6, capsize=4, capthick=1.3, zorder=3)
        axis.set_title(typ, fontweight="bold", pad=13)
        axis.set_xticks(range(4), [f"K {k:,}\nM {m:,}" for k, m in shapes])
        axis.tick_params(axis="x", length=0, pad=10)
        axis.set_xlim(-0.45, 3.45)
        axis.yaxis.set_major_locator(MultipleLocator(1))
        axis.spines["left"].set_color("#AEB6C0")
        axis.spines["bottom"].set_color("#AEB6C0")
    axes[0].set_ylabel("Speed increase vs current compact (%)", labelpad=12)
    axes[0].set_ylim(0, max(9, math.ceil(max(all_gains) + 0.35)))
    fig.suptitle("Input-abs CPU screen: supplied Xeon measurements", fontsize=20,
                 fontweight="bold", x=0.065, y=0.965, ha="left", color="#162537")
    fig.text(0.065, 0.901,
             "Single worker  |  synthetic weights and one input  |  preparation included  |  not end-to-end",
             fontsize=12, color="#48586B", ha="left")
    handles = [Line2D([0], [0], color=colors[s], marker=markers[s], linestyle="none",
                      markersize=8, label=f"CPU {3 if s == 0 else 9} / NUMA node {s}") for s in (0, 1)]
    handles.append(Line2D([0], [0], color="#707780", linestyle=(0, (5, 4)), linewidth=1.4,
                          label="Predeclared +3% median screen"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.863),
               ncol=3, frameon=False, handlelength=2.5, columnspacing=2.1)
    fig.text(0.065, 0.122,
             "Points: median of 4 paired speedups. Whiskers: observed minimum to maximum of those 4 pairs; not confidence intervals.",
             fontsize=10.5, color="#354457", ha="left")
    fig.text(0.065, 0.078,
             "Only wall-time input-abs results are shown. The complete screen also checked every pair > 1 and corresponding thread CPU gains.",
             fontsize=10, color="#556274", ha="left")
    fig.text(0.065, 0.038,
             "Source: XEON_SCREEN_REANALYSIS.json  |  supplied runs dated 2026-11-16  |  Xeon E5-2630L v2  |  no new performance measurement",
             fontsize=9.5, color="#657185", ha="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "conditions": 12, "sockets": 2,
                      "paired_ratios_shown": 96, "whiskers": "observed min-max, not CI"}))


if __name__ == "__main__":
    main()
