#!/usr/bin/env python3
"""Plot independently reaggregated saved results; no new inference measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output must be new")
    data = json.loads(args.results.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#162235", "xtick.color": "#475569", "ytick.color": "#334155"})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    titles = [("final", "Final: adopted configuration", "Fresh ABBA; 2 runs per arm", "#087F8C"),
              ("hot14", "Hot14: experimental configuration", "2 control runs, then 1 candidate run", "#B66A19")]
    for ax, (key, title, subtitle, color) in zip(axes, titles):
        rows = data[key]["cases"]
        y = np.arange(len(rows))
        old = [row["control_decode_tok_s"] for row in rows]
        new = [row["candidate_decode_tok_s"] for row in rows]
        ax.barh(y - .17, old, height=.29, color="#CAD3DE", label="Matched control")
        ax.barh(y + .17, new, height=.29, color=color, label=key)
        for i, row in enumerate(rows):
            ax.text(max(old[i], new[i]) + .12, i + .17,
                    f"{row['decode_rate_gain_percent']:+.1f}%", va="center", fontsize=10, color=color)
        ax.set_title(title, loc="left", fontsize=14, pad=31)
        ax.text(0, 1.018, subtitle, transform=ax.transAxes, fontsize=10, color="#596579")
        ax.set_yticks(y, [row["name"].removeprefix("fresh-") for row in rows])
        ax.set_xlabel("Pooled server decode rate (steps / second)", labelpad=12)
        ax.set_xlim(0, 11.7)
        ax.xaxis.grid(True, color="#E7ECF2")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Xeon MoE Lab | saved performance evidence", x=.095, y=.98, ha="left", fontsize=19, weight="bold")
    fig.text(.095, .918, "Final: output / token hashes match in all 8 cases.  Hot14: hashes match in 1 of 8 cases.", fontsize=11)
    fig.text(.095, .055, "Measurements: 12 Nov 2026. Independent reaggregation: 16 Nov 2026. The two panels use different controls.", fontsize=9, color="#596579")
    fig.text(.095, .03, "Bars measure server decode work, not streaming smoothness or answer quality. Hot14 is not an equal-output ABBA comparison.", fontsize=9, color="#596579")
    fig.subplots_adjust(left=.095, right=.985, top=.80, bottom=.15, wspace=.12)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
