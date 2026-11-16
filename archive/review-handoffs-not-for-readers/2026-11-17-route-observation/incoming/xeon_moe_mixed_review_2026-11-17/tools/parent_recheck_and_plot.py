#!/usr/bin/env python3
"""Recompute saved mixed-run totals; optionally plot every C0 chunk. No model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

COMMIT = "ff6168094858141aeeebf45abc5c309882fe5216"
HANDOFF = Path("archive/review-handoffs-not-for-readers/2026-11-17-mixed-operator")
LABELS = "FEEFEFFE"
PAIRS = [(0, 1), (3, 2), (5, 4), (6, 7)]
CASES = ("C0", "C6", "Cplan")


def recheck(repo: Path) -> tuple[dict, dict]:
    source = repo / HANDOFF
    raw = {}
    hashes = {}
    totals = {}
    for process in range(8):
        file = source / "mixed-run" / f"process-{process}" / "raw.jsonl"
        data = file.read_bytes()
        hashes[str(file.relative_to(repo))] = hashlib.sha256(data).hexdigest()
        records = [json.loads(line) for line in data.splitlines()]
        meta = [row for row in records if row["kind"] == "mixed_meta"]
        assert len(meta) == 1 and meta[0]["process_index"] == process
        assert meta[0]["library_label"] == LABELS[process]
        rows = [row for row in records if row["kind"] == "mixed_chunk"]
        assert len(rows) == 192
        assert {(r["case"], r["block"], r["arm"]) for r in rows} == {
            (c, b, a) for c in CASES for b in range(16) for a in range(4)
        }
        for row in rows:
            assert row["end_ns"] - row["begin_ns"] == row["elapsed_ns"] > 0
            assert row["repeats"] == 32 and row["bit_mismatches"] == 0
            assert row["actual_mode"] == (1 if LABELS[process] == "E" and row["logical_mode"] == "S" else 0)
            assert math.isclose(row["ms_per_graph"], row["elapsed_ns"] / 32e6, rel_tol=1e-14)
        raw[process] = rows
        totals[process] = {
            c: {m: sum(r["elapsed_ns"] for r in rows if r["case"] == c and r["logical_mode"] == m)
                for m in ("O", "S")}
            for c in CASES
        }
    for p in range(7):
        assert max(r["end_ns"] for r in raw[p]) < min(r["begin_ns"] for r in raw[p + 1])

    summaries = []
    for case in CASES:
        eo = sum(totals[p][case]["O"] for p in range(8) if LABELS[p] == "E")
        es = sum(totals[p][case]["S"] for p in range(8) if LABELS[p] == "E")
        f = sum(sum(totals[p][case].values()) for p in range(8) if LABELS[p] == "F")
        within = []
        cross = []
        for fp, ep in PAIRS:
            for block in range(16):
                get = lambda p, mode: sum(r["elapsed_ns"] for r in raw[p]
                    if r["case"] == case and r["block"] == block
                    and (mode is None or r["logical_mode"] == mode))
                within.append(get(ep, "O") / get(ep, "S"))
                cross.append(get(fp, None) / (2 * get(ep, "S")))
        threshold = 1.02 if case == "C6" else 1 / 1.01
        summaries.append({
            "case": case,
            "E_off_total_ns": eo, "E_shared_total_ns": es,
            "F_all_modes_total_ns": f, "F_normalization_divisor": 2,
            "E_shared_vs_off_time_change_percent": 100 * (es / eo - 1),
            "E_shared_vs_F_time_change_percent": 100 * (2 * es / f - 1),
            "E_off_vs_F_time_change_percent": 100 * (2 * eo / f - 1),
            "within_paired_block_median_ratio": median(within),
            "cross_paired_block_median_ratio": median(cross),
            "within_ratio_of_sums": eo / es,
            "cross_ratio_of_normalized_sums": f / (2 * es),
            "threshold_speed_ratio": threshold,
            "within_gate_pass": median(within) >= threshold and eo / es >= threshold,
            "cross_gate_pass": median(cross) >= threshold and f / (2 * es) >= threshold,
        })
    saved = json.loads((source / "SCREEN_DECISION.json").read_text())
    assert saved["numeric_screen_passed"] is False
    assert [r["case"] for r in summaries if not r["cross_gate_pass"]] == ["C0"]
    assert all(r["within_gate_pass"] for r in summaries)
    process_c6 = [totals[p]["C6"]["O"] / totals[p]["C6"]["S"]
                  for p in range(8) if LABELS[p] == "E"]
    assert sum(r > 1 for r in process_c6) == 4
    # Twice each difference retains exact integers even if F/2 is a half ns.
    pair_deltas_2ns = [2 * totals[ep]["C0"]["S"] - sum(totals[fp]["C0"].values())
                      for fp, ep in PAIRS]
    c0 = summaries[0]
    decomposition = ((1 + c0["E_off_vs_F_time_change_percent"] / 100)
                     * (1 + c0["E_shared_vs_off_time_change_percent"] / 100) - 1) * 100
    assert math.isclose(decomposition, c0["E_shared_vs_F_time_change_percent"], abs_tol=1e-12)
    return {
        "scope": "Parent independent raw-only recheck; no new performance run or model execution.",
        "source_commit": COMMIT, "raw_sha256": hashes,
        "counts": {"processes": 8, "chunks": 1536, "timed_graph_calls": 49152,
                   "C0_plotted_chunks": 512, "excluded_observations": 0},
        "case_summaries": summaries, "process_O_S_totals_ns": totals,
        "E_C6_process_total_speed_ratios": process_c6,
        "C0_pair_order_F_E": PAIRS,
        "C0_pair_deltas_in_half_nanosecond_units": pair_deltas_2ns,
        "C0_total_E_shared_minus_F_ns": sum(pair_deltas_2ns) / 2,
        "C0_last_two_pairs_share_of_signed_delta_percent": 100 * sum(pair_deltas_2ns[2:]) / sum(pair_deltas_2ns),
        "C0_multiplicative_decomposition_time_change_percent": decomposition,
        "numeric_screen_passed": False,
        "production_adoption": False,
        "limitations": [
            "32 timed calls are averaged in each chunk; no individual graph latency is available.",
            "Four process units per library in one sequential suite, not 49152 independent experiments.",
            "Saved final-output checks cover chunk ends, not every timed intermediate result.",
            "Temporal concentration of a difference does not establish an external cause.",
        ],
    }, raw


def plot(raw: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0), sharex=True, sharey=True)
    colors = {"O": "#176B87", "S": "#D36B2C"}
    for p, ax in enumerate(axes.flat):
        rows = [r for r in raw[p] if r["case"] == "C0"]
        origin = min(r["begin_ns"] for r in rows)
        for row in rows:
            start = (row["begin_ns"] - origin) / 1e9
            end = (row["end_ns"] - origin) / 1e9
            y = row["elapsed_ns"] / 32e6
            ax.plot([start, end], [y, y], color=colors[row["logical_mode"]], linewidth=1.5)
            ax.scatter((start + end) / 2, y, color=colors[row["logical_mode"]], s=10, zorder=3)
        label = "F: both labels run mode 0" if LABELS[p] == "F" else "E: O = off, S = shared"
        ax.set_title(f"Process {p}  |  {label}", fontsize=10, loc="left", pad=9)
        ax.set_xlim(0, 3.35)
        ax.set_ylim(1.30, 2.75)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([0, 1, 2, 3])
    fig.suptitle("C0: all 512 saved chunks, with no exclusions", x=0.06, ha="left", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.06, 0.925, "Each mark is 32 timed graph calls averaged. A horizontal segment spans the saved timing interval.", fontsize=11)
    fig.supxlabel("Seconds since the first timed C0 chunk in each process", y=0.08)
    fig.supylabel("Mean milliseconds per graph within a chunk", x=0.01)
    handles = [Line2D([0], [0], color=colors[m], marker="o", label=f"Logical {m}") for m in ("O", "S")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, 0.01), frameon=False, ncol=2)
    fig.text(0.06, 0.02, "Same axis scales. Source: ff61680 / mixed-run. Temporal clustering does not identify its cause.", fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.16, hspace=0.30, wspace=0.16)
    fig.savefig(output / "C0_ALL_CHUNKS.png", dpi=180, facecolor="white")
    fig.savefig(output / "C0_ALL_CHUNKS.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot", action="store_true", help="Requires matplotlib; no engine execution.")
    args = parser.parse_args()
    report, raw = recheck(args.repo.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "PARENT_RAW_RECHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot(raw, args.out)
    print(json.dumps({"recheck": "passed", "numeric_screen_passed": False,
                      "counts": report["counts"], "summaries": report["case_summaries"]}, indent=2))


if __name__ == "__main__":
    main()
