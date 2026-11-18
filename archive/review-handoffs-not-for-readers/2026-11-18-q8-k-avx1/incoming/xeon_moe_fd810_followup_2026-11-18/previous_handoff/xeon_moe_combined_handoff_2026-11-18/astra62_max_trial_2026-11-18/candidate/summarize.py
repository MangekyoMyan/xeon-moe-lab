#!/usr/bin/env python3
"""Summarize per-process medians and paired candidate/baseline elapsed-time ratios."""
import csv, sys
from collections import defaultdict
from statistics import median

rows = list(csv.DictReader(open(sys.argv[1])))
groups = defaultdict(lambda: defaultdict(list))
hashes = defaultdict(set)
for r in rows:
    groups[r["case"], int(r["round"])][r["variant"]].append(float(r["ns_per_op"]))
    hashes[r["case"],r["round"],r["sample"]].add(r["output_hash"])
assert all(len(hs) == 1 for hs in hashes.values()), "A/B output hashes differ"
cases = list(dict.fromkeys(r["case"] for r in rows))
w = csv.writer(sys.stdout)
w.writerow(["case","pairs","base_median_us","candidate_median_us","median_paired_time_ratio",
            "min_paired_ratio","max_paired_ratio","candidate_faster_pairs","all_paired_ratios"])
for c in cases:
    base, alt, rat = [], [], []
    for (name, _), variants in sorted(groups.items()):
        if name != c: continue
        assert len(variants) == 2 and "baseline" in variants
        bm = median(variants["baseline"])
        am = median(next(v for k,v in variants.items() if k!="baseline"))
        base.append(bm/1000); alt.append(am/1000); rat.append(am/bm)
    w.writerow([c,len(rat),round(median(base),3),round(median(alt),3),round(median(rat),4),
                round(min(rat),4),round(max(rat),4),sum(x<1 for x in rat)," ".join(f"{x:.4f}" for x in rat)])
