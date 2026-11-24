"""Reclassify the archived samples with relative ELF PCs and the matched inline frames."""
import collections
import csv
import json
from pathlib import Path
import sys


def check(root):
    data=json.loads((root/"CPU_SAMPLES.json").read_text());d=root/"runs/observe"
    run=json.loads((d/"RUN.json").read_text())
    for c,b in enumerate(run["blocks"]):
        windows=list(zip(b["start_ns"][4:],b["output_ready_ns"][4:]))
        with (d/f"gomp-relative-{c}.tsv").open() as f:
            offsets={(int(r["monotonic_ns"]),int(r["tid"]),int(r["cpu"])):r["relative_elf_va"] for r in csv.DictReader(f,delimiter="\t")}
        counts=collections.Counter();percpu=collections.defaultdict(collections.Counter);hist=collections.Counter();tids=set()
        with (d/f"pc-{c}.tsv").open() as f:
            for row in csv.DictReader(f,delimiter="\t"):
                t,tid,cpu=(int(row[k]) for k in ("monotonic_ns","tid","cpu"));sym=row["symbol"];dso=Path(row["dso"]).name
                if not any(a<=t<z for a,z in windows):continue
                if dso.startswith("libgomp"):
                    off=offsets.pop((t,tid,cpu));hist[off]+=1
                    names=[x[0].lower() for x in data["resolved_relative_elf_offsets"][off]]
                    wait=any("wait" in n for n in names);spin=any("spin" in n for n in names);barrier=any("barrier" in n for n in names)
                    if spin and barrier:label="OpenMP barrier spin"
                    elif wait and barrier:label="OpenMP barrier wait code (outside named spin)"
                    elif wait or spin:label="OpenMP other wait/spin code"
                    else:label="OpenMP other"
                elif dso.startswith("libggml-cpu") and any(s in sym for s in ("compact_dot","mx_rhs","mxfp4_rhs","vec_dot","tiled_run","quantize_row")):
                    label="CPU numerical kernels";tids.add(tid)
                elif dso.startswith("libggml-cpu"):label="other CPU backend"
                else:label="other libraries"
                counts[label]+=1;percpu[str(cpu)][label]+=1
        ref=data["cases"][b["case_id"]]
        assert not offsets and counts==ref["categories"] and percpu==ref["per_cpu"] and hist==ref["gomp_elf_va_counts"]
        assert sum(counts.values())==ref["total_within_T1_clocks"] and len(tids)==ref["numeric_worker_tids"]
    return [r["total_within_T1_clocks"] for r in data["cases"].values()]


if __name__=="__main__":print("Verified CPU PC samples",check(Path(sys.argv[1])))
