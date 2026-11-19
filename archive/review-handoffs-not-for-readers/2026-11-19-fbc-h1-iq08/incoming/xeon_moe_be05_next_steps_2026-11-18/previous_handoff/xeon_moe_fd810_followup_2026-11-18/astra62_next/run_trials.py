#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("mode",choices=["numeric","timing"])
p.add_argument("build", type=Path)
p.add_argument("results", type=Path)
a = p.parse_args()
run = a.build.resolve()
out = a.results.resolve()
out.mkdir(exist_ok=True, parents=True)
protocol = json.loads((here/"protocol.json").read_text())
arms=protocol["variants"]
masks=protocol["selection_masks"]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
probe=run/"graph_probe"
counter=run/"count_paths.so"
probe_sha, count_sha = sha(probe), sha(counter)
cpu={v:(run/("lib-"+v)/"libggml-cpu.so.0").resolve(strict=True) for v in arms}
base=(run/"lib-current/libggml-base.so.0").resolve(strict=True)
cpu_sha={v:sha(f) for v,f in cpu.items()}
base_sha=sha(base)
assert len(set(cpu_sha.values()))==3
def invoke(v, selection, cores, args):
    env=os.environ.copy()
    for k in ["LD_PRELOAD","GGML_CPU_IQ_AUDIT_DIR","GGML_CPU_TILED_MODE","OMP_PLACES","GOMP_CPU_AFFINITY"]:
        env.pop(k,None)
    env.update(LD_LIBRARY_PATH=str(run/("lib-"+v)), TEST_EXPERT_MASK=str(masks[selection]),
               GGML_CPU_IQ_ROW_LOCAL="0", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE")
    if a.mode=="numeric": env["LD_PRELOAD"]=str(counter)
    proc=subprocess.run(["taskset","-c",cores,str(probe),*map(str,args)],env=env,text=True,capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"{v} {selection} {args} returned {proc.returncode}: {proc.stderr}")
    rows=[line.split(",") for line in proc.stderr.splitlines()]
    trace={r[1]:Path(r[2]).resolve(strict=True) for r in rows if r[0]=="provider"}
    actual={Path(r[1]).resolve(strict=True) for r in rows if r[0]=="map"}
    if trace!={"cpu":cpu[v],"base":base} or actual!={cpu[v],base}:
        raise RuntimeError(f"Unexpected dladdr/maps for {v}: {proc.stderr}")
    return proc.stdout
def validate(x,v,selection,mode):
    def n(key): return int(x[key])
    qty=n("k")//256
    cols=n("inputs")
    nth=n("threads")
    phases=protocol["routes"][n("phase")]
    hitcols={slot%cols for slot,e in enumerate(phases) if e>=0 and (masks[selection]>>e)&1}
    eligible=(x["type"]!="mxfp4" and n("tokens")==1 and not n("ref") and not n("raw_q8") and bool(masks[selection]))
    if eligible and v=="08": expected=cols; calls=cols*nth
    elif eligible and v=="09": expected=len(hitcols); calls=expected*min(nth,qty)
    else: expected=0; calls=0
    assert n("prepare_blocks")==qty*expected and n("prepare_calls")==calls,(v,selection,x,expected,calls)
    assert n("prepare_round_mask")==((1<<mode) if calls else 0),(v,selection,x)
    assert bool(n("fold_select"))==(eligible and bool(hitcols) and v!="current"),(v,selection,x)
    if x["type"]=="mxfp4":
        assert n("mx_prepare")>0,(v,selection,x)
        if mode==0: assert n("mx_dot")>0,(v,selection,x)
    else: assert n("mx_prepare")==n("mx_dot")==0,(v,selection,x)
def numeric():
    dumpdir=out/"dumps"
    dumpdir.mkdir(exist_ok=False)
    with (out/"numerical.csv").open("x",newline="") as log, (out/"dump-sha256.csv").open("x",newline="") as dlog:
        writer=None
        summary=csv.writer(dlog)
        summary.writerow(["selection","mode","variant","graphs","bytes","sha256","file","cpu_sha256","base_sha256","probe_sha256","counter_sha256"])
        for selection in masks:
            for mode,mode_name in enumerate(protocol["numerical_round_modes"]):
                ref=None
                for v in arms:
                    file=dumpdir/f"{selection}-{mode_name}-{v}.bin"
                    result=invoke(v,selection,"2-4",["numeric",mode,file])
                    rows=list(csv.DictReader(io.StringIO(result)))
                    blob=file.read_bytes()
                    if ref is None: ref=blob
                    elif ref!=blob: raise RuntimeError(f"Different bytes: {selection} {mode} {v}")
                    assert sum(int(row["bytes"]) for row in rows)==len(blob)
                    if writer is None:
                        writer=csv.writer(log)
                        writer.writerow(["selection","mode","variant","cpu_sha256",*rows[0]])
                    for row in rows:
                        validate(row,v,selection,mode)
                        writer.writerow([selection,mode_name,v,cpu_sha[v],*row.values()])
                    gzfile=file.with_suffix(".bin.gz")
                    with gzfile.open("wb") as gz:
                        with gzip.GzipFile(fileobj=gz,mode="wb",filename="",mtime=0) as stream: stream.write(blob)
                    file.unlink()
                    summary.writerow([selection,mode_name,v,len(rows),len(blob),hashlib.sha256(blob).hexdigest(),
                                      "dumps/"+gzfile.name,cpu_sha[v],base_sha,probe_sha,count_sha])
        print("All numerical comparisons and path counts passed.")
def timing():
    with (out/"timing.csv").open("x",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["case","selection","round","order","variant","seed","k","rows","inputs","threads","affinity","loops",
                         "sample","ns_per_op","work_bytes","output_hash","cpu_sha256","base_sha256","probe_sha256"])
        for rid,order in enumerate(protocol["timing_orders"]):
            seed=protocol["seed_base"]+rid
            for c in protocol["cases"]:
                for selection in c["masks"]:
                    hashes=set()
                    for seq,v in enumerate(order):
                        args=["bench",c["k"],c["rows"],c["inputs"],c["threads"],c["loops"],protocol["samples"],seed]
                        rows=list(csv.reader(io.StringIO(invoke(v,selection,c["affinity"],args))))
                        assert len(rows)==protocol["samples"]
                        for sample,ns,size,digest in rows:
                            hashes.add(digest)
                            writer.writerow([c["name"],selection,rid,seq,v,seed,c["k"],c["rows"],c["inputs"],c["threads"],
                                            c["affinity"],c["loops"],sample,ns,size,digest,cpu_sha[v],base_sha,probe_sha])
                        f.flush()
                    assert len(hashes)==1,(c["name"],selection,rid,hashes)
            print("Completed six-position design round",rid+1,flush=True)
if a.mode=="numeric": numeric()
else: timing()
