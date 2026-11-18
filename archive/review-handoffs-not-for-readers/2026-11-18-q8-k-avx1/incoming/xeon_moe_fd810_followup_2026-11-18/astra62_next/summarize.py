#!/usr/bin/env python3
"""Recompute timing summaries and compare every stored output byte from disk."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("raw",type=Path)
parser.add_argument("destination",type=Path)
args=parser.parse_args()
raw=args.raw
dest=args.destination
dest.mkdir(parents=True,exist_ok=True)
protocol=json.loads((here/"protocol.json").read_text())
arms=protocol["variants"]
def read(name):
    with (raw/name).open() as f: return list(csv.DictReader(f))
def write(name, rows):
    with (dest/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
def median(values): return statistics.median(values)

numeric=read("numerical.csv")
dumps=read("dump-sha256.csv")
by_dump=defaultdict(list)
for x in numeric: by_dump[x["selection"],x["mode"],x["variant"]].append(x)
dump_groups=defaultdict(dict)
for x in dumps:
    group=x["selection"],x["mode"]
    v=x["variant"]
    assert v not in dump_groups[group]
    data=gzip.decompress((raw/x["file"]).read_bytes())
    assert hashlib.sha256(data).hexdigest()==x["sha256"]
    rows=by_dump[group+(v,)]
    assert len(rows)==int(x["graphs"]) and len(data)==int(x["bytes"])
    offset=0
    for row in rows:
        assert int(row["offset"])==offset and row["cpu_sha256"]==x["cpu_sha256"]
        end=offset+int(row["bytes"])
        h=14695981039346656037
        for byte in data[offset:end]:
            h=((h^byte)*1099511628211)&0xffffffffffffffff
        assert f"{h:016x}"==row["hash"],(group,v,row["case"],row["phase"])
        offset=end
        cols=int(row["inputs"])
        bits=protocol["selection_masks"][x["selection"]]
        eligible=bool(bits) and row["type"]!="mxfp4" and row["tokens"]=="1" and row["ref"]==row["raw_q8"]=="0"
        route=protocol["routes"][int(row["phase"])]
        selected={i%cols for i,e in enumerate(route) if e>=0 and bits&(1<<e)}
        packed=(len(selected) if v=="09" else cols) if eligible and v!="current" else 0
        blocks_per_col=int(row["k"])//256
        nth=int(row["threads"])
        assert int(row["prepare_blocks"])==packed*blocks_per_col
        assert int(row["prepare_calls"])==packed*(min(blocks_per_col,nth) if v=="09" else nth)
        prepare_round=(1<<protocol["numerical_round_modes"].index(x["mode"])) if packed else 0
        assert int(row["prepare_round_mask"])==prepare_round
        assert bool(int(row["fold_select"]))==(eligible and bool(selected) and v!="current")
        if row["type"]=="mxfp4":
            assert int(row["mx_prepare"])>0 and int(row["mx_dot"])>0
        else:
            assert int(row["mx_prepare"])==int(row["mx_dot"])==0
    assert offset==len(data)
    dump_groups[group][v]=data
assert set(dump_groups)==set(itertools.product(protocol["selection_masks"],protocol["numerical_round_modes"]))
for group, blobs in dump_groups.items():
    assert set(blobs)==set(arms)
    assert blobs["current"]==blobs["08"]==blobs["09"],group

timing=read("timing.csv")
assert len({x["cpu_sha256"] for x in timing})==3
assert len({x["base_sha256"] for x in timing})==len({x["probe_sha256"] for x in timing})==1
for variant in arms:
    ids=lambda items:{(x["cpu_sha256"],x["base_sha256"],x["probe_sha256"]) for x in items if x["variant"]==variant}
    assert len(ids(timing))==1 and ids(timing)==ids(dumps)
groups=defaultdict(lambda:defaultdict(list))
case_groups=defaultdict(list)
for row in timing:
    key=row["case"],row["selection"],int(row["round"])
    groups[key][row["variant"]].append(row)
expected={(c["name"],sel,i) for c in protocol["cases"] for sel in c["masks"] for i in range(len(protocol["timing_orders"]))}
assert set(groups)==expected
info=[]
rounds=[]
for key,by_arm in groups.items():
    case,sel,rid=key
    assert set(by_arm)==set(arms)
    digest={x["output_hash"] for part in by_arm.values() for x in part}
    assert len(digest)==1
    m={}
    for v,part in by_arm.items():
        assert len(part)==protocol["samples"]
        assert {int(x["sample"]) for x in part}==set(range(protocol["samples"]))
        assert {int(x["order"]) for x in part}=={protocol["timing_orders"][rid].index(v)}
        assert {int(x["seed"]) for x in part}=={protocol["seed_base"]+rid}
        m[v]=median([float(x["ns_per_op"]) for x in part])/1000
    row=dict(case=case,selection=sel,round=rid,order="/".join(protocol["timing_orders"][rid]),seed=protocol["seed_base"]+rid,
             current_us=m["current"],arm08_us=m["08"],arm09_us=m["09"],ratio_08_current=m["08"]/m["current"],
             ratio_09_current=m["09"]/m["current"],ratio_09_08=m["09"]/m["08"],hash=digest.pop())
    rounds.append(row)
    case_groups[case,sel].append(row)
for (case,sel),part in case_groups.items():
    summary=dict(case=case,selection=sel,triplets=len(part))
    for k in ["current_us","arm08_us","arm09_us"]:
        summary[k+"_median"]=f"{median([x[k] for x in part]):.3f}"
    for k in ["ratio_08_current","ratio_09_current","ratio_09_08"]:
        vals=[x[k] for x in part]
        summary[k+"_median"]=f"{median(vals):.6f}"
        summary[k+"_range"]=f"{min(vals):.6f}..{max(vals):.6f}"
        summary[k+"_wins"]=sum(value<1 for value in vals)
    info.append(summary)
write("process-triplets.csv",rounds)
write("timing-summary.csv",info)
result=dict(numerical_three_way_graphs=sum(int(x["graphs"]) for x in dumps if x["variant"]=="current"),
    f32_values_per_variant=sum(len(blobs["current"]) for blobs in dump_groups.values())//4,
    dump_files=len(dumps),three_way_dump_streams=len(dump_groups),
    numerical_graph_hashes_checked=len(numeric),per_graph_prepare_counts_match=True,
    unequal_output_bytes_against_current={"08":0,"09":0},
    timing_cases=len(info),timing_triplets=len(rounds),timing_processes=len(rounds)*len(arms),
    timed_samples=len(timing),equal_timing_output_hashes=True)
(dest/"checks.json").write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
