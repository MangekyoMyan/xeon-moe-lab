"""Map sampled OpenMP spins into the same worker's instrumented time intervals."""
import bisect
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import sys

from score_workers import load_table, read

ROOT=Path(__file__).resolve().parent
SO=Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")


def get_frames(offsets):
    debug=ROOT.parent/"normal24-timeline-20261124/private/libgomp.debug"
    ids=[re.search(r"Build ID: (\w+)",subprocess.check_output(["readelf","-n",str(p)],text=True,stderr=subprocess.DEVNULL))[1] for p in (debug,SO)]
    assert ids[0]==ids[1]=="abac9c46bfe169c31d97018db81b5386bdf38a17"
    cmd=["/usr/bin/addr2line","-a","-f","-C","-i","-e",str(debug)]
    output=subprocess.check_output(cmd,input="".join(f"{x:x}\n" for x in sorted(offsets)),text=True)
    fields=re.split(r"(?m)^0x([0-9a-f]+)\n",output)
    return {int(k,16):v.rstrip("\n").splitlines()[::2] for k,v in zip(fields[1::2],fields[2::2])}


def records():
    att=read(ROOT/"PC_ATTACHMENT.json")
    mapping=read(ROOT/"private/gomp-map.json")
    assert mapping["pid"]==att["pid"] and mapping["start_ticks"]==att["start_ticks"]
    row=mapping["mapping"]
    base,top=map(lambda x:int(x,16),row[0].split("-"));off=int(row[2],16)
    ph=[s.split() for s in subprocess.check_output(["readelf","-lW",str(SO)],text=True).splitlines() if s.lstrip().startswith("LOAD ") and " E " in s]
    assert len(ph)==1;seg_file,seg_va=(int(ph[0][i],16) for i in (1,2))
    rx=re.compile(r"^\s*(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):\s+([a-fA-F0-9]+)\s+(.*?)\s+\((.*)\)\s*$")
    run=ROOT/"runs/worker-trace";j=read(run/"RUN.json")
    windows=sorted((b["start_ns"][step-1],b["output_ready_ns"][step-1],b["case_index"],step)
                   for b in j["blocks"] for step in range(5,65,8) if b["start_ns"][step-1]>att["begin_ns"] and b["output_ready_ns"][step-1]<att["end_ns"])
    assert len(windows)>=14 and all(end<=aa for (_,end,_,_), (aa,_,_,_) in zip(windows,windows[1:]))
    begins=[v[0] for v in windows]; out=[];offsets=set()
    for line in (ROOT/"private/workers.perf.txt").open():
        if not line.strip() or line.startswith("#"):continue
        m=rx.fullmatch(line)
        if not m:
            if "LOST" in line:raise ValueError(line[:120])
            raise ValueError("unparsed perf row: "+line[:140])
        pid,tid,cpu,t,addr,sym,dso=m.groups()
        assert int(pid)==att["pid"]
        ns=int(Decimal(t)*1_000_000_000)
        ix=bisect.bisect_right(begins,ns)-1
        if ix<0 or ns>=windows[ix][1]:continue
        rel=None
        if dso==str(SO):
            address=int(addr,16);assert base<=address<top
            rel=address-base+off-seg_file+seg_va;offsets.add(rel)
        out.append((windows[ix][2],windows[ix][3],int(tid),int(cpu),ns,Path(dso).name,sym,rel))
    frames=get_frames(offsets)
    with (ROOT/"cpu-PC.tsv").open("w") as f:
        wr=csv.writer(f,delimiter="\t",lineterminator="\n")
        wr.writerow(("case","step","tid","cpu","ns","dso","symbol","offset"))
        for *fields,rel in out:wr.writerow([*fields,hex(rel) if rel is not None else ""])
    (ROOT/"GOMP_FRAMES.json").write_text(json.dumps({hex(k):v for k,v in frames.items()},indent=2)+"\n")


def stream_intervals(table):
    workers=collections.defaultdict(list);graph_ranges=[]
    graphs=collections.defaultdict(list)
    for (g,_),rows in table.items():
        for r in rows:
            graphs[g].append((r["start"],r["released"]))
            workers[r["tid"]].append(r)
    for r in graphs.values():graph_ranges.append((min(v[0] for v in r),max(v[1] for v in r)))
    return {k:(sorted(v,key=lambda x:x["start"]),sorted(r["start"] for r in v)) for k,v in workers.items()}, sorted(graph_ranges)


def analyze(root):
    frames=read(root/"GOMP_FRAMES.json");run=root/"runs/worker-trace"
    counts=collections.Counter();where=collections.Counter();ratios=collections.defaultdict(collections.Counter)
    attribution=collections.defaultdict(collections.Counter);tables={}
    for r in csv.DictReader((root/"cpu-PC.tsv").open(),delimiter="\t"):
        key=(int(r["case"]),int(r["step"]));tid=int(r["tid"]);ns=int(r["ns"])
        if key not in tables:tables[key]=stream_intervals(load_table(run/f"worker-{key[0]}-{key[1]}.tsv"))
        tids,gr=tables[key]
        origin="outside_all_logged_cpu_graph_work"
        if any(a<=ns<=b for a,b in gr):origin="CPU_graph_team_or_worker_gap"
        if tid in tids:
            rows,starts=tids[tid];ix=bisect.bisect_right(starts,ns)-1
            if ix>=0:
                v=rows[ix]
                if ns<=v["released"]:
                    if v["prep_in"] and v["prep_in"]<=ns<=v["prep_out"]:origin="MUL_MAT_ID_preparation_barrier"
                    elif v["sync"] and ns>=v["finish"]:origin="graph_end_barrier" if v["op"]=="graph_end" else "after_"+v["op"]+"_barrier"
                    else:origin="inside_worker_node_or_bookkeeping"
        group="other"
        if r["offset"]:
            stack=frames[r["offset"]]
            if any("spin" in x for x in stack) and any("barrier" in x for x in stack):group="OpenMP_barrier_spin"
            else:group="other_libgomp"
        elif r["dso"].startswith("libggml-cpu") and any(x in r["symbol"] for x in ("compact_dot","mxfp4_rhs_dot","vec_dot","tiled_run","quantize_row")):
            group="CPU_numerics"
        counts[group]+=1;attribution[group][origin]+=1
        if group=="OpenMP_barrier_spin":where[origin]+=1
        ratios[str(key)][group]+=1
    assert len(tables)>=14 and sum(counts.values())>=1000
    nb=sum(where.values());assert nb>100
    return {"sample_windows":len(tables),"user_cpu_samples":sum(counts.values()),"counts":dict(counts),
        "spin_by_location":dict(where),"percent_of_spin":{k:100*v/nb for k,v in where.items()},
        "by_group_and_interval":{k:dict(v) for k,v in attribution.items()},"individual_counts":{k:dict(v) for k,v in ratios.items()},
        "definition":"On-CPU PC periods only; outside all intervals is outside the measured CPU backend worker graph from its first node entry to its last barrier return. It can include thread-pool wait, CPU submission, GPU work or inter-graph overhead. It does not measure GPU activity directly."}


if __name__=="__main__":
    if sys.argv[1]=="decode":records()
    else:
        root=Path(sys.argv[1]);result=analyze(root);p=root/"PC_RESULT.json"
        if "--check" in sys.argv:assert result==read(p)
        else:p.write_text(json.dumps(result,indent=2)+"\n")
        print(result["sample_windows"],result["counts"],result["percent_of_spin"])
