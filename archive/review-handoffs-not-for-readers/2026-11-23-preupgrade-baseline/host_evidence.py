"""Reduce stock Linux perf evidence for the Japanese quality case, outside all primary clocks."""
import argparse
import csv
import json
import re
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
from measure import read, dump

def rows_delta(before,after):
    old={(r["tid"],r["start_ticks"]):r for r in before["tasks"]}
    data=[]
    for end in after["tasks"]:
        start=old.get((end["tid"],end["start_ticks"]))
        if start:
            data.append({"tid":end["tid"],"runtime_ns_delta":end["runtime_ns"]-start["runtime_ns"],
                         "affinity_before":start["affinity"],"affinity_after":end["affinity"]})
    return data

def parse_pcs(source,target):
    regex=re.compile(r'^\s*(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):\s+[0-9a-fA-F]+\s+(.*?)\s+\((.*)\)\s*$')
    total=unparsed=0
    with source.open() as f,target.open("x") as out:
        data=csv.writer(out,delimiter="\t",lineterminator="\n")
        data.writerow(["pid","tid","cpu","monotonic_ns","symbol","dso"])
        for line in f:
            if not line.strip():continue
            total+=1
            m=regex.fullmatch(line)
            if m is None:
                unparsed+=1
                continue
            pid,tid,cpu,t,symbol,dso=m.groups()
            data.writerow([pid,tid,int(cpu),int(Decimal(t)*1000000000),symbol,dso])
    return {"input_sample_lines":total,"unparsed_rows":unparsed,"kept":total-unparsed}

def analyze(root):
    run=root/"trial/runs/old-p12-quality-01"
    q=read(run/"RUN.json");b=q["blocks"][0]
    assert b["case_id"]=="text_cache_budget_ja"
    mark=q["ja_quality_task_observations"]
    timestamp_pairs={
        "prefill":[(b["prefill_begin_ns"],b["prefill_ready_ns"])],
        "decode":list(zip(b["start_ns"],b["output_ready_ns"],strict=True))}
    cpu=next(v["path"] for k,v in read(run/"LAUNCH.json")["libraries"].items() if k.startswith("libggml-cpu."))
    pcs=root/"quality-host/worker-samples.tsv"
    phases={}
    samples=[]
    if pcs.exists():
        with pcs.open() as f:samples=list(csv.DictReader(f,delimiter="\t"))
    for phase,windows in timestamp_pairs.items():
        deltas=rows_delta(mark["before_"+phase],mark["after_"+phase])
        all_tids=defaultdict(Counter)
        numeric=defaultdict(Counter)
        counts=Counter()
        window_sets=[set() for _ in windows]
        samples_total=0
        for r in samples:
            tick=int(r["monotonic_ns"])
            wi=next((i for i,(lo,hi) in enumerate(windows) if lo<=tick<hi),None)
            if wi is None:continue
            assert int(r["pid"])==q["linux_entry"]["pid"]
            samples_total+=1
            if r["dso"]!=cpu:continue
            tid=int(r["tid"]);core=int(r["cpu"]);symbol=r["symbol"]
            all_tids[tid][core]+=1;counts[symbol]+=1
            if any(v in symbol for v in ("vec_dot","compact_dot","tiled_run","ggml_compute_forward","quantize_row","mx_rhs")):
                numeric[tid][core]+=1;window_sets[wi].add(tid)
        phases[phase]={"task_deltas":deltas,"sampled_pid_pc_count":samples_total,"sampled_cpu_provider_tids":dict(all_tids),
             "sampled_numeric_tids_per_cpu":dict(numeric),"distinct_tids_sampled_in_cpu_numeric_symbols":len(numeric),
             "per_call_distinct_sampled_numeric_tids":[len(v) for v in window_sets],
             "cpu_dso_symbol_counts":counts,"direct_per_parallel_region_team_size":None}
    msr_file=root/"quality-host/msr.csv"
    rows=defaultdict(dict)
    if msr_file.exists():
        with msr_file.open() as f:
            for r in csv.reader(f,delimiter=";"):
                if len(r)<7 or not r[1].startswith("CPU"):continue
                when,core=r[0].strip(),int(r[1][3:])
                event=r[4]
                if not r[2].isdigit():continue
                rows[(when,core)][event]={"count":int(r[2]),"running_ns":int(r[5]),"running_percent":float(r[6])}
    valid=[];missing=[]
    for (t,c),v in rows.items():
        if set(v)!={"msr/aperf/","msr/mperf/","msr/tsc/"} or min(x["running_percent"] for x in v.values())<99.0 or min(x["running_ns"] for x in v.values())<=0:
            missing.append([t,c]);continue
        ts=v["msr/tsc/"]; ap=v["msr/aperf/"]["count"];mp=v["msr/mperf/"]["count"]
        if not ts["count"] or not mp:
            missing.append([t,c]);continue
        tsc_ghz=ts["count"]/ts["running_ns"]
        valid.append({"recorder_elapsed_s":float(t),"cpu":c,"aperf":ap,"mperf":mp,"tsc":ts["count"],
                      "tsc_running_ns":ts["running_ns"],"tsc_reference_MHz":tsc_ghz*1000,
                      "active_ns_estimated":mp/tsc_ghz,"busy_MHz_estimated":1000*ap*tsc_ghz/mp})
    def busy(subset):
        return 1000*sum(r["aperf"] for r in subset)/sum(r["active_ns_estimated"] for r in subset) if subset else None
    return {"trace_context":"One quality process; phase/T1 windows use CLOCK_MONOTONIC. Numeric samples prove execution only at sampled instants. Per-graph complete team membership is unobserved.",
            "phases":phases,"effective_clock":{"scope":"Entire low-rate APERF/MPERF/TSC observation over the combined Japanese quality prefill and 64 decode calls, including short observer boundaries. CPU-wide, selected cores. Active-time weighted across cores.",
            "method":"Each interval: TSC reference GHz = TSC cycles / perf running nanoseconds. Estimated active ns = MPERF / TSC reference GHz. Combined MHz = 1000 * sum(APERF) / sum(active ns).",
            "busy_MHz_estimated":busy(valid),"by_cpu_MHz":{str(c):busy([r for r in valid if r["cpu"]==c]) for c in sorted({r["cpu"] for r in valid})},
            "usable_cpu_intervals":len(valid),"incomplete_or_zero_intervals":missing,"raw_counts_and_rates":valid}}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("--parse",type=Path)
    a=p.parse_args()
    if a.parse:
        m=parse_pcs(a.parse,a.root/"quality-host/worker-samples.tsv")
        dump(a.root/"quality-host/PC_DECODE.json",m)
    out=analyze(a.root)
    dump(a.root/"QUALITY_HOST_EVIDENCE.json",out)
    print(json.dumps({"sampled_workers":{k:v["distinct_tids_sampled_in_cpu_numeric_symbols"] for k,v in out["phases"].items()},"busy_MHz_estimated":out["effective_clock"]["busy_MHz_estimated"],"cpu_intervals":out["effective_clock"]["usable_cpu_intervals"]}))
