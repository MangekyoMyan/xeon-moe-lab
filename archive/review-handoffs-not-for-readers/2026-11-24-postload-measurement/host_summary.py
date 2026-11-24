"""Attribute saved PCs to case/call windows and reduce the guard's rolling readings."""
import collections
import csv
import json
from pathlib import Path
import statistics
import sys


def read(p):
    return json.loads(p.read_bytes())


def phase_counts(windows, samples):
    total=set(); sockets={};per=[];small=collections.Counter()
    for a,b in windows:
        subset=[s for s in samples if a<=s[0]<b]
        tids={s[1] for s in subset};per.append(len(tids));total.update(tids)
        for _,tid,cpu,sym in subset:
            sockets.setdefault(cpu,set()).add(tid)
            small[sym]+=1
    return {"tids_total":len(total),"tids_by_os_cpu":{str(cpu):sorted(tids) for cpu,tids in sorted(sockets.items())},
            "tids_per_call":per,"sampled_pc_count":sum(small.values()),"symbols":dict(small.most_common(12))}


def quality(run,folder):
    rec=read(run/"RUN.json"); b=rec["blocks"][0]
    assert b["case_id"]=="text_cache_budget_ja"
    libraries=read(run/"LAUNCH.json")["libraries"]
    provider=next(x["path"] for k,x in libraries.items() if k.startswith("libggml-cpu."))
    samples=[]
    with (folder/"worker-samples.tsv").open() as f:
        for r in csv.DictReader(f,delimiter="\t"):
            if r["dso"]!=provider or not any(k in r["symbol"] for k in ("vec_dot","compact_dot","tiled_run","quantize_row","mx_rhs")):
                continue
            assert int(r["pid"])==rec["linux_entry"]["pid"]
            samples.append((int(r["monotonic_ns"]),int(r["tid"]),int(r["cpu"]),r["symbol"]))
    return {"decode":phase_counts(list(zip(b["start_ns"],b["output_ready_ns"])),samples),
            "prefill":phase_counts([(b["prefill_begin_ns"],b["prefill_ready_ns"])],samples)}


def ranges(rows):
    if not rows:return {"sample_count":0}
    columns={
        "package0_C":[v["cpu_package_C"]["S0"] for v in rows],"package1_C":[v["cpu_package_C"]["S1"] for v in rows],
        "all_CPU_max_C":[v["cpu_peak_C"] for v in rows],"gpu_C":[v["gpu"]["gpu_C"] for v in rows],
        "hbm_C":[v["gpu"]["hbm_C"] for v in rows],"gpu_power_W":[v["gpu"]["power_W"] for v in rows],
        "gpu_util_pct":[v["gpu"]["gpu_util"] for v in rows],"available_GiB":[v["available_kib"]/1024**2 for v in rows],
        "fan_pct":[max(v["fan"]["percent"]) for v in rows]}
    return {"sample_count":len(rows),**{k:{"min":min(v),"median":statistics.median(v),"max":max(v)} for k,v in columns.items()}}


def analyze(root,old):
    info=read(root/"guard-archive/end.json")
    rows=[json.loads(line) for line in (root/"guard-archive/guard.jsonl").read_text().splitlines()]
    watch=[json.loads(line) for line in (root/"guard-archive/watch.jsonl").read_text().splitlines()]
    clockoffsets=[s["reader_wall_ns"]-s["reader_mono_ns"] for s in rows]
    grouped=collections.defaultdict(list)
    for s in rows:
        grouped[s["stage"]].append(s)
    result={}
    for stage,items in sorted(grouped.items()):
        assert len(items)==info["unique_guard_samples"][stage] and len({v["unix"] for v in items})==len(items)
        run=root/"trial/runs"/stage;j=read(run/"RUN.json")
        assert all(s["pid"]==j["linux_entry"]["pid"] for s in items)
        def mono(sample):
            return sample["unix"]*1e9-(sample["reader_wall_ns"]-sample["reader_mono_ns"])
        cases={}
        for b in j["blocks"]:
            cases[b["case_id"]]={
                "prefill":ranges([s for s in items if b["prefill_begin_ns"]<=mono(s)<b["prefill_ready_ns"]]),
                "decode_sequence":ranges([s for s in items if b["start_ns"][0]<=mono(s)<b["output_ready_ns"][-1]])}
        result[stage]={"all":ranges(items),"case_windows":cases,
                       "max_guard_sample_interval_s":max((b["unix"]-a["unix"] for a,b in zip(items,items[1:])),default=None)}
    cohorts={}
    for label,run,perf,ev in (("OLD_P12",old/"trial/runs/old-p12-quality-01",old/"quality-host",old/"QUALITY_HOST_EVIDENCE.json"),
            *[(name,root/"trial/runs"/(slug+"-quality-01"),root/"trial/runs"/(slug+"-quality-01")/"quality-host",
               root/"trial/runs"/(slug+"-quality-01")/"QUALITY_HOST_EVIDENCE.json") for name,slug in
              (("NEW_P12","new-p12"),("NEW_D24_B12","new-d24-b12"))]):
        raw=read(ev)
        cohorts[label]={"strict_numeric_pc":quality(run,perf),"weighted_CPU_wide_busy_MHz":raw["effective_clock"]["busy_MHz_estimated"],
                        "usable_MSR_intervals":raw["effective_clock"]["usable_cpu_intervals"],
                        "direct_per_graph_team_size":None}
    return {"guard":result,"all_stages":ranges(rows),"watch_samples":len(watch),"watch_max_dimm_C":max(v["max_dimm_C"] for v in watch),
          "watch_warnings_or_errors":sum(bool(v["alarms"] or v["changes"] or v["sensor_error"]) for v in watch),
          "guard_sample_wall_mono_offset_range_ms":(max(clockoffsets)-min(clockoffsets))/1e6,
          "worker_evidence":cohorts,"limits":"Guard temperatures/utilization are asynchronous point samples. The MSR estimates cover the Japanese quality prefill+decode window and all activity on its chosen CPUs. Numerical PCs are positive evidence at sampled instants, not a complete team census."}


if __name__=="__main__":
    root=Path(sys.argv[1]); old=Path(sys.argv[2])
    data=analyze(root,old);out=root/"HOST_AND_THERMALS.json"
    if "--check" in sys.argv:
        assert data==read(out)
        print("Replayed per-condition worker PCs and the stored guard sensors")
    else:
        out.write_text(json.dumps(data,indent=2)+"\n")
