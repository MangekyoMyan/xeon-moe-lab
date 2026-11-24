"""Small cross-checks and reporting inputs for the measured stages."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_bytes())


def compute(root):
    trace=read(root/"TIMELINE.json")
    cpu=read(root/"CPU_SAMPLES.json")
    mem=read(root/"MEMORY_REFERENCE.json")
    op=read(root/"OPERATOR.json")
    pieces={}
    for label,v in mem["runs"].items():
        assert v["read"]["checksum_passed"] and sorted(v["read"]["cpus"])==list(range(24))
        pieces[label]={"median_GB_s":statistics.median(v["read"]["gb_s"]),"readings":v["read"]["gb_s"]}
    calls=trace["calls"]
    for c in calls:assert c["CPU_only_ns"]+c["device_only_ns"]+c["overlap_ns"]+c["neither_ns"]==c["span_ns"]
    summary={}
    labels=["control","observe","operator","whole-F1","whole-S1","whole-S2","whole-F2","pp-A1","pp-L","pp-H","pp-A2"]
    decision=read(root/"prefix_quality/DECISION.json")
    if decision["run"]:labels.extend(["qpf-64","qpf-"+str(decision["candidate"])])
    for label in labels:
        d=root/"runs"/label
        rows=[json.loads(s) for s in (d/"guard-samples.jsonl").read_text().splitlines()]
        gg=[v for v in rows if v["source"]=="guard"];ww=[v for v in rows if v["source"]=="watch"]
        assert gg and ww
        summary[label]={"guard_samples":len(gg),"watch_samples":len(ww),"cpu_peak_C":max(x["cpu_max_C"] for x in gg),
            "gpu_peak_C":max(x["gpu_C"] for x in gg),"hbm_peak_C":max(x["hbm_C"] for x in gg),
            "fan_peak_pct":max(x["fan_max_pct"] for x in gg),"DIMM_peak_C":max(x["max_dimm_C"] for x in ww),
            "min_available_GiB":min(x["available_kib"] for x in gg)/1024**2,"flagged_samples":sum(x["alert"] or x["error"] for x in ww)}
    secs={}
    for stage in ("control","observe"):
        j=read(root/"runs"/stage/"RUN.json")
        for k,e in j["task_activity"].items():
            left=e["before_late_decode"];right=e["after_late_decode"]
            a={(v["tid"],v["start_ticks"]):v for v in left["tasks"]};total=0;queued=0
            for v in right["tasks"]:
                v0=a.get((v["tid"],v["start_ticks"]))
                if v0:
                    total+=v["runtime_ns"]-v0["runtime_ns"];queued+=v["runqueue_ns"]-v0["runqueue_ns"]
            secs[stage+"-"+k]={"elapsed_s":(right["end_ns"]-left["begin_ns"])/1e9,"matched_worker_cpu_s":total/1e9,
                "matched_runnable_queue_s":queued/1e9,"scope":"Sum over matched TIDs, includes code between T1 clocks. Not isolated barrier or off-CPU time."}
    transfers={}
    for direction,kind in (("1","HtoD"),("2","DtoH"),("8","DtoD")):
        sizes=[x["copy_bytes"].get(direction,0) for x in calls]
        transfers[kind]={"min_bytes":min(sizes),"median_bytes":statistics.median(sizes),"max_bytes":max(sizes)}
    partitions = {}
    for case, evidence in cpu["cases"].items():
        per_cpu = evidence["per_cpu"]
        assert {int(core) for core in per_cpu} == set(range(24))
        partitions[case] = {}
        for socket in (0, 1):
            group = Counter()
            for core in range(socket * 12, (socket + 1) * 12):
                group.update(per_cpu[str(core)])
            partitions[case][str(socket)] = {"on_cpu_samples": group.total(),
                "numeric_samples": group["CPU numerical kernels"], "barrier_spin_samples": group["OpenMP barrier spin"],
                "barrier_spin_pct": 100 * group["OpenMP barrier spin"] / group.total()}
    return {"memory_reference":pieces,"thermal_by_run":summary,"trace_transfer_volumes":transfers,"late_decode_task_accounting":secs,
      "on_cpu_by_physical_socket": partitions,
      "matching_control_and_trace_hashes":read(root/"runs/control/VERIFIED.json")["output_sha256"]==read(root/"runs/observe/VERIFIED.json")["output_sha256"],
      "all_traced_cpu_device_intersections_zero":all(c["overlap_ns"]==0 for c in calls),
      "cpu_spin_pct":{k:v["percent"]["OpenMP barrier spin"] for k,v in cpu["cases"].items()},
      "L7_H1_noncompact_over_compact_median_C_minus_S":op["paired_by_membership_and_hits"]["85-H1"]["C_minus_S_ms"]["median_ms"]/op["paired_by_membership_and_hits"]["331-H1"]["C_minus_S_ms"]["median_ms"]}


if __name__=="__main__":
    root=Path(sys.argv[1]);v=compute(root);p=root/"CHECKS.json"
    if "--check" in sys.argv:assert v==read(p)
    else:p.write_text(json.dumps(v,indent=2)+"\n")
