"""Align actual thread runtimes with the content-receive window of the one normal request."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def read(p):
    return json.loads(p.read_bytes())


def select(p):
    summary=read(p/"summary.json")
    rows=[json.loads(line) for line in (p/"events.jsonl").read_text().splitlines()]
    timing=[e["json"]["timings"] for e in rows if isinstance(e.get("json"),dict) and "timings" in e["json"]]
    assert len(timing)==1
    return {"client_first_content_ms":summary["ttft_content_receive_ms"],"eof_ms":summary["elapsed_from_request_start"]["eof_ms"],
            "reported_usage":summary["usage_reports"][-1]["usage"],"server_timings":timing[0],
            "content_sha256":hashlib.sha256((p/"content.txt").read_bytes()).hexdigest(),
            "effective_request_sha256":hashlib.sha256((p/"request.effective.json").read_bytes()).hexdigest()}


def analyze(root,old,topology_root=None):
    if topology_root is None:
        topology_root=root
    t=read(root/"stream/summary.json")["perf_counter_ns"]
    rows=read(root/"normal-scheduling/samples.json")
    eligible=[r for r in rows if t["first_content"]<=r["begin"]<=r["end"]<=t["last_content"]]
    assert len(eligible)>=2
    topology=read(topology_root/"production-before/snapshot.json")["hardware"]["cpu"]["sysfs_topology"]
    # Recover socket mapping from the verified host lscpu CSV; its 'core'
    # column need not have the same numbering as the hardware core_id.
    csvrows=read(topology_root/"production-before-live.json")["CPU_socket_core_node"]
    cpu_socket={int(s.split(",")[0]):int(s.split(",")[1]) for s in csvrows if not s.startswith("#")}
    bycpu=defaultdict(int);byt=defaultdict(lambda:defaultdict(int)); total=ambiguous=0
    for a,b in zip(eligible,eligible[1:]):
        assert a["end"] < b["begin"]
        left={(v[0],v[1]):v for v in a["tasks"]}
        for row in b["tasks"]:
            p=left.get((row[0],row[1]))
            if p is None:
                continue
            ns=row[3]-p[3]
            assert ns>=0
            total+=ns
            if row[4]!=p[4] or len(row[4])!=1 or row[2]!=p[2] or row[4][0]!=row[2]:
                ambiguous+=ns
                continue
            if ns:
                bycpu[row[2]]+=ns
                byt[row[0]][row[2]]+=ns
    bysocket={s:sum(n for c,n in bycpu.items() if cpu_socket[c]==s) for s in sorted(set(cpu_socket.values()))}
    assert sum(bycpu.values())+ambiguous==total and topology
    first,last=eligible[0],eligible[-1]
    return {"old_normal_request":select(old),"new_normal_request":select(root/"stream"),
        "normal_sampler":read(root/"normal-scheduling/run.json"),"qualifying_snapshots":len(eligible),
        "intervals_entirely_within_first_to_last_content":len(eligible)-1,
        "coverage_elapsed_s":(last["end"]-first["begin"])/1e9,
        "client_content_span_s":(t["last_content"]-t["first_content"])/1e9,
        "matched_thread_runtime_ns":total,"runtime_ns_on_threads_with_same_single_cpu_at_both_endpoints":dict(bycpu),
        "pinned_thread_runtime_ns_by_socket":bysocket,"other_or_ambiguous_runtime_ns":ambiguous,
        "pinned_by_tid_cpu_ns":{tid:dict(row) for tid,row in byt.items()},
        "source":"Per-TID schedstat deltas and affinity snapshots, not instruction samples. Records only matching task identity. Temporary migrations between samples are unobserved."}


if __name__=="__main__":
    root=Path(sys.argv[1])
    data=analyze(root,Path(sys.argv[2]))
    dest=root/"NORMAL_STREAM_COMPARISON.json"
    if "--check" in sys.argv:
        assert json.dumps(data,sort_keys=False)==json.dumps(read(dest),sort_keys=False)
        print("Normal request, server timings and scheduled thread accounting reproduced")
    else:
        dest.write_text(json.dumps(data,indent=2)+"\n")
