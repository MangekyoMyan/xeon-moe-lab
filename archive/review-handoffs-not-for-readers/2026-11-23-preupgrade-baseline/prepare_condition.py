"""Freeze a measurement spec from the CPU topology visible on this host."""
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent

def read(p): return json.loads(p.read_text())
def write(p,j):
    with p.open("x") as f:
        json.dump(j,f,indent=2,ensure_ascii=False)
        f.write("\n")
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    protocol=read(ROOT/"intake/BASELINE_PROTOCOL.json")
    allowed=sorted(os.sched_getaffinity(0))
    cores=defaultdict(dict)
    for cpu in allowed:
        d=Path(f"/sys/devices/system/cpu/cpu{cpu}")
        socket=int((d/"topology/physical_package_id").read_text())
        core=int((d/"topology/core_id").read_text())
        nodes=sorted(int(n.name[4:]) for n in d.glob("node[0-9]*"))
        record=cores[socket].setdefault(core,{"socket":socket,"core":core,"available_siblings":[],"nodes":nodes})
        record["available_siblings"].append(cpu)
    actual=read(ROOT/"production-before-live.json")
    assert "E5-2630L v2" in actual["CPU_model"] and len(cores)==2 and all(len(s)==6 for s in cores.values())
    selected=[min(group["available_siblings"]) for sock in sorted(cores) for _,group in sorted(cores[sock].items())[:6]]
    selected.sort()
    nodes=sorted({node for sock in cores.values() for c in sock.values() for node in c["nodes"]})
    assert nodes==[0,1] and len(selected)==len(set(selected))==12
    mask=hex(sum(1<<n for n in selected))
    condition={"name":"OLD_P12","decode_threads":12,"prefill_threads":12,"projector_threads":12,"decode_cpu_ids":selected,
               "prefill_cpu_ids":selected,"process_cpu_ids":selected,"decode_mask":mask,"prefill_mask":mask,"strict":True,
               "interleave_nodes":nodes,"OMP_THREAD_LIMIT":24,"packing_threads_at_load":12}
    write(ROOT/"TOPOLOGY.json",{"observed_utc":datetime.now(timezone.utc).isoformat(),"host":platform.node(),"cpu":actual["CPU_model"],
        "available_before_binding":allowed,"cores":[v for s in sorted(cores) for _,v in sorted(cores[s].items())],"measurement":condition})
    cases=ROOT/"trial/CASES.json"
    cases.write_bytes((ROOT/"intake/reference/CASES_633e.json").read_bytes())
    assert sha(cases)==protocol["fixed_cases_sha256"]
    labels=[]
    for row in protocol["runs"]:
        labels.append({"name":row["run_id"],"stage":row["stage"],"cases":row["case_indices"],"rows":row["save_rows_per_case"],
                       "condition":condition,"quality_host_observation":row["stage"]=="quality"})
    write(ROOT/"trial/DESIGN.json",{"condition":condition,"stages":labels,"protocol_sha256":sha(ROOT/"intake/BASELINE_PROTOCOL.json"),
        "fixed_before_runs":True,"source_commit":protocol["source_commit"],"quality_reference":"first new OLD_P12 run; historical comparison recorded independently",
        "timing_reference":"same P12 quality last row, exact","timing_clock":"llama_decode to llama_get_logits_ith(-1) return",
        "capture_policy":"CPU task snapshots and optional perf only in quality; OS page cache retained"})
    write(ROOT/"trial/STATE.json",{"phase":"prepared","runs":{},"completed":False})
if __name__=="__main__":main()
