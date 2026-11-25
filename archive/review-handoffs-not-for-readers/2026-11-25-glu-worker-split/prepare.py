"""Build the measurement executable and fix the block order before seeing timing results."""
from pathlib import Path
import random
import subprocess
from build_cpu import HERE, LAB, read, save, sha


def main():
    build = read(HERE/"BUILD.json")
    original = read(LAB/"fd810-followup-20261118/q8-avx1/FFN_BUILD.json")["command"]
    src=LAB/"mxfp4-hybrid-20261102/hybrid/engine/ggml/src"
    cmd=[str(HERE/"probe.cpp") if s.endswith("/real_ffn.cpp") else
         str(HERE/"probe") if s.endswith("/real_ffn") else
         "-L"+str(HERE/"build/O/runtime") if s.startswith("-L") and "q8-avx1" in s else s for s in original]
    cmd[1:1]=["-I"+str(src),"-I"+str(src/"ggml-cpu")]
    with (HERE/"probe-build.log").open("w") as f:
        subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    files=["probe.cpp","pointwise.hpp","fixture.hpp","pool24.h"]
    save(HERE/"PROBE_BUILD.json",{"command":cmd,"source_sha256":{f:sha(HERE/f) for f in files},"binary_sha256":sha(HERE/"probe")})
    capture_path=LAB/"parallel-pilot-20261117/capture/runs/20261117T052657Z-U1024-3483cb75"
    c=read(capture_path/"CAPTURE.json")
    fixture_files={"CAPTURE.json"}
    for x in c["cells"][:2]:
        fixture_files.update(x[k] for k in ("ids_file","x_file","down_file","routing_weights_file"))
    cap_hash={k:sha(capture_path/k) for k in sorted(fixture_files)}
    reference=read(LAB/"normal24-timeline-20261124/operator/CAPTURE_PIN.json")
    assert all(reference["files"][k]==h for k,h in cap_hash.items())
    rt=read(LAB/"current/runtime.json")
    assert sha(LAB/rt["service_config"]["file"])==rt["service_config"]["sha256"]
    plan=LAB/rt["plan"]["file"];assert sha(plan)==rt["plan"]["sha256"]
    cpu_plan={}
    for line in plan.read_text().splitlines()[1:]:
        cols=line.split()
        if cols[0].startswith("blk.7."):
            ids=list(map(int,cols[6:]));assert len(ids)==int(cols[5]);cpu_plan[cols[0]]=ids
    assert set(cpu_plan)=={x["name"] for x in c["cells"][0]["projections"]}
    shards=[str(LAB/f["file"]) for f in rt["model"]["files"]]
    assert all(Path(path).stat().st_size==f["size"] for path,f in zip(shards,rt["model"]["files"]))
    modes=[(first,order) for first in ["O","G"] for order in ([0,1],[1,0])]*3
    random.Random(20261125).shuffle(modes)
    pairs=[{"pair":i+1,"arms":[first,"O" if first=="G" else "G"],"cells":order} for i,(first,order) in enumerate(modes)]
    design={"source_commit":"a8405cde923d0ea9098999344dccc2f2460196de",
            "normal_profile_sha256":rt["service_config"]["sha256"],"plan_sha256":sha(plan),
            "capture_files_sha256":cap_hash,"probe_sha256":sha(HERE/"probe"),"CPU_build_record_sha256":sha(HERE/"BUILD.json"),
            "common":{"capture_json":str(capture_path/"CAPTURE.json"),"shards":shards,"cpu_plan":cpu_plan,
                      "base":str(LAB/rt["libraries"]["base"]["file"]),"blocks":6,"per_block":16,"trace_calls":6},
            "pairs":pairs,"quality_order":["O","G"],"trace_order":["G","O"],
            "timing":"12 adjacent independent O/G process pairs, six per order, six per cell order. Per process/cell one first-use, 4 warmups, then six blocks of 16 isolated four-node FFNs. Keep every steady call. Primary per-pair difference uses the mean over all 96 calls in each cell and arm. Per-block and per-call data retained; same inputs repeat.",
            "endpoint":"host frozen x and six IDs ready, including both tensor_set calls and the synchronous CPU graph_compute with plan/team, until canonical CPU down is ready; output readback/finite checks outside clock for both arms",
            "diagnostic":"separate CPU graph instrumentation; six full four-node calls per cell, 24 workers per node. Compare max finish minus min start of GLU and do not mix with primary clock.",
            "decision":"Local benefit requires correct outputs and both cells favoring G across process-order and cell-order strata, overall paired bootstrap 95% interval below zero for each cell. Unresolved differences remain inconclusive; no additional rescue runs. Full-model performance or adoption is a separate decision.",
            "physical_cores":list(range(24)),"interleave_nodes":[0,1],"THP_disabled":True,
            "arms":{arm:{mode:v["file"] for mode,v in a["outputs"].items()} for arm,a in build["arms"].items()}}
    target=HERE/"DESIGN.json"
    if target.exists():
        assert read(target)==design,"existing frozen design differs"
    else:
        save(target,design)
    print("Prepared 12 paired process blocks; 4608 primary FFN calls across two cells.")


if __name__=="__main__":
    main()
