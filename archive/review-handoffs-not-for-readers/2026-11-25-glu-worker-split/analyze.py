"""Score archived F32 arrays, per-worker records and fresh O/G timing pairs."""
import argparse
import array
import collections
import csv
import json
import hashlib
import math
import random
import statistics as st
from pathlib import Path


def read(p):
    return json.loads(p.read_bytes())


def require(v,message):
    if not v:raise ValueError(message)


def hashbytes(data):return hashlib.sha256(data).hexdigest()


def stats(values):
    s=sorted(values);n=len(s)
    return {"n":n,"mean":st.mean(s),"median":st.median(s),"p95_nearest_rank":s[math.ceil(0.95*n)-1],
            "min":s[0],"max":s[-1]}


def quality(root):
    runs=[read(root/"runs"/f"quality-{arm}"/"RUN.json") for arm in ("O","G")]
    d=read(root/"DESIGN.json")
    require(all(r["passed"] for r in runs),"quality process")
    result={"passed":True,"cells":[],"compared_f32_values":0,"synthetic":{}}
    for arm,r in zip("OG",runs):
        cases=r["synthetic"]
        require(len(cases)==10 and all(c["arrays_and_guards_bit_exact"] and sum(c["worker_elements"])==c["elements"] for c in cases),"edge and guard checks")
        result["synthetic"][arm]={"cases":len(cases),"single_worker_invocations":sum(c["individual_worker_checks"] for c in cases),
                                  "eligible_shapes_tested_in_channel_mode":[c["name"] for c in cases if c["partition"]=="channels"]}
    for i in range(2):
        a,b=[r["cells"][i] for r in runs]
        require(a["ids"]==b["ids"] and a["weights"]==b["weights"] and a["compact_bytes"]==b["compact_bytes"],"actual weight set")
        require(a["actual_written_per_worker"]==[2304]*6+[0]*18 and b["actual_written_per_worker"]==[576]*24,"actual output assignment")
        hashes={}
        for name in ("gate","up","glu","down"):
            f=a["tensors"][name]["file"]
            raw=(root/"runs/quality-O"/f).read_bytes()
            require(raw==(root/"runs/quality-G"/f).read_bytes(),"O/G differ "+f)
            count=30720 if name=="down" else 13824
            require(len(raw)==4*count and all(math.isfinite(x) for x in array.array("f",raw)),"nonfinite/shape")
            hashes[name]=hashbytes(raw);result["compared_f32_values"]+=count
        require(hashes["down"]==d["capture_files_sha256"][f"cell{i}-down.f32"],"capture mismatch")
        result["cells"].append({"cell":i,"ids":a["ids"],"sha256_equal":hashes,"compact_bytes":a["compact_bytes"],
                                "compact_per_projection":[w["compact_experts"] for w in a["weights"]],
                                "down_equal_to_capture":True,"O_worker_elements":a["actual_written_per_worker"],
                                "G_worker_elements":b["actual_written_per_worker"]})
    return result


def trace(root):
    out={}
    for arm in "OG":
        folder=root/"runs"/f"trace-{arm}"
        run=read(folder/"RUN.json")
        require(run["passed"],"trace process")
        out[arm]={}
        for cell in run["cells"]:
            cs=cell["cell"]; spans=[];worker_times=collections.defaultdict(list); tails=[]
            for p in sorted(folder.glob(f"cell{cs}-*.tsv")):
                with p.open() as f: rows=list(csv.DictReader(f,delimiter="\t"))
                glu=[r for r in rows if r["name"]=="glu" and r["op"]=="GLU"]
                require(len(glu)==24 and {int(r["ith"]) for r in glu}==set(range(24)),"GLU coverage")
                require({r["name"] for r in rows if r["op"]!="graph_end"}=={"gate","up","glu","down"},"graph shape")
                for r in glu:
                    require(r["cpu"]==r["end_cpu"]==r["ith"],"unexpected physical placement")
                    worker_times[int(r["ith"])].append((int(r["finish"])-int(r["start"]))/1000)
                spans.append((max(int(r["finish"]) for r in glu)-min(int(r["start"]) for r in glu))/1000)
                tails.append(int(max(glu,key=lambda r:int(r["finish"]))["ith"]))
            require(len(spans)==6,"trace call count")
            out[arm][str(cs)]={"GLU_envelope_us":stats(spans),"per_call_us":spans,"last_worker":tails,
                               "worker_wrapper_median_us":[st.median(worker_times[t]) for t in range(24)]}
    return {"passed":True,"arms":out,"definition":"same graph, earliest GLU worker start to latest finish before the node barrier; diagnostic CPU objects only"}


def bootstrap(pairs):
    rnd=random.Random(20261125)
    boot=sorted(st.mean(rnd.choices(pairs,k=len(pairs))) for _ in range(20000))
    return [boot[499],boot[19499]]


def timing(root):
    design=read(root/"DESIGN.json")
    records={str(c):[] for c in (0,1)}
    raw=collections.defaultdict(list)
    first=collections.defaultdict(list)
    guards=[]
    for p in design["pairs"]:
        pair=p["pair"];values=collections.defaultdict(dict)
        for arm in p["arms"]:
            folder=root/"runs"/f"pair-{pair:02d}-{arm}"
            r=read(folder/"RUN.json")
            require(r["passed"] and r["arm"]==arm and [v["cell"] for v in r["cells"]]==p["cells"],"arm/case order")
            require([x["cpu"] for x in r["team_after_runs"]]==list(range(24)),"worker topology")
            guards.extend(read(folder/"GUARD.json"))
            for c in r["cells"]:
                ci=str(c["cell"])
                byblock=collections.defaultdict(list)
                stamps=c["stamps"]
                require([v["phase"] for v in stamps]==["first"]+["warmup"]*4+["timed"]*96,"phases")
                for row in stamps:
                    dt=(row["host_down_ready_ns"]-row["input_ready_ns"])/1000
                    require(dt>0,"invalid wall")
                    if row["phase"]=="first":first[arm,ci].append(dt)
                    if row["phase"]=="timed":byblock[row["block"]].append(dt);raw[arm,ci].append(dt)
                require(len(byblock)==6 and all(len(v)==16 for v in byblock.values()),"block counts")
                values[ci][arm]=[st.mean(byblock[k]) for k in range(6)]
        for ci,vals in values.items():
            o,g=st.mean(vals["O"]),st.mean(vals["G"])
            records[ci].append({"pair":pair,"arm_order":"".join(p["arms"]),"cell_order":p["cells"],
                                "O_mean_us":o,"G_mean_us":g,"delta_us_G_minus_O":g-o,
                                "block_mean_us":vals})
    totals={}
    for ci,values in records.items():
        delta=[v["delta_us_G_minus_O"] for v in values]
        strata={k:st.mean(v["delta_us_G_minus_O"] for v in values if fn(v)) for k,fn in [
            ("O_first",lambda v:v["arm_order"]=="OG"),("G_first",lambda v:v["arm_order"]=="GO"),
            ("cell0_first",lambda v:v["cell_order"]==[0,1]),("cell1_first",lambda v:v["cell_order"]==[1,0])]}
        o,g=st.mean(raw["O",ci]),st.mean(raw["G",ci])
        ci95=bootstrap(delta)
        totals[ci]={"O_us":stats(raw["O",ci]),"G_us":stats(raw["G",ci]),"O_first_use_us":stats(first["O",ci]),
                    "G_first_use_us":stats(first["G",ci]),"mean_delta_us":g-o,"G_O_time_change_percent":100*(g/o-1),
                    "paired_mean_delta_us_CI95":ci95,"pairs_favoring_G":sum(v<0 for v in delta),
                    "order_strata_delta_us":strata,"passes_local_gate":ci95[1]<0 and max(strata.values())<0}
    return {"per_cell":totals,"pairs":records,"all_cells_pass":all(s["passes_local_gate"] for s in totals.values()),
            "primary_calls":sum(map(len,raw.values())),"minimum_available_GiB":min(g["MemAvailable_B"] for g in guards)/1024**3,
            "CI_scope":"percentile bootstrap over 12 paired timing processes per case, fixed inputs, not independent prompts"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["quality","trace","timing"])
    parser.add_argument("--check",action="store_true")
    opts=parser.parse_args();root=Path(__file__).resolve().parent
    result={"quality":quality,"trace":trace,"timing":timing}[opts.stage](root)
    path=root/(opts.stage.upper()+".json")
    if opts.check:require(read(path)==result,"saved result differs")
    else:path.write_text(json.dumps(result,indent=2)+"\n")
    if opts.stage=="timing":
        for k,v in result["per_cell"].items():print(k,{n:v[n] for n in ("mean_delta_us","G_O_time_change_percent","paired_mean_delta_us_CI95","pairs_favoring_G","order_strata_delta_us","passes_local_gate")})
    else:print(opts.stage,"passed")
