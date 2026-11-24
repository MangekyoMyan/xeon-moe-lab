"""Recompute layer-7 within-block effects, cold slots and dequantized references."""
import array
import collections
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_bytes())
def median(v):return statistics.median(v)
def ms(x):return x/1e6


def f32(path):
    raw=path.read_bytes();values=array.array("f");values.frombytes(raw)
    if sys.byteorder!="little":values.byteswap()
    assert values.itemsize==4 and all(math.isfinite(x) for x in values)
    return values


def error(x,y):
    assert len(x)==len(y)
    energy=math.fsum(a*a for a in y);sse=math.fsum((a-b)**2 for a,b in zip(x,y))
    return {"nmse":sse/energy if energy else None,"max_abs":max(abs(a-b) for a,b in zip(x,y))}


def analyze(root):
    d=root/"runs/operator";r=read(d/"RUN.json")
    assert r["passed"] and (r["cparams"]["threads"],r["cparams"]["threads_batch"],r["projector_threads_argument"])==(24,24,24)
    assert len({(v["socket"],v["core"]) for v in r["entry"]["cores"]})==24
    all_periods=[];overall=collections.defaultdict(lambda:collections.defaultdict(list))
    gearnames=("O","C","S","P")
    for idx,third in enumerate((331,85,85,331),1):
        period=d/f"period-{idx}";rep=read(period/"REPLAY.json")
        assert rep["passed"] and rep["private_initialized_layers"]==[7] and rep["resident_ids_in_physical_order"]==[19,154,third]
        assert rep["bank_native_bytes_verified"]==rep["bank_native_bytes"]==36218880
        for k in ("O_worker_receipt","CSP_worker_receipt"):
            assert {v["cpu"] for v in rep[k]}==set(range(24)) and len({v["tid"] for v in rep[k]})==24
        cells=[]
        for c,cell in enumerate(rep["cells"]):
            assert cell==read(period/f"cell{c}.json") and cell["cell"]==cell["matched_H"]==c
            assert cell["layer"]==7 and cell["passed"] and cell["correctness"]["O_capture"]["byte_equal"]
            raw={m:(period/f"cell{c}-{m}.f32").read_bytes() for m in gearnames}
            assert raw["O"]==raw["C"]==(root/"reference"/f"cell{c}-down.f32").read_bytes() and raw["S"]==raw["P"]
            assert all(len(v)==5120*6*4 for v in raw.values())
            ids=cell["ids"]
            assert [i for i,e in enumerate(ids) if e in (19,154,third)]==cell["gpu_slots"]
            for slot in range(6):
                if slot not in cell["gpu_slots"]:
                    a,z=slot*5120*4,(slot+1)*5120*4;assert raw["O"][a:z]==raw["P"][a:z]
            samples=cell["samples"]; first={s["mode"]:ms(s["output_ready_ns"]-s["input_ready_ns"]) for s in samples if s["first_use"]}
            data=[s for s in samples if not s["first_use"]];assert len(data)==48 and len(first)==4
            blocks=[];dur=collections.defaultdict(list);parts=collections.defaultdict(lambda:collections.defaultdict(list))
            for b in range(12):
                seq=sorted((v for v in data if v["block"]==b),key=lambda v:v["order"])
                assert "".join(v["mode"] for v in seq)==("OCPS","CSOP","SPCO","POSC")[b%4]
                times={}
                for s in seq:
                    m=s["mode"];dt=ms(s["output_ready_ns"]-s["input_ready_ns"]);assert dt>0 and s["executed_gpu_H"]==(c if m in "SP" else 0)
                    times[m]=dt;dur[m].append(dt)
                    parts[m]["cpu_graph_ms"].append(ms(s["cpu_end_ns"]-s["cpu_begin_ns"]))
                    parts[m]["join_ms"].append(ms(s["join_end_ns"]-s["join_begin_ns"]))
                    assert s["gpu_operations"]==(5 if s["executed_gpu_H"] else 0)
                effects={"C_minus_O_ms":times["C"]-times["O"],"C_minus_S_ms":times["C"]-times["S"],
                         "S_minus_P_ms":times["S"]-times["P"],"O_minus_S_ms":times["O"]-times["S"],"O_minus_P_ms":times["O"]-times["P"]}
                blocks.append(effects)
                for k,v in effects.items():overall[f"{third}-H{c}"][k].append(v)
            ref=f32(d/f"cell{c}-f64ref.f32")
            cells.append({"cell":c,"hits":c,"ids":ids,"gpu_slots":cell["gpu_slots"],"first_ms":first,
                 "mode_median_ms":{k:median(v) for k,v in dur.items()},
                 "paired_median_ms":{k:median([b[k] for b in blocks]) for k in blocks[0]},
                 "parts_median":{m:{k:median(v) for k,v in l.items()} for m,l in parts.items()},
                 "cpu_vs_gpu_weighted_nmse":cell["correctness"]["P_O_weighted"]["nmse"],
                 "reference_errors":{m:error(f32(period/f"cell{c}-{m}.f32"),ref) for m in ("O","P")}})
        all_periods.append({"period":idx,"third_id":third,"cells":cells})
    routes=read(root/"reference/old_training_L7.json")
    routing={}
    for third in (331,85):
        ids={19,154,third};hists={};states=collections.Counter();f=0
        for name,v in routes["cases"].items():
            hist=collections.Counter(len(set(x)&ids) for x in v["ids"]);states.update(hist)
            hists[name]=dict(hist);f+=sum(third in x for x in v["ids"])
        routing[str(third)]={"historical_H":dict(states),"historical_cases":hists,"historical_third_id_occurrences":f}
    return {"periods":all_periods,"paired_by_membership_and_hits":{
             name:{k:{"median_ms":median(v),"positive":sum(x>0 for x in v),"blocks":len(v)} for k,v in fields.items()}
             for name,fields in overall.items()},"old_training_routing":routing,
            "memory_MiB":{x["phase"]:x["cuda"]["free_bytes"]/2**20 for x in r["memory_phases"]},
            "timed_local_samples":4*3*12*4,"first_use_samples":4*3*4,
            "notes":"Positive C−S is the benefit of moving work in series; positive S−P is the extra benefit of parallel submission. O−P includes all private overhead. H0/H1/H2 are different fixed inputs. Historical training routes are not an online hit-rate measurement."}


if __name__=="__main__":
    root=Path(sys.argv[1] if len(sys.argv)>1 else ".")
    out=analyze(root)
    if "--check" in sys.argv:assert json.loads(json.dumps(out))==read(root/"OPERATOR.json")
    else:(root/"OPERATOR.json").write_text(json.dumps(out,indent=2)+"\n")
    for key,v in out["paired_by_membership_and_hits"].items():print(key, {k:round(x["median_ms"],4) for k,x in v.items()})
