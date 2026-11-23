"""Saved-data comparisons for the fixed 12-physical-core baseline."""
import array
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TRIAL=ROOT/"trial"
OLD=ROOT.parent/"ecc9-simplification-20261119"
VOCAB=129280
ROWS=65

def read(p):return json.loads(p.read_text())
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,j):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(j,indent=2,allow_nan=False)+"\n")
    temp.replace(p)
def floats(raw):
    v=array.array("f")
    v.frombytes(raw)
    if sys.byteorder!="little":v.byteswap()
    return v
def compare_quality(run):
    results={}
    gold_refs=read(OLD/"GOLDEN_OUTPUTS.json")
    for b in read(run/"RUN.json")["blocks"]:
        name=b["file"]
        old=OLD/"trial/runs/quality-control"/name
        new=run/name
        assert sha(old)==gold_refs[name]["full"]
        assert old.stat().st_size==new.stat().st_size==ROWS*VOCAB*4
        rows=[]
        with old.open("rb") as f, new.open("rb") as g:
            for i in range(ROWS):
                a,binary=f.read(VOCAB*4),g.read(VOCAB*4)
                v=floats(a);w=floats(binary)
                assert all(map(math.isfinite,v)) and all(map(math.isfinite,w))
                denominator=math.fsum(x*x for x in v)
                equal=(a==binary)
                if equal:err=absolute=0.0
                else:
                    err=math.fsum((x-y)**2 for x,y in zip(v,w,strict=True))
                    absolute=max(abs(x-y) for x,y in zip(v,w,strict=True))
                expected_top=v.index(max(v)); seen_top=expected_top if equal else w.index(max(w))
                rows.append({"row":i,"byte_equal":equal,"max_abs":absolute,"squared_error":err,
                             "reference_energy":denominator,"nmse":err/denominator if denominator else None,"historical_argmax":expected_top,"p12_argmax":seen_top})
        denominator=math.fsum(row["reference_energy"] for row in rows)
        error=math.fsum(row["squared_error"] for row in rows)
        results[name]={"historical_sha256":sha(old),"old_p12_sha256":sha(new),"shape":[ROWS,VOCAB],
            "byte_equal":all(r["byte_equal"] for r in rows),"argmax_changes":sum(r["historical_argmax"]!=r["p12_argmax"] for r in rows),
            "max_abs_error":max(r["max_abs"] for r in rows),"nmse":error/denominator if denominator else None,"finite":True,"per_row":rows}
    return {"historical":"633e quality-control, 12 decode / 24 batch","comparison":"OLD_P12 quality, 12/12","cases":results}

def metrics(values):
    n=len(values)
    return {"N":n,"mean_ms":sum(values)/n/1e6,"median_ms":statistics.median(values)/1e6,
            "p95_ms_nearest_rank":sorted(values)[math.ceil(n*.95)-1]/1e6,"max_ms":max(values)/1e6,
            "T1_per_second":n*1e9/sum(values)}

def aggregate(trial):
    stages=read(trial/"DESIGN.json")["stages"]
    groups={}
    per_process=[]
    for s in stages:
        if s["stage"]!="timing":continue
        j=read(trial/"runs"/s["name"]/"RUN.json")
        assert j["passed"] and j["spec"]==s
        items={}
        for block in j["blocks"]:
            ns=[b-a for a,b in zip(block["start_ns"],block["output_ready_ns"],strict=True)]
            assert len(ns)==64 and min(ns)>0 and block["status"]==[0]*64
            name=block["case_id"]
            items[name]={**metrics(ns),"durations_ns":ns,"prefill_s":(block["prefill_ready_ns"]-block["prefill_begin_ns"])/1e9}
            groups.setdefault(name,[]).append(ns)
        per_process.append({"name":s["name"],"case_order":s["cases"],"pid":j["linux_entry"]["pid"],"cases":items,
                            "model_context_load_s":j["text_model_context_load_ns"]/1e9,"vision_init_s":j["vision_init_ns"]/1e9,
                            "warmup_s":(j["preparation"]["warmup_end_ns"]-j["preparation"]["warmup_start_ns"])/1e9,
                            "unload_s":j["unload_ns"]/1e9,"text_RSS_GiB":j["loaded_text_memory_KiB"]["VmRSS"]/1024**2,
                            "registry_bytes_after_unload":j["compact_bytes_after_unload"],"locked_KiB_after_unload":j["after_unload_KiB"]["VmLck"],
                            "io_during_text_load":{k:j["io_after_load"][k]-v for k,v in j["io_before_load"].items() if k!="monotonic_ns" and k in j["io_after_load"]}})
    result={}
    for n,groups_for_case in groups.items():
        flat=[x for g in groups_for_case for x in g]
        assert len(flat)==128
        means=[metrics(v)["mean_ms"] for v in groups_for_case]
        result[n]={**metrics(flat),"process_mean_range_ms":[min(means),max(means)],
                   "prefill_mean_s":statistics.fmean(p["cases"][n]["prefill_s"] for p in per_process)}
    assert len(per_process)==len({p["pid"] for p in per_process})==2
    return {"condition":"OLD_P12","timing_processes":len(per_process),"timed_T1_calls":256,"cases":result,"processes":per_process}
