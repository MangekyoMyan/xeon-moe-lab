"""Compute fixed-work process comparisons from the saved output-ready timestamps."""
import argparse
import json
from pathlib import Path
from statistics import fmean

def read(path):
    return json.loads(path.read_text())

def contrast(values):
    a0,b0,b1,a1=values
    a=fmean([a0,a1])
    b=fmean([b0,b1])
    return {"process_order":"ABBA","process_values":values,"control_mean":a,"ordinary_mean":b,
            "change_percent":(b/a-1)*100,"pair_changes_percent":[(b0/a0-1)*100,(b1/a1-1)*100],
            "control_last_minus_first":a1-a0,"ordinary_pair_difference":b1-b0,
            "both_ordinary_above_both_control":min(b0,b1)>max(a0,a1),
            "both_ordinary_below_both_control":max(b0,b1)<min(a0,a1)}

def calculate(root):
    trial=root/"trial"
    design=read(trial/"DESIGN.json")
    stages=[s for s in design["stages"] if s["stage"]=="timing"]
    assert [s["arm"] for s in stages]==["control","ordinary","ordinary","control"]
    all_runs=[]
    for stage in stages:
        folder=trial/"runs"/stage["name"]
        run=read(folder/"RUN.json")
        assert stage==run["spec"] and run["passed"]
        assert read(folder/"LAUNCH.json")["environment"].get("LD_PRELOAD") is None
        by_case={}
        for b,i in zip(run["blocks"],stage["cases"],strict=True):
            assert b["case_index"]==i and b["status"]==[0]*64
            ns=[end-start for start,end in zip(b["start_ns"],b["output_ready_ns"],strict=True)]
            assert len(ns)==64 and min(ns)>0
            by_case[b["case_id"]]={"durations_ns":ns,"mean_ms":sum(ns)/64e6,
                                   "prefill_ms":(b["prefill_ready_ns"]-b["prefill_begin_ns"])/1e6}
        all_runs.append({"name":stage["name"],"arm":stage["arm"],"cases":by_case,
                         "load_seconds":run["text_model_context_load_ns"]/1e9,"vision_seconds":run["vision_init_ns"]/1e9,
                         "unload_seconds":run["unload_ns"]/1e9,
                         "text_rss_GiB":run["loaded_text_memory_KiB"]["VmRSS"]/1024**2,
                         "peak_rss_GiB":run["before_unload_KiB"]["VmHWM"]/1024**2,
                         "unloaded_rss_GiB":run["after_unload_KiB"]["VmRSS"]/1024**2,
                         "unloaded_locked_KiB":run["after_unload_KiB"]["VmLck"]})
    cases={n:{m:contrast([r["cases"][n][m] for r in all_runs]) for m in ["mean_ms","prefill_ms"]}
           for n in all_runs[0]["cases"]}
    phases={m:contrast([r[m] for r in all_runs]) for m in ["load_seconds","vision_seconds","unload_seconds","text_rss_GiB","peak_rss_GiB","unloaded_rss_GiB"]}
    qualities={}
    for arm in ["control","ordinary"]:
        j=read(trial/"runs"/("quality-"+arm)/"RUN.json")
        qualities[arm]={"compact_bytes":j["compact_bytes"],"after_unload":j["compact_bytes_after_unload"],
                        "rows":[b["rows"] for b in j["blocks"]],"registration":{k:v for k,v in j["registration"].items() if k!="experts"}}
    return {"fresh_timing_processes":4,"timed_T1_calls":512,"cases":cases,"load_memory":phases,
            "quality":qualities,"runs":all_runs,"decision_rule":design["decision"]}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    data=calculate(args.root)
    with args.output.open("x") as f:
        json.dump(data,f,indent=2)
        f.write("\n")
    for k,v in data["cases"].items():print(k,json.dumps(v))
    print("Model/context load",json.dumps(data["load_memory"]["load_seconds"]))
