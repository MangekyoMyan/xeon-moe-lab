"""Recompute native GPU precision and paired timings from the two independent graph-key runs."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
ARMS=("base","split","paired")
MODES=("all","MMVQ","MMQ","FP16")
def read(p):return json.loads(p.read_bytes())
def table(p):
    with p.open() as s:return list(csv.DictReader(s,delimiter="\t"))
def rms(x):return math.sqrt(statistics.fmean(v*v for v in x))
def numeric(rows,field,reference="full"):
    err=[float(v[field])-float(v[reference]) for v in rows]
    return {"RMSE":rms(err),"mean_abs":statistics.fmean(map(abs,err)),"max_abs":max(map(abs,err))}


def summarize_one(folder):
    receipt=read(folder/"RUN.json");guard=read(folder/"GUARD.json");ds=table(folder/"dots.tsv")
    groups=table(folder/"groups.tsv");schedule=table(folder/"timings.tsv");steady=table(folder/"steady.tsv")
    assert receipt["passed"] and receipt["distinct_graph_keys_checked"] and receipt["all_three_arms_repeated_bitwise"]
    assert guard["passed"] and guard["probe_exit"]==0 and guard["initial_metrics"]==guard["final_metrics"]
    assert receipt["sample_count"]==len(ds)==4608 and receipt["selected_jobs"]==len(groups)==114
    assert len(schedule)==receipt["timed_intervals"]==114*18 and len(steady)==receipt["steady_intervals"]==114*15
    assert sum(int(g["compared"]) for g in groups)==receipt["full_outputs_compared_to_prior"]
    assert sum(int(g["base_bit_mismatches"]) for g in groups)==receipt["baseline_bit_mismatches"]==0
    coords=("fixture","projection","token","slot","expert","row")
    bypos={tuple(v[k] for k in coords):v for v in ds}
    assert len(bypos)==4608 and all(float(v["base"])==float(v["archived"]) for v in ds)
    cases={int(g["job"]):g for g in groups}
    assert set(cases)==set(range(len(groups)))
    for v in ds:
        g=cases[int(v["job"])]
        assert all(g[k]==v[k] for k in ("fixture","projection","expert","inputs","mode","paired_mode"))
    timing={}
    for label,raw,repeat in (("interleaved",schedule,6),("consecutive",steady,5)):
        batches=defaultdict(dict);orders=defaultdict(dict)
        for row in raw:
            j,arm,rep,ns=map(int,(row["job"],row["arm"],row["repetition"],row["ns"]))
            assert 0<=arm<3 and 0<=rep<repeat and ns>0 and rep not in batches[j,arm]
            batches[j,arm][rep]=ns
            if label=="interleaved":
                i=int(row["order"]);assert i not in orders[j,rep];orders[j,rep][i]=arm
        assert len(batches)==114*3 and all(len(q)==repeat for q in batches.values())
        if label=="interleaved":
            assert all(len({tuple(orders[j,rep][i] for i in range(3)) for rep in range(6)})==6 for j in cases)
        modes={}
        for mode in MODES:
            chosen=[j for j,g in cases.items() if mode=="all" or g["mode"]==mode]
            times=[sum(statistics.median(batches[j,arm].values()) for j in chosen)/1e6 for arm in range(3)]
            modes[mode]={"jobs":len(chosen),"sum_of_job_medians_ms":dict(zip(ARMS,times)),
                         "ratios":{k:v/times[0] for k,v in zip(ARMS,times)}}
        timing[label]=modes
    shape={};numeric_by_case={}
    for mode in MODES:
        rows=[v for v in ds if mode=="all" or v["mode"]==mode]
        shape[mode]={"sampled_dots":len(rows),"GGUF_reference_RMS":rms(float(v["full"]) for v in rows),
                     "errors":{k:numeric(rows,k) for k in (*ARMS,"paired_first")},
                     "closer_than_ordinary":{k:sum(abs(float(v[k])-float(v["full"]))<abs(float(v["base"])-float(v["full"])) for v in rows) for k in ARMS[1:]}}
        if mode=="FP16":
            shape[mode]["ideal_if_only_activations_corrected"]=numeric(rows,"weight_half_ideal")
            shape[mode]["split_against_that_ideal"]=numeric(rows,"split","weight_half_ideal")
    for case in ("JA96","PY96"):
        rows=[v for v in ds if v["fixture"]==case]
        numeric_by_case[case]={a:numeric(rows,a)["RMSE"] for a in ARMS}
    transitions=Counter((v["mode"],v["paired_mode"]) for v in ds)
    cross_first=sum(int(g["paired_first_bit_mismatches"]) for g in groups)
    return {"order":receipt["visit_order"],"outputs_reproduced":receipt["full_outputs_compared_to_prior"],
            "numerics":shape,"by_prompt_RMSE":numeric_by_case,"timing":timing,
            "mode_transitions_by_sample":{"/".join(k):v for k,v in sorted(transitions.items())},
            "paired_first_different_bits_full":cross_first,"min_free_MiB_sampled":receipt["minimum_cuda_free_bytes_sampled"]/2**20,
            "real_requests_while_timing":0,"instantaneous_guard_samples":len(guard["samples"]),
            "api_pid":guard["server_pid"]},bypos


def summarize():
    trials=[];positions=[]
    for order in ("forward","reverse"):
        folder=ROOT/read(ROOT/("MEASURED-"+order+".json"))["run"]
        s,p=summarize_one(folder)
        assert s["order"]==order
        trials.append(s);positions.append(p)
    assert set(positions[0])==set(positions[1])
    for k in positions[0]:
        assert {n:v for n,v in positions[0][k].items() if n!="job"}=={n:v for n,v in positions[1][k].items() if n!="job"}
    assert trials[0]["api_pid"]==trials[1]["api_pid"]
    return {"distinct_dots":len(positions[0]),"trials":trials,"cross_process_all_dot_columns_match":True,
            "scope":"Real GGUF weights and activations, selected L0 experts only. Timings sum medians for the 114 selected resident projection groups. No routing, host-to-device weight transfer, long-context, quality or end-to-end engine timing in this probe."}


if __name__=="__main__":
    result=summarize();out=ROOT/"RESULTS.json"
    if "--check" in sys.argv:assert result==read(out)
    else:out.write_text(json.dumps(result,indent=2)+"\n")
    for s in result["trials"]:
        print(s["order"],s["outputs_reproduced"],"baseline values exact")
        for mode in MODES:
            m=s["numerics"][mode]
            print(mode,m["sampled_dots"],[f"{m['errors'][a]['RMSE']:.5g}" for a in ARMS],
                  {k:[round(v["ratios"][a],3) for a in ARMS] for k,v in ((name,s["timing"][name][mode]) for name in ("interleaved","consecutive"))})
