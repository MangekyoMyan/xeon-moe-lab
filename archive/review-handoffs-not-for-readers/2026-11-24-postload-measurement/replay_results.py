"""Offline replay from the frozen inputs, model outputs and internal call clocks."""
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

DRIVER="b32c3777afd478e0a726f898795c3bcbe14005c69fd4f116083c87520f646994"
CASES="0458b7a358d0eb3a0c289eb1a1aa9903ee0bb972158acc64abbf0952f472d746"
GOLD={"text_cache_budget_ja":"35d028fed9fa08bbe8f304c598ced362ff466aff176ecf407133c0a1870bbd64",
      "code_stable_grouping_python":"077eefd33928bfbec340162c5b5e54c2bbad9e4ac3ae72bc23f38a7a33d69d01"}
VOCAB=129280
GiB=2**30


def read(p):
    return json.loads(p.read_bytes())


def stats(ns):
    assert len(ns)>0 and all(isinstance(v,int) and v>0 for v in ns)
    ms=[t/1e6 for t in ns]
    return {"calls":len(ns),"T1_s":len(ns)*1e9/sum(ns),"mean_ms":statistics.fmean(ms),
            "median_ms":statistics.median(ms),"p95_ms":sorted(ms)[math.ceil(.95*len(ms))-1],"max_ms":max(ms)}


def stats_for_condition(folder, slug, baseline_gold, cases):
    per_case={};runs=[];pids=set();new_quality={}
    for stage in ("quality-01","timing-01","timing-02"):
        d=folder/(slug+"-"+stage)
        j=read(d/"RUN.json");s=read(d/"SPEC.json");l=read(d/"LAUNCH.json")
        c=s["condition"]; pid=j["linux_entry"]["pid"]
        assert j["passed"] and j["spec"]==s and l["spec"]==s and l["driver_sha256"]==DRIVER and l["cases_sha256"]==CASES
        assert j["context"]["threads"]==c["decode_threads"] and j["context"]["threads_batch"]==c["prefill_threads"]
        assert c["projector_threads"]==j["projector_init_threads_argument"]==12 and len(c["process_cpu_ids"])==c["decode_threads"]
        assert j["linux_entry"]["effective_cpu_ids"]==c["process_cpu_ids"] and j["linux_entry"]["mempolicy_mode"]==3
        assert j["linux_entry"]["thp_disabled"] and j["linux_entry"]["mempolicy_nodes_mask"]==3
        assert pid not in pids and j["compact_bytes"]==127401984000 and j["compact_bytes_after_unload"]==j["after_unload_KiB"]["VmLck"]==0
        pids.add(pid)
        q=stage.startswith("quality")
        if q:
            reg=j["registration"]
            assert (reg["registered_tensors"],reg["selected_projections"])==(80,19200)
        assert [b["case_id"] for b in j["blocks"]]==[cases[ix]["case_id"] for ix in s["cases"]]
        metrics={}
        for ix,b in zip(s["cases"],j["blocks"]):
            case=cases[ix]; name=b["case_id"]
            computed=b["start_ns"];ends=b["output_ready_ns"]
            assert len(computed)==len(ends)==64 and computed[0]>b["prefill_ready_ns"]>b["prefill_begin_ns"]
            assert all(a<b2 for a,b2 in zip(ends,computed[1:]))
            times=[y-x for x,y in zip(computed,ends)]
            assert min(times)>0
            filename=b["file"]; assert filename==name+".f32"
            content=(d/filename).read_bytes()
            if q:
                assert b["rows"]==65 and len(content)==VOCAB*65*4 and hashlib.sha256(content).hexdigest()==GOLD[name]
                assert content==baseline_gold[name]
                new_quality[name]=content[-4*VOCAB:]
            else:
                assert b["rows"]==1 and len(content)==VOCAB*4 and content==new_quality[name]
                row=per_case.setdefault(name,{"groups":[],"prefill":[]})
                row["groups"].append(times)
                row["prefill"].append((b["prefill_ready_ns"]-b["prefill_begin_ns"])/1e9)
                metrics[name]={**stats(times),"first_T1_ms":times[0]/1e6,"prefill_s":row["prefill"][-1]}
        before,after=j["io_before_load"],j["io_after_load"]
        runs.append({"name":s["name"],"pid":pid,"condition":c["name"],"quality":q,"actual_threads":j["context"]["threads"],
                     "actual_batch_threads":j["context"]["threads_batch"],"text_context_load_s":j["text_model_context_load_ns"]/1e9,
                     "projector_load_s":j["vision_init_ns"]/1e9,
                     "warmup_s":(j["preparation"]["warmup_end_ns"]-j["preparation"]["warmup_start_ns"])/1e9,
                     "unload_s":j["unload_ns"]/1e9,"load_read_bytes":after["read_bytes"]-before["read_bytes"],
                     "load_major_faults":after["major_faults"]-before["major_faults"],"per_case":metrics})
    assert len(pids)==3
    summary={}
    for name,groups in per_case.items():
        assert len(groups["groups"])==2
        summary[name]={**stats(groups["groups"][0]+groups["groups"][1]),
                       "prefill_mean_s":statistics.fmean(groups["prefill"]),
                       "per_process_T1_s":[stats(ts)["T1_s"] for ts in groups["groups"]]}
    return {"cases":summary,"runs":runs,"quality_rows_byte_equal_to_old":len(new_quality)*65,
            "saved_timing_tails_equal_to_same_condition":4,"calls_main_timing":sum(s["calls"] for s in summary.values())}


def analyze(root, old):
    cases_file=root/"trial/CASES.json"
    assert hashlib.sha256(cases_file.read_bytes()).hexdigest()==CASES
    cases=read(cases_file)["cases"]
    olddir=old/"trial/runs"
    gold={}
    for n,h in GOLD.items():
        b=(olddir/"old-p12-quality-01"/(n+".f32")).read_bytes()
        assert hashlib.sha256(b).hexdigest()==h
        # All other reported full outputs are also directly byte-compared to these rows.
        values=array.array("f")
        values.frombytes(b)
        if sys.byteorder!="little":
            values.byteswap()
        assert values.itemsize==4 and len(values)==65*VOCAB and all(map(math.isfinite,values))
        gold[n]=b
    result={}
    for name,folder,slug in (("OLD_P12",olddir,"old-p12"),("NEW_P12",root/"trial/runs","new-p12"),
                             ("NEW_D24_B12",root/"trial/runs","new-d24-b12")):
        result[name]=stats_for_condition(folder,slug,gold,cases)
        assert all(r["condition"]==name for r in result[name]["runs"])
    ratios={}
    for c1,c2 in (("OLD_P12","NEW_P12"),("NEW_P12","NEW_D24_B12"),("OLD_P12","NEW_D24_B12")):
        ratios[c2+"_over_"+c1]={name:result[c2]["cases"][name]["T1_s"]/result[c1]["cases"][name]["T1_s"] for name in GOLD}
    new_ids=[r["pid"] for c in ("NEW_P12","NEW_D24_B12") for r in result[c]["runs"]]
    assert len(set(new_ids))==6
    return {"conditions":result,"throughput_ratios":ratios,"prospective_plan":"Each condition has 1 quality + 2 fresh timings in reversed task order. 64 fixed serial decode calls per case and timing process.",
            "new_primary_calls":512,"new_quality_rows_compared":260,"new_saved_timing_tail_rows_compared":8,
            "quality_scope":"All saved F32 values checked. Timing middle rows are not saved. Direct vectors cover two fixed cases."}


if __name__=="__main__":
    root=Path(sys.argv[1]); old=Path(sys.argv[2])
    result=analyze(root,old);dest=root/"FIXED_COMPARISON.json"
    if "--check" in sys.argv:
        assert result==read(dest)
        print("Replayed 512 new timing calls, 260 new quality rows and 8 timing tails against the saved references")
    else:
        dest.write_text(json.dumps(result,indent=2)+"\n")
