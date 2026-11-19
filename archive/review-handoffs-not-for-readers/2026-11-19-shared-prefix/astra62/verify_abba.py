"""Recompute the prespecified A/B screen directly from integer wall-clock endpoints."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

here=Path(__file__).resolve().parent
study=here.parent
t=study/"trial"
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
specs=[s for s in read(t/"DESIGN.json")["stages"] if s["stage"]=="timing"]
assert [(s["arm"],s["cases"]) for s in specs]==[("A",[0,1]),("B",[0,1]),("B",[1,0]),("A",[1,0])]
ref=read(study/"main/runs/baseline/RUN.json")
frozen={b["case_id"]:b for b in ref["blocks"]}
arm_sums={c:[] for c in frozen}
sources={}
for spec in specs:
    d=t/"runs"/spec["name"]
    s=read(d/"RUN.json");cfg=read(d/"LAUNCH.json");health=read(d/"health.json")
    assert s["passed"] and s["spec"]==spec and "LD_PRELOAD" not in cfg["environment"]
    assert health["status"]=="exited" and health["child_returncode"]==0
    assert [b["case_index"] for b in s["blocks"]]==spec["cases"]
    assert s["vocab_size"]==129280 and s["context"]==ref["context"]
    for b in s["blocks"]:
        name=b["case_id"];base=frozen[name]
        for k in ("prefix_ids","input_tokens","positions_begin","case_index","split"):
            assert b[k]==base[k]
        start,end=b["start_ns"],b["output_ready_ns"]
        assert len(start)==len(end)==64 and b["status"]==[0]*64 and b["traced_calls"]==0
        assert all(0<a<z for a,z in zip(start,end)) and all(end[i-1]<=start[i] for i in range(1,64))
        assert b["rows"]==1 and b["all_finite"]
        output=(d/b["file"]).read_bytes()
        assert len(output)==129280*4 and output==(study/"main/runs/baseline"/base["file"]).read_bytes()[-len(output):]
        a=b["prefix_stats_before_T1"];z=b["prefix_stats_after_T1"]
        if spec["arm"]=="B":
            assert z[2]-a[2]==z[3]-a[3]==2368 and z[4]==0 and z[5]==27648 and z[6]==37
        else:
            assert a==z==[0]*7
        arm_sums[name].append(sum(end)-sum(start))
    sources[spec["name"]]=sha(d/"RUN.json")
cases={}
for name,(a0,b1,b2,a3) in arm_sums.items():
    pair=(Fraction(b1,a0),Fraction(b2,a3));whole=Fraction(b1+b2,a0+a3)
    cases[name]=dict(process_sums_ns=arm_sums[name],B_A_paired_time_ratio=[float(v) for v in pair],
       B_A_case_ratio=float(whole),time_saved_percent=100*float(1-whole),
       mean_saved_ms=float(Fraction(a0+a3-b1-b2,128*1_000_000)),
       baseline_bracket_last_minus_first_ms=float(Fraction(a3-a0,64*1_000_000)),
       passes=all(v<1 for v in pair) and whole<=Fraction(995,1000))
result=dict(method="Per-process per-case exact integer sums of all 64 output-ready minus start timestamps. Ratios are exact fractions for the decision.",
    runs_sha256=sources,predeclared_metric_sha256=sha(t/"PERFORMANCE_METRIC.json"),
    source_sha256=sha(Path(__file__)),calls=512,case_results=cases,passes=all(d["passes"] for d in cases.values()))
with (here/"ABBA_INDEPENDENT.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
print(json.dumps(result,indent=2))
