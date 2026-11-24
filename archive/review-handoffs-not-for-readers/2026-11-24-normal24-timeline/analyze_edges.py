import json
from pathlib import Path
import sys
from analyze_whole import score


def read(p):return json.loads(p.read_bytes())


def analyze(root):
    decision=read(root/"prefix_quality/DECISION.json")
    if not decision["run"]:return {"ran":False,"scope":decision}
    cand=decision["candidate"]
    dirs=[root/"runs"/f"qpf-{x}" for x in (64,cand)]
    runs=[read(d/"RUN.json") for d in dirs]
    for j in runs:
        assert j["passed"] and j["compact_bytes_after_unload"]==0 and j["context"]["threads"]==j["context"]["threads_batch"]==24
    cases=read(root/"prefix_quality/FIXTURES.json")["cases"];out={}
    for case,b,s in zip(cases,runs[0]["blocks"],runs[1]["blocks"],strict=True):
        name=case["id"];assert b["id"]==s["id"]==name and b["rows"]==s["rows"]==len(case["targets"])
        assert b["prefix_ids"]==s["prefix_ids"]==case["prefix"] and b["forced_inputs"]==s["forced_inputs"]==case["forced"]
        quality=score(dirs[0]/(name+".f32"),dirs[1]/(name+".f32"),case["targets"])
        out[name]={"length":len(case["prefix"]),"quality":quality,"prefix_ms_64":(b["prefill_ready_ns"]-b["prefill_begin_ns"])/1e6,
            "prefix_ms_candidate":(s["prefill_ready_ns"]-s["prefill_begin_ns"])/1e6}
    return {"ran":True,"normal":64,"candidate":cand,"target_length":decision["target_length"],"cases":out,
            "total_rows_per_arm":sum(v["quality"]["shape"][0] for v in out.values()),
            "scope":"Full fixed GGUF logits at 7 prefix lengths per family; the affected native input length also has 64 subsequent forced steps. Timings of these extra run prefixes are descriptive (one process per policy)."}


if __name__=="__main__":
    root=Path(sys.argv[1]);v=analyze(root);p=root/"PREFIX_QUALITY.json"
    if "--check" in sys.argv:assert v==read(p)
    else:p.write_text(json.dumps(v,indent=2)+"\n")
    if v["ran"]:
        print(v["candidate"], {k:(q["quality"]["exact_rows"],q["quality"]["argmax_changes"],round(q["prefix_ms_candidate"]/q["prefix_ms_64"],2)) for k,q in v["cases"].items()})
