import collections
import json
import math
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_bytes())


def analyze(root):
    sets=read(root/"prefill/TOKENS.json");policies=read(root/"prefill/DESIGN.json")["policies"]
    results={};all_rows={};pids=set()
    for name,t in policies:
        d=root/"runs"/("pp-"+name); ready=read(d/"READY.json");stop=read(d/"STOP.json")
        assert stop["supervisor_exit"]==0 and stop["guard_status"]=="stopped_by_request"
        spec=read(d/"SPEC.json")
        assert ready["pid"] not in pids and ready["backing"] and spec["threshold"]==t
        assert spec["public_environment_delta"]=={k:str(t) for k in ("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")}
        pids.add(ready["pid"])
        r=[json.loads(s) for s in (d/"requests.jsonl").read_text().splitlines()]
        assert len(r)==15 and r[0]["rep"]=="warmup" and r[0]["server"]["cache_n"]==0 and r[0]["server"]["prompt_n"]==64
        assert [v["fixture"] for v in r[1:]]==list(sets)+list(reversed(sets))
        grouped=collections.defaultdict(list)
        for v in r[1:]:
            key=v["fixture"];n=len(sets[key]);s=v["server"]
            assert v["input_len"]==s["prompt_n"]==n and s["cache_n"]==0 and s["predicted_n"]==len(v["output_ids"])==4 and not v["cache_prompt"]
            assert v["output_ids"]==[i for c in v["chunks"] for i in c["tokens"]]
            assert v["first_nonempty_content_ns"]==next((c["ns"] for c in v["chunks"] if c["text"]),None)
            assert v["first_token_ns"]==next(c["ns"] for c in v["chunks"] if c["tokens"])
            ttft=(v["first_nonempty_content_ns"]-v["start_ns"])/1e6 if v["first_nonempty_content_ns"] is not None else None
            grouped[key].append({"rep":v["rep"],"prompt_ms":s["prompt_ms"],"first_text_ms":ttft,
                "first_token_ms":(v["first_token_ns"]-v["start_ns"])/1e6,"output_ids":v["output_ids"],"text":v["text"]})
        all_rows[name]=grouped
        results[name]={"threshold":t,"pid":ready["pid"],"cases":{key:{"prompt_mean_ms":statistics.mean(v["prompt_ms"] for v in items),
            "text_mean_ms":statistics.mean(v["first_text_ms"] for v in items) if all(v["first_text_ms"] is not None for v in items) else None,
            "repeat_outputs_equal":items[0]["output_ids"]==items[1]["output_ids"],"repetitions":items} for key,items in grouped.items()}}
    comparisons={}
    for n,t in (("L",32),("H",128)):
        g={}
        for key in sets:
            metrics={}
            for field in ("first_text_ms","first_token_ms","prompt_ms"):
                base=[v[field] for a in ("A1","A2") for v in all_rows[a][key]]
                cand=[v[field] for v in all_rows[n][key]]
                if any(x is None for x in base+cand):
                    metrics[field]=None;continue
                ratio=statistics.mean(cand)/statistics.mean(base)
                metrics[field]={"mean_candidate":statistics.mean(cand),"mean_baseline":statistics.mean(base),
                    "ratio":ratio,"candidate_below_both_controls_in_each_order":all(cand[i]<min(all_rows[a][key][i][field] for a in ("A1","A2")) for i in (0,1)),
                    "candidate_above_both_controls_in_each_order":all(cand[i]>max(all_rows[a][key][i][field] for a in ("A1","A2")) for i in (0,1))}
            baseline=all_rows["A1"][key][0]["output_ids"]
            g[key]={"length":len(sets[key]),"metrics":metrics,
                    "baseline_ids_stable":all(v["output_ids"]==baseline for a in ("A1","A2") for v in all_rows[a][key]),
                    "all_four_ids_match_64":all(v["output_ids"]==baseline for v in all_rows[n][key])}
        comparisons[str(t)]=g
    return {"timed_requests":56,"warmup_requests":4,"configurations":results,"compared_to_bracketing_64":comparisons,
            "scope":"Streamed native /completion, first nonempty text vs first token ID vs server prompt_ms. 2 order-reversed repeats per process. 32/128 one fresh process each, 64 two. No prefix cache (server cache_n=0). Four generated tokens per request, not a long-form quality check."}


if __name__=="__main__":
    root=Path(sys.argv[1] if len(sys.argv)>1 else ".");data=analyze(root)
    if "--check" in sys.argv:assert data==read(root/"PREFILL.json")
    else:(root/"PREFILL.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    for t,cases in data["compared_to_bracketing_64"].items():
        print(t,[(k,round(v["metrics"]["prompt_ms"]["ratio"],3),v["all_four_ids_match_64"]) for k,v in cases.items()])
