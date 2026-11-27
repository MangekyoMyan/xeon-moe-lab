"""Validate full-state restore receipts, independent functional probes and matched outputs."""
import argparse
import json
from analyze import R, read, lines, sha, matches, elapsed, grade, score, FULL_MASK

PERIODS = ("SA1","SB1","SB2","SA2")
CASES = ("warehouse","departments","python")


def assemble(root):
    f = read(root/"FIXTURES.json")
    files = read(root/"shared/IMAGES.json")
    targets = read(root/"TARGET_TOKENS.json")["expected_spellings"]
    result, arrays, continuous, direct_probes = {}, {}, {}, {}
    for period in PERIODS:
        run = root/"runs"/period
        request = {r["label"]:r for r in lines(run/"requests.jsonl")}
        actions = lines(run/"slot.jsonl")
        trace = lines(run/"trace/calls.jsonl")
        assert read(run/"STOP.json")["child_rc"]==0
        checked = read(run/"CHECKED.json")
        assert checked["requests_sha256"]==sha(run/"requests.jsonl") and checked["actions_sha256"]==sha(run/"slot.jsonl")
        setting = read(run/"READY.json")["thresholds"][0]
        assert len(request)==(13 if period=="SA1" else 7) and len(actions)==(12 if period=="SA1" else 9)
        result[period],arrays[period] = {},{}
        for name in CASES:
            case,im = f["cases"][name],files[name]
            n = len(case["history_ids"])
            assert im["cached_tokens"]==n
            def prompt(stage,ids,offset,device=None):
                cs = [c for c in trace if c["phase"]==name+"-"+stage]
                assert cs and all(c["rc"]==0 for c in cs)
                end = offset+len(ids)
                captured = [(i,k) for c in cs for i,k in zip(c["pos"],c["input_ids"]) if i<end]
                assert captured==list(zip(range(offset,end),ids))
                pc = [c for c in cs if c["pos"][-1]<end]
                valid = [c for c in pc if c["outputs"]]
                assert len(valid)==1 and valid[0]["pos"][-1]==end-1
                if device:
                    assert len(pc)==len(pc[0]["ubatches"])==1
                    u=pc[0]["ubatches"][0]
                    assert u["tokens"]==len(ids) and u[device]==[FULL_MASK]*3
                    assert u["CPU" if device=="CUDA0" else "CUDA0"]==[0]*3
                return cs,valid[0],run/"trace"/valid[0]["logits_file"]
            if period=="SA1":
                _,_,boundary=prompt("seed",case["history_ids"],0)
                _,_,direct=prompt("directprobe",case["new_ids"][:8],n,"CPU")
                origin,=[a for a in actions if a["verb"]=="save" and a["response"]["filename"]==im["file"]]
                assert im["sha256"]==origin["sha256"] and im["bytes"]==origin["bytes"]
                continuous[name] = {"prefix_logit_sha256":sha(boundary),"direct_probe_sha256":sha(direct)}
                direct_probes[name] = direct
            restores = [a for a in actions if a["verb"]=="restore" and a["response"]["filename"]==im["file"]]
            assert len(restores)==2 and all(a["sha256"]==im["sha256"] and a["response"]["n_restored"]==n for a in restores)
            cp, = [a for a in actions if a["verb"]=="save" and a["response"]["filename"]==period+"-"+name+"-roundtrip.slot"]
            _,obs,pr = prompt("probe",case["new_ids"][:8],n,"CPU")
            rows,v,file = prompt("follow",case["new_ids"],n,"CPU" if setting==128 else "CUDA0")
            lab=name+"-follow"
            req=request[lab]
            assert restores[0]["done_ns"] < obs["begin_ns"] < restores[1]["begin_ns"] < restores[1]["done_ns"] < req["start_ns"]
            t=req["timings"]
            assert t["cache_n"]==n and t["prompt_n"]==len(case["new_ids"]) and req["usage"]["prompt_tokens_details"]["cached_tokens"]==n
            output=[c for c in rows if c["outputs"]]
            natural=[c["input_ids"][0] for c in rows if c["pos"][0]>=n+len(case["new_ids"])]
            assert len(output)==t["predicted_n"]==len(natural)+1
            arrays[period][name]=[(run/"trace"/c["logits_file"]) for c in output]
            events=read(run/(lab+"-SSE.json"))
            pieces=[(e["ns"],c["delta"]["content"]) for e in events if "event" in e for c in e["event"]["choices"] if c["delta"].get("content")]
            assert pieces[0][0]==req["first_visible_ns"] and "".join(s for _,s in pieces)==req["content"] and events[-1]["done"]
            result[period][name]={"canonical_state_sha256":im["sha256"],"roundtrip_sha256":cp["sha256"],
                "roundtrip_bytes":cp["bytes"],"roundtrip_byte_identical":cp["sha256"]==im["sha256"],
                "restored_probe_sha256":sha(pr),"8_token_probe_matches_never_restored":sha(pr)==continuous[name]["direct_probe_sha256"],
                "direct_vs_restored_probe":score(direct_probes[name],pr),
                "request_sha256":req["request_sha256"],"actual_prompt_ids_positions_verified":True,
                "cached":n,"new":len(case["new_ids"]),"host_37_projections_device":"CPU" if setting==128 else "CUDA0",
                "prompt_s":t["prompt_ms"]/1000,"logits_s":elapsed(rows[0]["begin_ns"],v["logits_ready_ns"]),
                "visible_s":elapsed(req["start_ns"],req["first_visible_ns"]),"done_s":elapsed(req["start_ns"],req["done_ns"]),
                "content":req["content"],"correct":grade(req["content"],case["expected"]),"greedy_inputs_after_prompt":natural,
                "suffix_raw_sha256":sha(file),"restores_ms":[a["response"]["timings"]["restore_ms"] for a in restores]}
    comparison={}
    for a,b in (("SA1","SB1"),("SA2","SB2"),("SA1","SA2"),("SB1","SB2")):
        grp=comparison[a+"-"+b]={}
        for name in CASES:
            left,right=result[a][name],result[b][name]
            assert left["request_sha256"]==right["request_sha256"]
            ga,gb=left["greedy_inputs_after_prompt"],right["greedy_inputs_after_prompt"]
            stats=[]
            for i,(x,y) in enumerate(zip(arrays[a][name],arrays[b][name])):
                if i and ga[i-1]!=gb[i-1]:
                    break
                target={spelling:ids[i] for spelling,ids in targets[name+"-follow"].items() if i<len(ids) and ids[:i]==ga[:i]==gb[:i]}
                stats.append(score(x,y,target))
            grp[name]={"all_matched_outputs":stats,"visible_delta_right_minus_left_s":right["visible_s"]-left["visible_s"]}
    probe_pairs = {}
    for name in CASES:
        probe_pairs[name] = {}
        for a,b in (("SA1","SB1"),("SA2","SB2"),("SA1","SA2"),("SB1","SB2")):
            paths=[]
            for per in (a,b):
                base=root/"runs"/per/"trace"
                call=next(c for c in lines(base/"calls.jsonl") if c["phase"]==name+"-probe" and c["outputs"])
                paths.append(base/call["logits_file"])
            probe_pairs[name][a+"-"+b]=score(*paths)
    return {"canonical":files,"producer_never_restored":continuous,"periods":result,
            "restored_probes_across_periods":probe_pairs,"pairs":comparison}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    actual=assemble(R)
    output=R/"shared/RESULT.json"
    if args.check:
        assert matches(read(output),actual)
    else:
        output.write_text(json.dumps(actual,ensure_ascii=False,indent=2)+"\n")
    for key,group in actual["periods"].items():
        print(key,[(name,round(v["visible_s"],3),v["roundtrip_byte_identical"],v["8_token_probe_matches_never_restored"]) for name,v in group.items()])
