"""Replay the four native API trials. Python standard library only."""
import argparse
import array
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

R = Path(__file__).resolve().parent
N_VOCAB = 129280
FULL_MASK = (1 << 37) - 1
FOLLOW = ["warehouse-follow", "departments-follow", "python-follow"]
CHATS = FOLLOW + ["departments-edit"]


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elapsed(start, end):
    return (end - start) / 1e9


def inputs(fixtures, label):
    control = fixtures["controls"]
    if label == "longcontrol-history":
        return control["warm_control_history"]
    if label == "longcontrol-follow":
        return control["warm_control_history"] + control["warm_control_suffix"]
    if label in ("warmup", "cold96"):
        return control[label]
    name, which = label.split("-")
    case = fixtures["cases"][name]
    suffix = {"history": [], "follow": case["new_ids"], "edit": fixtures["edit"]["new_ids"]}[which]
    return case["history_ids"] + suffix


def grade(content, target):
    if isinstance(target, list):
        try:
            value = json.loads(content)
        except ValueError:
            return False
        return isinstance(value, list) and all(type(v) is int for v in value) and value == target
    return content.strip() == target


def verify_answers(f):
    cases = f["cases"]
    warehouse = cases["warehouse"]["messages"]
    warehouse_key, = re.findall(r"R-[0-9]+", warehouse[-1]["content"])
    value, = re.findall(rf"^{warehouse_key} \| (Q[0-9]+) \|", warehouse[0]["content"], re.M)
    assert value == cases["warehouse"]["expected"]
    jprows = {}
    for lab,case in (("departments",cases["departments"]),("departments-edit",f["edit"])):
        key, = re.findall(r"北[0-9]{3}", case["messages"][-1]["content"])
        (sent,returned), = re.findall(rf"^{key}　([0-9]+)台　([0-9]+)台$",case["messages"][0]["content"],re.M)
        assert str(int(sent)-int(returned)) == case["expected"]
        jprows[lab] = [key,int(sent),int(returned)]
    p = cases["python"]["messages"]
    name, = re.findall(r"totals\(ROWS, '(p[0-9]+)'\)", p[-1]["content"])
    code = p[0]["content"].split("```python\n",1)[1].split("\n```",1)[0]
    fn, assignment = ast.parse(code).body
    assert isinstance(fn,ast.FunctionDef) and fn.name == "totals"
    assert isinstance(assignment,ast.Assign) and assignment.targets[0].id == "ROWS"
    items = [[sent,returned] for key,sent,returned in ast.literal_eval(assignment.value) if key==name]
    assert [len(items),sum(x-y for x,y in items)] == cases["python"]["expected"]
    original, edited = inputs(f,"departments-follow"), inputs(f,"departments-edit")
    assert len(original) == len(edited)
    diffs = [i for i,(a,b) in enumerate(zip(original,edited)) if a!=b]
    return {"warehouse_key_and_value":[warehouse_key,value],
            "departments_loaned_returned":jprows, "python_target":name,"python_sent_returned":items,
            "edit_input_tokens":len(edited),"edit_changed_positions_zero_based":diffs,"edit_common_prefix":diffs[0]}


def requests(root, fixtures, period):
    p = root / "runs" / period
    out = {}
    for v in lines(p / "requests.jsonl"):
        lab = v["label"]
        tm = v["timings"]
        n = len(inputs(fixtures, lab))
        assert n == v["input_tokens_expected"] == tm["prompt_n"] + tm["cache_n"]
        assert v["start_ns"] <= v["headers_ns"] <= v["done_ns"] <= v["eof_ns"]
        row = {"request_sha256": v["request_sha256"], "cached": tm["cache_n"], "new": tm["prompt_n"],
               "generated": tm["predicted_n"], "prompt_s": tm["prompt_ms"]/1000,
               "server_decode_s": tm["predicted_ms"]/1000, "content": v["content"], "finish": v["finish"],
               "done_s": elapsed(v["start_ns"], v["done_ns"])}
        if lab in CHATS:
            sse = read(p / (lab + "-SSE.json"))
            assert sse[-1] == {"ns": v["done_ns"], "done": True}
            text = [(e["ns"], choice["delta"]["content"]) for e in sse if "event" in e
                    for choice in e["event"]["choices"] if choice["delta"].get("content")]
            assert v["first_visible_ns"] == text[0][0] and "".join(x for _, x in text) == v["content"]
            assert v["usage"]["prompt_tokens_details"]["cached_tokens"] == tm["cache_n"]
            target = fixtures["edit"]["expected"] if lab.endswith("-edit") else fixtures["cases"][lab.split("-")[0]]["expected"]
            row.update({"expected": target, "correct": grade(v["content"], target),
                        "visible_s": elapsed(v["start_ns"], text[0][0]), "text_events": len(text),
                        "content_event_intervals_ns": [b[0]-a[0] for a, b in zip(text, text[1:])]})
        assert lab not in out
        out[lab] = row
    assert len(out) == 11 and read(p/"CHECKED.json")["requests_sha256"] == sha(p/"requests.jsonl")
    return out


def trace(root, fixtures, period, rows):
    base = root / "runs" / period / "trace"
    calls = lines(base / "calls.jsonl")
    assert [c["seq"] for c in calls] == list(range(1, len(calls) + 1))
    grouped = defaultdict(list)
    for call in calls:
        grouped[call["phase"]].append(call)
    assert set(grouped) == set(rows)
    result, arrays = {}, {}
    for label, record in rows.items():
        sequence = inputs(fixtures, label)
        cs = grouped[label]
        full = len(sequence)
        submitted = []
        expected = list(zip(range(record["cached"], full), sequence[record["cached"]:]))
        new_outputs, step_outputs = [], []
        assert all(c["rc"] == 0 for c in cs)
        for c in cs:
            assert c["pos"] and len(c["pos"]) == len(c["input_ids"])
            assert sorted(c["pos"]) == c["pos"]
            for pos, tok in zip(c["pos"], c["input_ids"]):
                if pos < full:
                    submitted.append((pos, tok))
            if c["outputs"] != 0:
                assert c["outputs"] == 1 and (base / c["logits_file"]).stat().st_size == N_VOCAB*4
                assert c["begin_ns"] <= c["submit_return_ns"] <= c["logits_ready_ns"]
                (new_outputs if c["pos"][-1] < full else step_outputs).append(c)
        assert len(new_outputs) == 1 and new_outputs[0]["pos"][-1] == full-1
        assert submitted == expected, (period,label,len(submitted),len(expected),submitted[:1],expected[:1])
        prompt_calls = [c for c in cs if c["pos"][0] < full]
        decode_calls = [c for c in cs if c["pos"][0] >= full]
        assert all(len(c["pos"]) == 1 and c["pos"][0] == full+i for i,c in enumerate(decode_calls))
        assert len(decode_calls) == record["generated"]-1
        stage = []
        for c in prompt_calls:
            for u in c["ubatches"]:
                masks = (u["CPU"], u["CUDA0"])
                assert all(not (a & b) and (a | b) == FULL_MASK for a,b in zip(*masks)), (label,u)
                if masks[0] == [FULL_MASK]*3:
                    backend = "CPU"
                elif masks[1] == [FULL_MASK]*3:
                    backend = "CUDA0"
                else:
                    backend = "mixed"
                stage.append([u["tokens"], backend])
        assert sum(n for n,_ in stage) == record["new"], (period,label,stage,record["new"])
        last = new_outputs[0]
        result[label] = {
            "actual_ids_positions_match_requested_after_cache": True,
            "cached": record["cached"], "new": len(submitted), "first_position": submitted[0][0],
            "last_position": submitted[-1][0], "prompt_ubatches_37_host_layers": stage,
            "decode_ubatches": [u["tokens"] for c in decode_calls for u in c["ubatches"]],
            "first_prompt_call_to_logits_s": elapsed(prompt_calls[0]["begin_ns"], last["logits_ready_ns"]),
            "terminal_native_row_sha256": sha(base / last["logits_file"]),
            "greedy_inputs_after_prompt": [c["input_ids"][0] for c in decode_calls],
        }
        files = new_outputs + step_outputs
        arrays[label] = [(base / c["logits_file"], c["seq"]) for c in files]
    return result, arrays


def probabilities(path):
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == N_VOCAB and all(math.isfinite(v) for v in values)
    top = max(range(len(values)), key=values.__getitem__)
    m = values[top]
    exp = [math.exp(v-m) for v in values]
    z = math.fsum(exp)
    return values, top, [v/z for v in exp]


def score(a, b, expected=None):
    xa, at, ap = probabilities(a)
    same = a.read_bytes() == b.read_bytes()
    xb, bt, bp = (xa, at, ap) if same else probabilities(b)
    result = {"exact": same, "top1_A": at, "top1_B": bt, "p_A_top1": ap[at], "p_B_top1": bp[bt],
              "max_abs_logit": 0.0 if same else max(abs(x-y) for x,y in zip(xa,xb)),
              "TV": 0.0 if same else math.fsum(abs(x-y) for x,y in zip(ap,bp))/2,
              "first_choice_same": at == bt}
    for spelling,target in (expected or {}).items():
        info = {"token_id":target,"p_A":ap[target],"p_B":bp[target]}
        for arm,values in (("A",xa),("B",xb)):
            value = values[target]
            info["rank_"+arm] = 1 + sum(v > value or (v == value and j < target) for j,v in enumerate(values))
        result.setdefault("expected_next_with_matching_spelling",{})[spelling] = info
    return result


def assemble(root):
    fixtures = read(root/"FIXTURES.json")
    design = read(root/"DESIGN.json")
    assert sha(root/"FIXTURES.json") == design["fixture_sha256"]
    records, traces, arrays = {}, {}, {}
    for period, threshold, observed in design["periods"]:
        run = root/"runs"/period
        assert read(run/"READY.json")["thresholds"] == [threshold]*2
        assert read(run/"READY.json")["observer"] == observed
        assert read(run/"STOP.json")["child_rc"] == 0
        records[period] = requests(root,fixtures,period)
        if observed:
            traces[period], arrays[period] = trace(root,fixtures,period,records[period])
    comparisons = {}
    for a,b,name in [("A1","B1","plain"),("A2","B2","diagnostic")]:
        comparisons[name] = {}
        for lab,x in records[a].items():
            y = records[b][lab]
            assert x["request_sha256"] == y["request_sha256"]
            metrics = {}
            for clock in ["prompt_s","done_s"]+(["visible_s"] if lab in CHATS else []):
                metrics[clock] = {"64": x[clock], "128":y[clock], "saved_s":x[clock]-y[clock],
                                  "128_over_64":y[clock]/x[clock]}
            comparisons[name][lab] = {"cached_new_equal": (x["cached"],x["new"]) == (y["cached"],y["new"]),
                                      "output_text_equal":x["content"] == y["content"], **metrics}
    quality = {}
    labels = read(root/"TARGET_TOKENS.json")["expected_spellings"]
    for label in records["A2"]:
        aa,bb = arrays["A2"][label],arrays["B2"][label]
        ga = traces["A2"][label]["greedy_inputs_after_prompt"]
        gb = traces["B2"][label]["greedy_inputs_after_prompt"]
        original = {}
        if label not in CHATS:
            for period in ("A2","B2"):
                original[period] = next(r["tokens"][0] for r in lines(root/"runs"/period/"requests.jsonl") if r["label"] == label)
        matched = []
        for i,((pa,sa),(pb,sb)) in enumerate(zip(aa,bb)):
            if i and ga[i-1] != gb[i-1]:
                break
            expected = {spelling:ids[i] for spelling,ids in labels.get(label,{}).items()
                        if i < len(ids) and ga[:i] == ids[:i] and gb[:i] == ids[:i]}
            z = score(pa,pb,expected)
            target_a = ga[i] if i < len(ga) else original.get("A2")
            target_b = gb[i] if i < len(gb) else original.get("B2")
            z.update({"row":i,"actual_next_input_A":target_a,"actual_next_input_B":target_b,
                      "raw_argmax_matches_sample_A":None if target_a is None else target_a==z["top1_A"],
                      "raw_argmax_matches_sample_B":None if target_b is None else target_b==z["top1_B"],
                      "seq_A2_B2":[sa,sb]})
            matched.append(z)
        assert matched
        quality[label] = matched
    return {"answers_recomputed_from_question_data":verify_answers(fixtures),
            "periods":records,"pairs":comparisons,"dispatch":traces,"matched_logits":quality}


def matches(saved,actual,field=""):
    if type(saved) is not type(actual):
        return False
    if isinstance(saved,dict):
        return saved.keys()==actual.keys() and all(matches(saved[k],actual[k],k) for k in saved)
    if isinstance(saved,list):
        return len(saved)==len(actual) and all(matches(a,b,field) for a,b in zip(saved,actual))
    if isinstance(saved,float) and field in {"TV","p_A_top1","p_B_top1","p_A","p_B"}:
        return math.isclose(saved,actual,rel_tol=1e-13,abs_tol=1e-15)
    return saved==actual


def verify_manifest(root):
    manifest = root/"MANIFEST.json"
    if manifest.is_file():
        groups = read(manifest)
        for entry in ("record","full_vocab_little_endian_float32","full_slot_images"):
            for name,spec in groups[entry].items():
                path = (root/name).resolve()
                assert path.is_relative_to(root.resolve()) and path.stat().st_size==spec["bytes"] and sha(path)==spec["sha256"], name


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--check", action="store_true")
    opts = arg.parse_args()
    if opts.check:
        verify_manifest(R)
    result = assemble(R)
    output = R/"RESULT.json"
    if opts.check:
        assert matches(read(output),result), "replay differs from saved RESULT"
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    for case in CHATS:
        print(case,[(name,round(stats[case]["visible_s"]["64"],3),round(stats[case]["visible_s"]["128"],3)) for name,stats in result["pairs"].items()])
