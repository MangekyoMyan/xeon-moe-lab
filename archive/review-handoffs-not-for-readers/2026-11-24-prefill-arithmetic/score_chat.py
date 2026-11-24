"""Score the eight fixed free-generation tasks. Exercise only known AST nodes and in-memory SQL."""
import ast
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys


def read(p):
    return json.loads(p.read_bytes())


def plain(text):
    s=text.strip()
    block=re.fullmatch(r"\x60\x60\x60(?:[a-zA-Z0-9]+)?\s*\n(.+?)\n\x60\x60\x60",s,re.S)
    return (block[1].strip(),False) if block else (s,True)


def evaluate(node, env):
    if isinstance(node, ast.Expression):
        return evaluate(node.body,env)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate(node.operand,env)
    if isinstance(node, ast.Compare) and len(node.ops)==len(node.comparators)==1:
        a,b=evaluate(node.left,env),evaluate(node.comparators[0],env)
        ops={ast.GtE:lambda:a>=b,ast.Gt:lambda:a>b,ast.Lt:lambda:a<b,ast.LtE:lambda:a<=b}
        return ops[type(node.ops[0])]()
    if isinstance(node, ast.ListComp) and len(node.generators)==1:
        g=node.generators[0]
        if not isinstance(g.target,ast.Name) or g.is_async:
            raise ValueError("unknown comprehension")
        ys=[]
        for x in evaluate(g.iter,env):
            inner={**env,g.target.id:x}
            if all(evaluate(test,inner) for test in g.ifs):
                ys.append(evaluate(node.elt,inner))
        return ys
    if isinstance(node, ast.Subscript) and isinstance(node.slice,ast.Slice):
        s=node.slice
        args=[evaluate(v,env) if v is not None else None for v in (s.lower,s.upper,s.step)]
        return evaluate(node.value,env)[slice(*args)]
    raise ValueError("unsupported syntax: "+type(node).__name__)


def python_check(body):
    expr=ast.parse(body,mode="eval")
    n=0
    for k in range(6):
        for xs in itertools.product((-2,0,1,5),repeat=k):
            source=list(xs)
            assert evaluate(expr,{"xs":source})==[v for v in xs if v>=0][-2:] and source==list(xs)
            n+=1
    return {"input_lists":n}


def sql_check(body):
    samples=[[],[("C",2)],[("b",2),("a",1),("a",1),("d",2),("k",4)],
             [("C",-2),("A",-1),("B",-5),("B",4)],[("Z",1),("X",4),("Y",-4),("Y",8)]]
    assert re.match(r"(?is)^\s*(select|with)\b",body)
    for rows in samples:
        with sqlite3.connect(":memory:") as db:
            db.execute("create table sales(shop text,amount integer)")
            db.executemany("insert into sales values(?,?)",rows)
            db.commit()
            db.execute("pragma query_only=on")
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,16384)
            db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH,16384)
            def permit(action, a, b, name, trigger):
                ok=action==sqlite3.SQLITE_SELECT or (action==sqlite3.SQLITE_READ and a=="sales") or (action==sqlite3.SQLITE_FUNCTION and b and b.lower()=="sum")
                return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY
            db.set_authorizer(permit)
            db.set_progress_handler(lambda:1,100000)
            sums={}
            for k,v in rows:sums[k]=sums.get(k,0)+v
            expect=sorted(sums.items(),key=lambda x:(-x[1],x[0]))[:2]
            assert db.execute(body).fetchmany(4)==expect
    return {"sqlite_tables":len(samples)}


def score(case,row):
    body,unfenced=plain(row["text"])
    detail={}
    ok=False
    try:
        expected=case["expect"]
        if case["id"]=="py-slice":
            detail=python_check(body)
            ok=True
        elif case["id"]=="sql-sales":
            detail=sql_check(body)
            ok=True
        elif case["id"]=="jp-mail":
            detail={"recipient":"佐藤" in body,"new_due":"火曜" in body and bool(re.search(r"17(?:時|:00)|午後[5５]時",body)),
                    "shared_folder":"共有フォルダ" in body,"polite":bool(re.search(r"ます|さい|ました|です",body)),
                    "sentence_count":len([s for s in re.split("[。！？]",body.strip()) if s.strip()]),
                    "text_for_review":body}
            ok=all(detail[k] for k in ("recipient","new_due","shared_folder","polite")) and 2<=detail["sentence_count"]<=3
        elif isinstance(expected,dict):
            pairs=json.loads(body,object_pairs_hook=lambda p:p)
            if not isinstance(pairs,list) or len(dict(pairs))!=len(pairs):raise ValueError("invalid object")
            actual=dict(pairs)
            ok=actual==expected and all(type(actual.get(k))==type(v) for k,v in expected.items())
        else:
            ok=body==expected
    except (AssertionError,ValueError,KeyError,SyntaxError,TypeError,sqlite3.Error) as exc:
        detail["error"]=type(exc).__name__+": "+str(exc)
    return {"passes":ok and unfenced and row["finish"]=="stop","content_ok":ok,
            "format_without_fence":unfenced,"detail":detail}


def summarize(root):
    design=read(root/"CHAT_DESIGN.json");fixtures=read(root/"CHAT_FIXTURES.json")
    results={}; pids=set(); interval=read(root/"BUILD_INTERVAL.json")
    for i,(label,threshold) in enumerate(design["periods"]):
        base=root/"runs"/("chat-"+label)
        ready=read(base/"READY.json");stop=read(base/"STOP.json")
        assert ready["thresholds_verified_from_environ"]==threshold and ready["pid"] not in pids
        pids.add(ready["pid"])
        assert stop["supervisor_return"]==stop["child_return"]==0 and stop["guard"]=="stopped_by_request"
        records=[json.loads(s) for s in (base/"results.jsonl").read_text().splitlines()]
        order=list(design["prompts"])
        if i%2:order.reverse()
        assert [r["case"] for r in records]==["warmup",*order]
        for row in records:
            cap=read(root/"raw-chat"/label/row["case"]/"summary.json")
            a,b=(cap["perf_counter_ns"][key] for key in ("request_start","eof"))
            assert b<interval["start_ns"] or a>interval["finish_ns"],"build overlaps HTTP request"
            assert row["cache_n"]==0 and cap["status"]=="complete" and cap["http"]["status"]==200
        per={}
        for case in fixtures:
            name=case["id"]; row=next(r for r in records if r["case"]==name)
            assert row["prompt"]==design["prompts"][name]["prompt_tokens"]
            per[name]={**row,"score":score(case,row)}
        results[label]=per
    summaries={};band=[]
    for case in fixtures:
        name=case["id"]
        by={k:[results[t][name] for t,_ in design["periods"] if t.startswith(k)] for k in ("A","B")}
        vals={k:statistics.mean(v["ttft_ms"] for v in vv) for k,vv in by.items()}
        assert all(math.isfinite(x) and x>0 for x in vals.values())
        passed={k:[r["score"]["passes"] for r in vv] for k,vv in by.items()}
        count=by["A"][0]["prompt"]
        same=len({v["text"] for r in by.values() for v in r})==1
        stable={k:vv[0]["text"]==vv[1]["text"] for k,vv in by.items()}
        robust=max(v["ttft_ms"] for v in by["B"])<min(v["ttft_ms"] for v in by["A"])
        summaries[name]={"tokens":count,"mean_first_text_ms":vals,"B_over_A":vals["B"]/vals["A"],
            "128_faster_than_both_64":robust,"pass_by_repeat":passed,"text_same_all_four":same,"text_repeat_stable":stable,
            "example_64":by["A"][0]["text"],"example_128":by["B"][0]["text"]}
        if 64<=count<128:band.append(name)
    assert len(band)==6
    return {"periods":design["periods"],"requests_scored":32,"distinct_prompts":8,"eligible":band,"per_case":summaries,
            "mean_first_text_ms_affected":{k:statistics.mean(summaries[n]["mean_first_text_ms"][k] for n in band) for k in ("A","B")},
            "passed_by_period":{k:sum(v["score"]["passes"] for v in vv.values()) for k,vv in results.items()},
            "raw_scores":results}


if __name__=="__main__":
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    result=summarize(root);dst=root/"CHAT_RESULT.json"
    if "--check" in sys.argv:assert result==read(dst)
    else:dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for k,s in result["per_case"].items():
        print(k,s["tokens"],{l:round(v) for l,v in s["mean_first_text_ms"].items()},s["pass_by_repeat"],s["text_same_all_four"])
