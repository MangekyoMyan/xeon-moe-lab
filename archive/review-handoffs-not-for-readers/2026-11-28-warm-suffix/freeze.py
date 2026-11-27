"""Freeze ordinary template/tokenizer output for three synthetic multi-turn cases."""
import json
import urllib.request
from build import R,LAB,read,save,sha
from pathlib import Path

cfg=read(LAB/"deepseek41-bringup-20261101/server-config.json")
assert sha(LAB/"deepseek41-bringup-20261101/server-config.json")==read(LAB/"current/runtime.json")["service_config"]["sha256"]
key=Path(cfg["api_key_file"]).read_text().strip()
base="http://"+cfg["host"]+":"+str(cfg["port"]);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(route,body):
    req=urllib.request.Request(base+route,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    with opener.open(req,timeout=30) as f:return json.load(f)
def render(messages,gen):
    s=post("/apply-template",{"messages":messages,"add_generation_prompt":gen,"chat_template_kwargs":{"thinking_mode":"chat"}})["prompt"]
    return post("/tokenize",{"content":s,"add_special":True,"parse_special":True,"with_pieces":False})["tokens"]
def opening(s):return [{"role":"user","content":s},{"role":"assistant","content":"資料を確認しました。"}]
def freeze(history,question,expect):
    msgs=history+[{"role":"user","content":question}];old=render(history,False);full=render(msgs,True)
    server=post("/v1/chat/completions/input_tokens",{"model":cfg["model_alias"],"messages":msgs,"chat_template_kwargs":{"thinking_mode":"chat"}})
    assert server["input_tokens"]==len(full),(server,len(full))
    assert full[:len(old)]==old,"history is not an exact prefix"
    return {"messages":msgs,"history_ids":old,"new_ids":full[len(old):],"expected":expect}
def make_jp(n):
    title="支所の備品貸出表です。貸出数と返却数を使って未返却の台数を調べられるよう、次の質問までこの表を覚えておいてください。\n支所番号　貸出数　返却数"
    return opening(title+"\n"+"\n".join(f"北{i:03d}　{80+i*11%113}台　{7+i*13%59}台" for i in range(n)))
def jp_question(i):
    return (f"先ほどの備品貸出表で、支所番号「北{i:03d}」が借りた備品は、返却済みの分を差し引くとまだ何台残っていますか。"
            "該当する一つの支所について、貸出数から返却数を引いてください。答えは半角の整数だけで、単位・式・説明は付けないでください。")
def make_py(n):
    data=[(f"p{i:03d}",5+i*7%31,i*3%5) for i in range(n)]
    data.extend((f"p{i:03d}",3+i*5%17,i%3) for i in range(0,n,4))
    text=("Please read this Python fixture. I will ask about a single call in the next message.\n\n"
          "```python\ndef totals(rows, target):\n    items = [(sent, returned) for name, sent, returned in rows if name == target]\n"
          "    return [len(items), sum(sent - returned for sent, returned in items)]\n\nROWS = [\n"
          +"".join(f'    ("{k}", {x}, {y}),\n' for k,x,y in data)+"]\n```")
    return opening(text)
py_question=("Using the Python code and ROWS from my earlier message, what exact value would totals(ROWS, 'p036') return? "
             "Check every occurrence of that key, including the later entries; the second element is the sum of sent minus returned over those entries. "
             "Return just the two integers in a JSON array in the function's original order. No explanation or code fences.")
def choose(make,target):
    lo,hi=45,850
    while hi-lo>1:
        mid=(lo+hi)//2
        if len(render(make(mid),False))<target:lo=mid
        else:hi=mid
    h=make(hi);n=len(render(h,False));assert target<=n<=target+63,(n,target)
    return h,hi
def main():
    out=R/"FIXTURES.json";assert not out.exists()
    old=read(LAB/"prefill-split-20261127/large/CASES.json")["conversation"]
    c=freeze(old["messages"][:2],old["messages"][2]["content"],old["target"])
    assert c["history_ids"]==old["common_prefix_tokens"] and c["new_ids"]==old["suffix_tokens"] and len(c["new_ids"])==75
    jp,n=choose(make_jp,3072);j=freeze(jp,jp_question(37),str(80+37*11%113-(7+37*13%59)))
    edit=freeze(jp,jp_question(38),str(80+38*11%113-(7+38*13%59)))
    py,npy=choose(make_py,4096);i=36
    p=freeze(py,py_question,[2,(5+i*7%31-i*3%5)+(3+i*5%17-i%3)])
    assert all(64<=len(v["new_ids"])<128 for v in (j,p,edit)),[(k,len(v["new_ids"])) for k,v in (("jp",j),("py",p))]
    earlier=read(LAB/"parallel-pilot-20261117/replay/CASES.json")["cases"][1]["expected_token_ids"]
    controls={"cold96":read(LAB/"normal24-timeline-20261124/prefill/TOKENS.json")["JA96"],
              "warm_control_history":[0]+earlier[:1023],"warm_control_suffix":earlier[1023:1183],
              "warmup":[0]+earlier[1300:1340]}
    save(out,{"template":"deployed /apply-template + /tokenize, verified against chat input_tokens","cases":{"warehouse":c,"departments":j,"python":p},
              "edit":{"case":"departments","messages":edit["messages"],"new_ids":edit["new_ids"],"expected":edit["expected"]},
              "jp_rows":n,"python_base_rows":npy,"controls":controls})
    print({k:[len(v["history_ids"]),len(v["new_ids"]),v["expected"]] for k,v in read(out)["cases"].items()})
if __name__=="__main__":main()
