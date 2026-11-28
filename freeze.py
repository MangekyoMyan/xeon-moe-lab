"""Use the deployed template/tokenizer only. No inference calls are made here."""
import hashlib
import http.client
import json
from pathlib import Path

R=Path(__file__).resolve().parent
LAB=R.parent
def load(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
def follow(messages,role,text):return messages+[{"role":role,"content":text}]
cfg_path=LAB/"deepseek41-bringup-20261101/server-config.json"
cfg=load(cfg_path)
assert hashlib.sha256(cfg_path.read_bytes()).hexdigest()==load(R/"BEFORE.json")["config_sha256"]
key=Path(cfg["api_key_file"]).read_text().strip()
def post(route,data):
    con=http.client.HTTPConnection(cfg["host"],cfg["port"],timeout=30)
    try:
        con.request("POST",route,json.dumps(data).encode(),{"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        res=con.getresponse();assert res.status==200,(route,res.status)
        return json.loads(res.read())
    finally:con.close()
def render(messages):
    txt=post("/apply-template",{"messages":messages,"add_generation_prompt":True,"chat_template_kwargs":{"thinking_mode":"chat"}})["prompt"]
    return post("/tokenize",{"content":txt,"add_special":True,"parse_special":True})["tokens"]
def entry(messages,prev,max_tokens=1):
    ids=render(messages)
    assert ids[:len(prev)]==prev
    reply=post("/v1/chat/completions/input_tokens",{"model":cfg["model_alias"],"messages":messages,"chat_template_kwargs":{"thinking_mode":"chat"}})
    assert reply["input_tokens"]==len(ids)
    return {"messages":messages,"ids":ids,"max_tokens":max_tokens}
def first(anchor,name,target):
    base=anchor["messages"];n=len(anchor["tokens"])
    def content(k):
        return (f"作業票 {name}。登録番号は731です。後の質問に使います。\n"+
                "\n".join(f"仮伝票 {i:03d}　保管箱 {(i*23+7)%193:03d}　受領済" for i in range(30))+
                "\n区切り欄"+(" x"*k)+"\n今は OK とだけ返してください。")
    lo,hi=0,target+60
    while lo<hi:
        m=(lo+hi)//2
        if len(render(follow(base,"user",content(m))))<n+target:lo=m+1
        else:hi=m
    x=entry(follow(base,"user",content(lo)),anchor["tokens"])
    assert len(x["ids"])-n==target,(name,len(x["ids"])-n,target)
    return x
if __name__=="__main__":
    out=R/"fixtures/workload.json";assert not out.exists()
    anchors={n:load(R/"fixtures"/(n+"-anchor.json")) for n in ("short","long")}
    jobs={}
    for name,where,new in [("linear","short",1056),("practice1","long",1024),("practice2","long",1040),
                           ("practice3","long",1056),("late","long",1056),("early","long",768)]:
        b=first(anchors[where],name,new);jobs[name]={"root":where,"B":b}
        if name.startswith("practice"):continue
        base=follow(b["messages"],"assistant","OK")
        if name in ("linear","early"):
            question="先ほどの作業票の登録番号を半角の数字だけで答えてください。"
        else:
            question="新しい受付番号は731です。この新しい受付番号だけを半角の数字で答えてください。"
        c=entry(follow(base,"user",question),b["ids"],4);jobs[name]["C"]=c
        if name=="linear":continue
        if name=="late":edit=follow(base,"user",question.replace("731","284"))
        else:
            edit=[dict(m) for m in c["messages"]]
            edit[len(anchors[where]["messages"])]["content"]=edit[len(anchors[where]["messages"])]["content"].replace("番号は731","番号は284")
        e=entry(edit,anchors[where]["tokens"],4);jobs[name]["edit"]=e
        assert (e["ids"][:len(b["ids"])]==b["ids"])==(name=="late")
        assert e["ids"]!=c["ids"]
    save(out,jobs)
    print({k:{s:len(t["ids"]) for s,t in v.items() if s!="root"} for k,v in jobs.items()})
