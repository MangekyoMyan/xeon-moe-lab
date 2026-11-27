"""Render benchmark messages with the ordinary server; only template/tokenize endpoints are used."""
import json
from pathlib import Path
import urllib.request
from build_capture import HERE,LAB,read,save,sha

cfg=read(LAB/"deepseek41-bringup-20261101/server-config.json");secret=Path(cfg["api_key_file"]).read_text().strip()
base="http://"+cfg["host"]+":"+str(cfg["port"]);op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path,body):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+secret})
    with op.open(req,timeout=30) as f:return json.load(f)
def render(messages,gen):
    s=post("/apply-template",{"messages":messages,"add_generation_prompt":gen,"chat_template_kwargs":{"thinking_mode":"chat"}})["prompt"]
    return post("/tokenize",{"content":s,"add_special":True,"parse_special":True,"with_pieces":False})["tokens"]
long=read(LAB/"parallel-pilot-20261117/replay/CASES.json")["cases"][1]["expected_token_ids"]
ja=[0]+long
old=read(HERE/"capture/PY512.json");msg=old["request"]["messages"][0]["content"]
tail=(LAB/"glu-split-20261125/analyze.py").read_text().split('if __name__=="__main__":',1)[1]
code=msg.rsplit("```",1)[0]+'\nif __name__=="__main__":'+tail+"\n```\nThen review the input validation and the choice of the statistical unit."
py=render([{"role":"user","content":code}],True)
assert len(py)>=1040 and py[:512]==old["tokens"] and ja[:512]==read(HERE/"capture/FIXTURE.json")["tokens"]
ack="一覧を確認しました。";row_id=317
def expected(i):return f"Q{(i*173+5137)%10000:04d}"
def history(count):
    lines=["倉庫の照合表です。次の質問に使うので、まず一覧を確認してください。","受付番号 | 確認コード | 個数"]
    for i in range(count):lines.append(f"R-{i:04d} | {expected(i)} | {1+(i*37)%91}")
    return [{"role":"user","content":"\n".join(lines)},{"role":"assistant","content":ack}]
lo,hi=520,1900
while hi-lo>1:
    mid=(lo+hi)//2
    if len(render(history(mid),False))<16384:lo=mid
    else:hi=mid
hist=history(hi);q="先ほどの倉庫の照合表のうち、受付番号 R-0317 に対応する確認コードを、表のとおりに答えてください。個数や受付番号そのものではなく、Qで始まる確認コードの文字列を一つだけ返してください。説明や引用符も不要です。"
prior=render(hist,False);full=render(hist+[{"role":"user","content":q}],True);common=0
for x,y in zip(prior,full):
    if x!=y:break
    common+=1
assert 16000<=common<=16800 and 1<=len(full)-common<=160
dest=HERE/"large";dest.mkdir(exist_ok=True);f=dest/"CASES.json";assert not f.exists()
save(f,{"warm":read(HERE/"capture/FIXTURE.json")["warm"],"cases":[{"id":"JA1024","tokens":ja[:1024],"forced":ja[1024:1040]},
    {"id":"PY1024","tokens":py[:1024],"forced":py[1024:1040],"request":code}],
    "conversation":{"messages":hist+[{"role":"user","content":q}],"rows":hi,"target":expected(row_id),"previous_token_count":len(prior),
    "common_prefix_tokens":full[:common],"suffix_tokens":full[common:],"previous_tail_tokens":prior[common:],
    "scope":"Rebuild common history in exact 1024-token ubatches plus its remainder; append suffix at its true position in the same slot. Direct decoder continuation, not Open WebUI or server cache eviction."}})
print("1024 fixtures fixed; rendered conversation",common,"shared +",len(full)-common,"new tokens; target",expected(row_id))
