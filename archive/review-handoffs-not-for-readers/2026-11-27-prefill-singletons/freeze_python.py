"""Freeze a new Python code-review prefix using just the deployed template/tokenizer."""
import hashlib
import json
import urllib.request
from build_capture import HERE,LAB,read,save,sha
from pathlib import Path

src=LAB/"glu-split-20261125/analyze.py"
text=src.read_text()
code=text[text.index("def timing(root):"):text.index('if __name__=="__main__"')]
prompt="Review this Python benchmark aggregation function. Explain what counts as an independent sample and which comparisons could be biased by process order.\n\n```python\n"+code+"\n```"
cfg=read(LAB/"deepseek41-bringup-20261101/server-config.json")
assert sha(LAB/"deepseek41-bringup-20261101/server-config.json")==read(LAB/"current/runtime.json")["service_config"]["sha256"]
base=f"http://{cfg['host']}:{cfg['port']}";key=Path(cfg['api_key_file']).read_text().strip()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path,data):
    req=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
    with opener.open(req,timeout=20) as f:return json.load(f)
body={"messages":[{"role":"user","content":prompt}],"add_generation_prompt":True,"chat_template_kwargs":{"thinking_mode":"chat"}}
render=post("/apply-template",body)["prompt"]
full=post("/tokenize",{"content":render,"add_special":True,"parse_special":True,"with_pieces":False})["tokens"]
assert len(full)>=512 and all(isinstance(x,int) for x in full)
out=HERE/"capture/PY512.json";assert not out.exists()
save(out,{"id":"PY512","tokens":full[:512],"warm":read(HERE/"capture/FIXTURE.json")["warm"],
          "request":body,"rendered":render,"rendered_token_count":len(full),"source_sha256":sha(src),
          "scope":"first 512 actual token IDs of one previously unused rendered Python code-review message; truncated prefix for compute, no inference at fixture creation"})
print(len(full),"rendered tokens;",sha(out))
