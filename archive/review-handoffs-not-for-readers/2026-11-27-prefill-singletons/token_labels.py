"""Label the teacher-forced Python divergences using the deployed model tokenizer."""
from pathlib import Path
import csv
import json
import urllib.request
from build_capture import HERE,LAB,read,save
cfg=read(LAB/"deepseek41-bringup-20261101/server-config.json");key=Path(cfg["api_key_file"]).read_text().strip()
url="http://"+cfg["host"]+":"+str(cfg["port"])+"/detokenize"
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with (HERE/"scores/live-large-1024/PY1024-P.tsv").open() as f:rows=[r for r in csv.DictReader(f,delimiter="\t") if r["A_top"]!=r["B_top"]]
labels={}
for token in sorted({int(r[k]) for r in rows for k in ("target","A_top","B_top")}):
    req=urllib.request.Request(url,data=json.dumps({"tokens":[token]}).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    with op.open(req,timeout=12) as f:labels[str(token)]=json.load(f)["content"]
save(HERE/"TOKEN_CHANGES.json",{"case":"PY1024","mode":"same teacher-forced prefix for each pair","pieces":labels,
     "rows":[{k:int(r[k]) for k in ("row","target","A_top","B_top","Btop_rank_in_A")} for r in rows]})
print(labels)
