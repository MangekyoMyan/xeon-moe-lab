"""Read the full FP32 rows, check exact processed IDs and quantify each difference."""
import json
from pathlib import Path
import array
import math
R=Path(__file__).resolve().parent
def run():
    lookup={}
    for f in (R/"runs").glob("*/trace/calls.jsonl"):
        for line in f.read_text().splitlines():
            row=json.loads(line)
            if "logits_file" in row:
                lookup[row["phase"],row["pos"][-1]+1]=(f.parent/row["logits_file"],row)
    def diff(a,b):
        x=array.array("f");y=array.array("f")
        x.frombytes(a.read_bytes());y.frombytes(b.read_bytes())
        assert len(x)==len(y)==129280 and all(map(math.isfinite,x)) and all(map(math.isfinite,y))
        top=[x.index(max(x)),y.index(max(y))]
        if x.tobytes()==y.tobytes():return dict(exact=True,max_abs=0,top1=top,tv=0)
        def prob(v):
            m=max(v);es=[math.exp(z-m) for z in v];norm=sum(es)
            return [z/norm for z in es]
        return dict(exact=False,max_abs=max(abs(a-b) for a,b in zip(x,y)),top1=top,
                    tv=sum(abs(a-b) for a,b in zip(prob(x),prob(y)))/2)
    cases=[]
    for (name,end),(f,row) in sorted(lookup.items()):
        if name in ("create","split2","unsplit") or name.startswith("resume-"):
            ref=lookup.get(("split1",end))
            if not ref:continue
            j=dict(name=name,end=end,actual_ids_equal=(row["input_ids"]==ref[1]["input_ids"]),
                   backend_equal=row["ubatches"]==ref[1]["ubatches"],**diff(f,ref[0]))
            cases.append(j)
    out=R/"evidence/row-comparison.json"
    out.write_text(json.dumps(cases,indent=2)+"\n")
    for group in ("create","split2","resume-","unsplit"):
        arr=[x for x in cases if x["name"].startswith(group)]
        print(group,len(arr),sum(x["exact"] for x in arr), max(x["max_abs"] for x in arr),max(x["tv"] for x in arr),sum(x["top1"][0]==x["top1"][1] for x in arr))
    for v in cases:
        if v["name"].startswith("resume-"):print(v)
if __name__=="__main__":run()

