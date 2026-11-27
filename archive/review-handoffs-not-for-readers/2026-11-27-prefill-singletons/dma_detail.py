"""Bandwidth on device H2D intervals versus bandwidth over the whole measured FFN."""
import csv
import json
from pathlib import Path
from analyze import size

def calculate(root):
    result={}
    for case,index in json.loads((root/"READY_INDEX.json").read_bytes()).items():
        result[case]={}
        for layer,paths in index.items():
            run=root/"runs"/paths["profile"];rec=json.loads((run/"RUN.json").read_bytes())
            arms={}
            for row in (s for s in rec["samples"] if s["phase"]=="CUPTI"):
                lo,hi=row["start"],row["end"];b=0;intervals=[]
                with (run/(row["arm"]+"-activity.tsv")).open() as src:
                    for e in csv.DictReader(src,delimiter="\t"):
                        if e["kind"]!="1" or e["detail"]!="1":continue
                        a,z=int(e["start_ns"]),int(e["end_ns"])
                        if lo<=a<z<=hi:b+=int(e["bytes"]);intervals.append((a,z))
                span=size(intervals);assert span>0
                arms[row["arm"]]={"actual_H2D_B":b,"H2D_union_ms":span/1e6,"active_decimal_GB_s":b/span,"whole_call_GB_s":b/(hi-lo)}
            result[case][layer]=arms
    return result
if __name__=="__main__":
    r=Path(__file__).resolve().parent;(r/"DMA_DETAIL.json").write_text(json.dumps(calculate(r),indent=2)+"\n")
