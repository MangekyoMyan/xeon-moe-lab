"""Independent recalculation from the fixed Phase 1 timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

def audit(path):
    data=json.loads(path.read_bytes())
    out={}
    for group,clock in (("ordinary","first_ms"),("ordinary","total_ms"),("dirty_checkout","switch_ms"),("dirty_checkout","return_to_first_ms")):
        chosen=data[group]
        med={k:st.median(row[clock] for row in chosen if row["path"]==k) for k in ("native","managed")}
        med["difference_ms"]=med["managed"]-med["native"]
        out[group+"."+clock]=med
    out["ratio_of_median_differences"]=out["dirty_checkout.switch_ms"]["difference_ms"]/out["dirty_checkout.return_to_first_ms"]["difference_ms"]
    return {"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"results":out}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("path",type=Path);p.add_argument("--check",action="store_true");a=p.parse_args()
    dest=Path(__file__).with_name("PRIOR_AUDIT.json");result=audit(a.path)
    if a.check:assert result==json.loads(dest.read_bytes())
    else:dest.write_text(json.dumps(result,indent=2)+"\n")
