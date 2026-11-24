"""Offline verification for the compact evidence archive (Python 3.11+)."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

def read(p):return json.loads(p.read_bytes())


def main(root):
    manifest=read(root/"MANIFEST_SHA256.json")["files"]
    for e in manifest:
        p=root/e["path"]
        assert not Path(e["path"]).is_absolute() and ".." not in Path(e["path"]).parts
        assert p.stat().st_size==e["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==e["sha256"]
    for f in ("score_chat.py","score_workers.py","score_pc.py"):
        subprocess.run([sys.executable,"-B",str(root/f),str(root),"--check"],check=True,stdout=subprocess.DEVNULL,timeout=120)
    for pair in ("both","MX_only","general_only","general_after_MX","MX_after_general"):
        ref=read(root/"risk"/(pair+".json"))
        for name,v in ref["by_case"].items():
            rows=list(csv.DictReader((root/"risk"/pair/(name+".tsv")).open(),delimiter="\t"))
            assert len(rows)==v["rows"] and sum(r["A_top"]!=r["B_top"] for r in rows)==v["argmax_changes"]
            assert sum(int(r["exact"]) for r in rows)==v["identical_rows"]
            assert abs(statistics.mean(float(r["TV"]) for r in rows)-v["TV"]["mean"])<1e-11
    assert all(v["full_rows_byte_equal"]==v["rows"] for v in read(root/"risk/MX_after_general.json")["grouped"].values())
    cause=read(root/"CAUSAL_RESULT.json")
    for mode,v in cause.items():
        rows=[r for n in ("JA96","PY96") for r in csv.DictReader((root/"risk/causal"/(n+"-"+mode+".tsv")).open(),delimiter="\t")]
        assert len(rows)==130 and v["total_top_changes_to_64"]==sum(r["A_top"]!=r["B_top"] for r in rows)
        assert abs(statistics.mean(float(r["TV"]) for r in rows)-v["mean_TV_to_64"])<1e-10
    facts=read(root/"PRECISION_CAUSE.json")
    for n in ("JA96","PY96"):
        for who in ("cuda","cpu"):
            rs=list(csv.DictReader((root/"precision"/(n+"-"+who+"-dot.tsv")).open(),delimiter="\t"))
            assert len(rs)==2304
            err=math.sqrt(statistics.mean((float(r["observed"])-float(r["full"]))**2 for r in rs))
            assert abs(err-facts[n]["arithmetic_sample"]["all"]["vs_GGUF_F64"][who])<1e-10
        for v in (64,128):
            folder=root/"runs"/f"precision-{v}-{v}"
            for mode in ("early","routes"):
                assert read(folder/(n+"-"+mode)/"CAPTURE.json")["prefix_logits_byte_equal_to_no_callback"]
        for layer in range(40):
            from array import array
            seq=[]
            for v in (64,128):
                a=array("i");a.frombytes((root/"runs"/f"precision-{v}-{v}"/(n+"-routes")/f"ffn_moe_down-{layer}-ids.bin").read_bytes())
                assert len(a)==576
                seq.append(a)
            changed=sum(set(seq[0][i:i+6])!=set(seq[1][i:i+6]) for i in range(0,576,6))
            assert changed==facts[n]["per_layer_routing"][layer]["token_sets_changed"]
    end=read(root/"FINAL_STATE.json")
    assert end["thresholds"]==[64,64] and end["http"]=="ok" and end["host_counters_unchanged"]==112
    print(f"Verified {len(manifest)} files, 32 native requests, 16 worker traces, all five factorial contrasts, 4,608 dot samples and four single-layer arms.")


if __name__=="__main__":main(Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve())
