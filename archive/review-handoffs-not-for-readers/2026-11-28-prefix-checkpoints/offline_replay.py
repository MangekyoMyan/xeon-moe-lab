"""Validate the public synthetic corpus, full numerical rows, image envelopes and paired workflows."""
import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from summarize import agg
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    manifest=read(R/"MANIFEST.json")
    for name,m in manifest.items():
        data=(R/name).read_bytes()
        assert len(data)==m["bytes"] and sha(data)==m["sha256"],name
    assert agg(R)==read(R/"SUMMARY.json")
    refs={}
    for line in (R/"native-calls.jsonl").read_text().splitlines():
        row=json.loads(line)
        if "row_sha256" in row:
            refs[row["variant"],row["phase"],row["pos"][-1]+1]=row
    rows={}
    for f in (R/"rows").glob("*.gz"):
        data=gzip.decompress(f.read_bytes())
        assert len(data)==129280*4 and sha(data)==f.name.split(".")[0]
        rows[sha(data)]=data
    def verify(variant,report):
        count=0
        for j in report:
            base=refs[variant,"split1",j["end"]]
            test=refs[variant,j["name"],j["end"]]
            x,y=rows[base["row_sha256"]],rows[test["row_sha256"]]
            assert (test["input_ids"]==base["input_ids"])==j["actual_ids_equal"]
            assert (test["ubatches"]==base["ubatches"])==j["backend_equal"]
            assert (x==y)==j["exact"]
            if x!=y:
                a=array.array("f");b=array.array("f");a.frombytes(y);b.frombytes(x)
                if sys.byteorder!="little":a.byteswap();b.byteswap()
                gap=max(abs(u-v) for u,v in zip(a,b))
                assert abs(gap-j["max_abs"])<1e-6 and [a.index(max(a)),b.index(max(b))]==j["top1"]
            if j["name"].startswith("resume-"):count+=int(x==y)
        return count
    assert verify("compact",read(R/"fixtures/compact/row-comparison.json"))==0
    assert verify("physical",read(R/"fixtures/physical/row-comparison.json"))==9
    for j in read(R/"RING.json")["comparisons"]:
        a=refs["ring",j["control"],j["end"]];b=refs["ring",j["name"],j["end"]]
        assert a["row_sha256"]==b["row_sha256"] and a["input_ids"]==b["input_ids"] and a["ubatches"]==b["ubatches"]
    long=read(R/"long-evidence/paired.json");pattern=read(R/"long-evidence/fixture.json")
    assert len(long)==6 and next(i for i,(x,y) in enumerate(zip(pattern["ids"],pattern["edited_ids"])) if x!=y)==pattern["p"]
    for case in long:
        receipt=case["receipt"]
        assert case["seed_restore"]["replayed"]==0 and case["seed_restore"]["exact_head"]
        assert receipt["timings"]["prompt_n"]==(428 if case["mode"]=="fixed" else 3500)
        if case["mode"]=="fixed":assert receipt["resume"]["from_position"]==3072
    import tempfile
    sys.path.insert(0,str(R/"source"))
    from native import image_tokens, unpack
    count=0
    for item in read(R/"image-index.json"):
        if not item["parts"]:continue
        if item["variant"]=="long":
            e=next(r for r in read(R/"long-evidence/anchor.json")["positions"] if r["position"]==item["position"])
            expected_ids=read(R/"long-evidence/fixture.json")["ids"][:item["position"]]
        else:
            reference=read(R/"fixtures"/item["variant"]/("ring-create.json" if item["variant"]=="ring" else "create.json"))
            e=next(r for r in reference["events"] if r["position"]==item["position"])
            expected_ids=e["ids"]
        hasher=hashlib.sha256()
        with tempfile.NamedTemporaryFile(suffix=".slot") as out:
            for part in item["parts"]:
                data=gzip.decompress((R/part).read_bytes());hasher.update(data);out.write(data)
            out.flush()
            assert hasher.hexdigest()==e["sha"]==item["sha256"] and out.tell()==item["bytes"]
            assert unpack(image_tokens(Path(out.name),131072))==expected_ids
        count+=1
    subprocess.run([sys.executable,"-B","-m","unittest","discover","-q"],cwd=R/"source",check=True)
    print(json.dumps(dict(manifest_files=len(manifest),distinct_full_rows=len(rows),
          rebuilt_and_checked_full_images=count,workflow_operations=read(R/"SUMMARY.json")["operations"],
          new_process_exact_rows=9,ring_comparisons=len(read(R/"RING.json")["comparisons"]))))
if __name__=="__main__":main()

