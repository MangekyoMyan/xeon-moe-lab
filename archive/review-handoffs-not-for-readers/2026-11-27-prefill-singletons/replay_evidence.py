"""Replay the timing record; optionally verify the release ZIP's binary arrays."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import struct
import zipfile
from analyze import analyze
from full_summary import score
from large_summary import summarize
from dma_detail import calculate

root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_bytes())
def digest(file):
    with file.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arrays",type=Path,nargs="+",help="complete release ZIP(s) with objects/<SHA256> entries")
    args=p.parse_args()
    for n,v in read("MANIFEST.json").items():
        assert (root/n).stat().st_size==v["bytes"] and digest(root/n)==v["sha256"],n
    for case in ("JA512","PY512"):
        for matched,name in ((False,"RESULTS_"),(True,"MATCHED_")):
            assert analyze(root,case,matched)==read(name+case+".json")
    whole=read("WHOLE_MODEL.json")
    assert score(root,whole["driver_run"])==whole
    long=read("LARGE_RESULT.json")
    assert summarize(root,long["run"])==long
    assert calculate(root)==read("DMA_DETAIL.json")
    m=read("ARRAYS.json")
    def file_hash(path):return m["runs/"+path]["sha256"]
    for case,indices in read("READY_INDEX.json").items():
        cap="capture-"+case+"/"
        assert file_hash(cap+"native-logits.f32")==file_hash(cap+"capture-logits.f32")
        for l,row in indices.items():
            prefix=row["primary"]+"/"
            assert file_hash(prefix+"G-sum.f32")==file_hash(cap+"ffn_moe_out-"+l+".bin")
            assert file_hash(prefix+"S-sum.f32")==file_hash(prefix+"P-sum.f32")
        w=whole["driver_run"]+"/"+case+"-"
        assert file_hash(w+"S.f32")==file_hash(w+"P.f32") and file_hash(w+"O.f32")==file_hash(w+"GP.f32")
        for target in ("plain","P4","P24"):
            assert file_hash("full-model-check/"+case+"-"+target+".f32")==file_hash(w+("O" if target=="plain" else target)+".f32")
    if args.arrays:
        found=set(); owners={}
        for case in ("JA512","PY512"):
            for l in (4,24):owners[file_hash("capture-"+case+"/ids-"+str(l)+".bin")]=(case,l)
        for f in args.arrays:
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.startswith("objects/") and not n.endswith("/"):
                        h=n.split("/")[1]
                        with z.open(n) as raw:assert hashlib.file_digest(raw,"sha256").hexdigest()==h,n
                        found.add(h)
                        if h in owners:
                            case,l=owners[h];ids=struct.unpack("<3072i",z.read(n))
                            assert all(len(set(ids[i:i+6]))==6 for i in range(0,3072,6))
                            observed={e:ids.count(e) for e in set(ids)}
                            layer=next(x for x in read("ROUTING.json")[case] if x["layer"]==l)
                            assert observed=={v["id"]:v["inputs"] for v in layer["groups"]}
        assert found=={v["sha256"] for v in m.values()}
    print("Verified both route/transfer comparisons, full-model pairs and",len(m),"indexed arrays.")
if __name__=="__main__":main()
