"""Verify a supplemental ZIP, optionally regenerating all five full-vocabulary comparisons using C++17."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

INTEGER_COLUMNS={"row","target","exact","A_top","B_top","A_second","B_second","Btop_rank_in_A","Atop_rank_in_B","top5_intersection"}

def validate_scores(expected, actual):
    original=list(csv.DictReader(expected.decode().splitlines(),delimiter="\t"))
    with actual.open() as stream:rebuilt=list(csv.DictReader(stream,delimiter="\t"))
    if not original or len(original)!=len(rebuilt):raise ValueError("score row count")
    for a,b in zip(original,rebuilt,strict=True):
        if a.keys()!=b.keys():raise ValueError("score columns")
        for key in a:
            ok=int(a[key])==int(b[key]) if key in INTEGER_COLUMNS else math.isclose(float(a[key]),float(b[key]),rel_tol=5e-10,abs_tol=1e-9)
            if not ok:raise ValueError((actual.name,a["row"],key,a[key],b[key]))

def check(archive, rescore=False):
    with ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        names=z.namelist()
        if len(set(names))!=len(names) or any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError("invalid ZIP member name")
        folder=Path(tmp);index=json.loads(z.read("LOGIT_INDEX.json"));fixture=json.loads(z.read("FIXTURES.json"))["cases"]
        vocab=index["vocab"];count=rows=contrasts=0
        if vocab!=129280:raise ValueError("vocabulary size")
        if rescore:
            code=folder/"score_logits.cpp";code.write_bytes(z.read("score_logits.cpp"))
            subprocess.run(["g++","-std=c++17","-O2","-ffp-contract=off",str(code),"-o",str(folder/"score_logits")],check=True)
        for case,info in index["cases"].items():
            if case not in fixture or info["rows"]!=len(fixture[case]["targets"]):raise ValueError("fixture mismatch")
            arrays={}
            for arm,entry in info["arms"].items():
                name=entry["member"];expected=4*vocab*info["rows"]
                if name!=f"logits/{arm}/{case}.f32" or z.getinfo(name).file_size!=expected:raise ValueError("F32 array shape")
                digest=hashlib.sha256()
                with z.open(name) as src:
                    if rescore:
                        dest=folder/(arm+".f32");arrays[arm]=dest
                        with dest.open("wb") as out:
                            for block in iter(lambda:src.read(1024**2),b""):digest.update(block);out.write(block)
                    else:
                        for block in iter(lambda:src.read(1024**2),b""):digest.update(block)
                if digest.hexdigest()!=entry["sha256"]:raise ValueError("F32 digest: "+name)
                for a in entry["verified_originals"]:
                    native=json.loads(z.read("receipts/"+a["run"]+".json"))
                    if native["hashes"][a["filename"]]!=digest.hexdigest():raise ValueError("original receipt: "+name)
                count+=1;rows+=info["rows"]
            if rescore:
                targets=folder/"targets.ids";targets.write_text("\n".join(map(str,fixture[case]["targets"]))+"\n")
                for pair in ("F-S","F-R","F-A","S-A","R-A"):
                    left,right=pair.split("-")
                    if right not in arrays:continue
                    out=folder/"recomputed.tsv"
                    subprocess.run([str(folder/"score_logits"),str(arrays[left]),str(arrays[right]),str(targets),str(out)],check=True)
                    validate_scores(z.read(f"scores/{pair}/{case}.tsv"),out);contrasts+=1
                for p in arrays.values():p.unlink()
        if count!=index["distinct_arrays"] or rows!=index["distinct_rows"]:raise ValueError("index totals")
        return {"archive":Path(archive).name,"distinct_arrays":count,"distinct_rows":rows,"vocab":vocab,"rescored_case_pairs":contrasts}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives",type=Path,nargs="+")
    parser.add_argument("--rescore",action="store_true",help="Requires a C++17 compiler to regenerate the recorded per-position metrics")
    args=parser.parse_args()
    for path in args.archives:print(json.dumps(check(path,args.rescore)))
