"""Freeze the first-layer router gap using JA only, and report output gaps on saved common histories."""
import array
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
PREV=LAB/"worker-balance-20261124"
sys.path.insert(0,str(LAB))
from index_model_cache import unpack,string,value
def load(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def blob(p,code):
    a=array.array(code);a.frombytes(p.read_bytes());return a
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def f32(v):return struct.unpack("<f",struct.pack("<f",v))[0]


def bias():
    folder=next((LAB/"models").glob("orcarouter--DeepSeek-V4.1*"))/"Mix-256G"
    for p in sorted(folder.glob("*.gguf")):
        with p.open("rb") as f:
            assert f.read(4)==b"GGUF" and unpack(f,"I")==3
            nt,nk=unpack(f,"Q"),unpack(f,"Q");align=32
            for _ in range(nk):
                key=string(f);kind=unpack(f,"I")
                got=value(f,kind,key=="general.alignment")
                if key=="general.alignment":align=got
            hit=None
            for _ in range(nt):
                n=string(f);dims=[unpack(f,"Q") for _ in range(unpack(f,"I"))];q=unpack(f,"I");off=unpack(f,"Q")
                if n=="blk.0.exp_probs_b.bias":
                    assert dims==[384] and q==0
                    hit=off
            if hit is not None:
                base=(f.tell()+align-1)//align*align
                f.seek(base+hit);raw=f.read(384*4);assert len(raw)==384*4
                return list(struct.unpack("<384f",raw)),{"shard":p.name,"offset":base+hit,"sha256":hashlib.sha256(raw).hexdigest()}
    raise ValueError("router bias missing")


def gaps(vals,expected,b):
    out=[]
    for t in range(96):
        weights=[]
        for i,x in enumerate(vals[t*384:(t+1)*384]):
            soft=x if x>20 else f32(math.log1p(f32(math.exp(x))))
            weights.append(f32(f32(math.sqrt(soft))+b[i]))
        top=sorted(range(384),key=lambda i:(-weights[i],i))[:7]
        assert top[:6]==list(expected[t*6:t*6+6]),(t,top,expected[t*6:t*6+6])
        out.append({"token":t,"sixth":top[5],"seventh":top[6],"gap":f32(weights[top[5]]-weights[top[6]])})
    return out


def output_screen():
    groups=collections.defaultdict(list)
    for p in sorted((PREV/"risk/both").glob("*.tsv")):
        n=p.stem
        key="fixed96" if n in ("JA96","PY96") else "other_geometry" if n in ("JA160","PY160","hold-short-control","hold-long-control") else "six_chats"
        with p.open() as f:
            for v in csv.DictReader(f,delimiter="\t"):
                groups[key].append((n,int(v["row"]),float(v["A_margin"]),v["A_top"]!=v["B_top"]))
    out={}
    for k,vs in groups.items():
        out[k]={"rows":len(vs),"changes":sum(c for _,_,_,c in vs),"cuts":{}}
        for limit in (.05,.10,.20,.25,.50,1.):
            flagged=[v for v in vs if v[2]<=limit]
            out[k]["cuts"][str(limit)]={"flagged":len(flagged),"changed_in_flagged":sum(v[-1] for v in flagged),
                                        "stable_in_flagged":sum(not v[-1] for v in flagged)}
    return out


def main():
    b,info=bias();rows={}
    for label in ("JA96","PY96"):
        d=PREV/"runs/precision-64-64"/(label+"-early")
        rows[label]=gaps(blob(d/"ffn_moe_logits-0.bin","f"),blob(d/"ffn_moe_gate-0-ids.bin","i"),b)
    ordered=sorted(r["gap"] for r in rows["JA96"])
    cuts={str(p):f32((ordered[math.ceil(len(ordered)*p)-1]+ordered[math.ceil(len(ordered)*p)])/2) for p in (.05,.10,.20)}
    save(ROOT/"DESIGN.json",{"cut_selection":"midpoint enclosing the lowest 10 percent, rounded up, in the JA96 L0 router. No alternative-output/quality labels used. Python and six new chat prompts serve as holdouts.",
        "candidates_for_descriptive_plots":cuts,"threshold":cuts["0.1"],"compare":"biased sqrt(softplus(router logits)) sixth score minus seventh score",
        "bias":info,"L0_frozen":rows,"text_only":True,"range":[64,127],"num_routed_experts":384,
        "arms":{"F":"shipped CUDA control","D":"40 router gap decisions, first-layer flag plan, ordinary arithmetic","S":"correct only first-layer gate/up expert columns for near-boundary tokens",
                "R":"same number of corrected columns per prompt across the six chosen experts per token, rotate token mask by 37 before routing","A":"correct all first-layer gate/up columns"},
        "reference":"Compare choices and NLL on fixed histories, independent chat task outputs, and local FP64 error; full compensation is a numerical comparator, not a truth label."})
    save(ROOT/"OUTPUT_GAPS.json",output_screen())
    print(cuts,{k:sum(x["gap"]<=cuts["0.1"] for x in v) for k,v in rows.items()})

if __name__=="__main__":main()
