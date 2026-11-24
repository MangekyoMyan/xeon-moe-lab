"""Export two of the measured T1s as ordinary Chrome Trace Event files (microseconds)."""
import csv
import json
from pathlib import Path
import sys


def export(root):
    folder=root/"runs/observe";dest=root/"views";dest.mkdir(exist_ok=True)
    def table(name):
        with (folder/f"trace169-{name}.tsv").open() as f:return list(csv.DictReader(f,delimiter="\t"))
    tensors={t["tensor"]:t["name"] for t in table("tensors")}
    nodes={}
    for n in table("nodes"):nodes.setdefault(n["split"],[]).append(tensors[n["tensor"]])
    labels=json.loads((root/"tools/CUDA_API_NAMES.json").read_text())["names"]
    splits=table("splits");copies=table("copies");api=table("api");ks=table("kernels");mem=table("device-copies")
    for c in table("calls"):
        if c["call_id"]!="2":continue
        lo,hi=int(c["start_ns"]),int(c["end_ns"]);ev=[]
        def add(name,track,a,b,args=None):
            a,b=int(a),int(b)
            if not lo<=a<=b<=hi:return
            ev.append({"name":name,"cat":track,"ph":"X","pid":1,"tid":track,"ts":(a-lo)/1e3,"dur":(b-a)/1e3,"args":args or {}})
        for s in splits:
            if s["call"]!=c["call"]:continue
            names=nodes[s["split"]]
            ref=next((n for n in names if n.startswith("ffn_moe_gate-")),names[0])
            add(ref,"Host "+s["backend"]+" dispatch",s["dispatch_ns"],s["return_ns"],{"split":s["ordinal"],"nodes":names})
            add("metadata","Observer",s["metadata_start_ns"],s["dispatch_ns"])
        for v in copies:
            if v["call"]==c["call"]:add("tensor copy","Host copies",v["start_ns"],v["return_ns"],{"src":v["src"],"dst":v["dst"]})
        for v in api:
            n=labels.get(v["kind"],{}).get(v["cbid"],v["cbid"])
            if "Synchronize" in n:add(n,"Host sync",v["start_ns"],v["end_ns"])
        for v in ks:add(v["name"],"GPU kernels stream "+v["stream"],v["start_ns"],v["end_ns"],{"correlation":v["correlation"]})
        for v in mem:add({"1":"HtoD","2":"DtoH","8":"DtoD"}.get(v["direction"],v["direction"]),"GPU copies stream "+v["stream"],v["start_ns"],v["end_ns"],{"bytes":int(v["bytes"])})
        tag=("ja","python")[int(c["case_id"])]
        (dest/f"{tag}-T1-02.trace.json").write_text(json.dumps({"displayTimeUnit":"ms","traceEvents":ev},separators=(",",":"))+"\n")

if __name__=="__main__":export(Path(sys.argv[1]))
