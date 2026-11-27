"""Offline only. Recompute all timings and CUPTI unions from the archived calls."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
ARMS=("G","C","Gprobe","S","P")
def read(p):return json.loads(p.read_bytes())
def stats(xs):
    s=sorted(xs)
    return dict(n=len(s),mean=st.mean(s),median=st.median(s),p95=s[math.ceil(.95*len(s))-1],min=s[0],max=s[-1])
def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out
def size(xs):return sum(b-a for a,b in merge(xs))
def intersection(a,b):
    return size([(max(x,u),min(y,v)) for x,y in merge(a) for u,v in merge(b) if u<y and x<v])
def weights(route,arm):
    if arm=="C":return []
    ids=sorted(x["id"] for x in route["groups"] if arm in ("G","Gprobe") or x["inputs"]>=2)
    spans=[]
    for id in ids:
        if spans and id==spans[-1][1]+1:spans[-1]=(spans[-1][0],id)
        else:spans.append((id,id))
    per={"iq2_s":3778560,"iq3_xxs":4515840,"iq3_s":5068800,"mxfp4":6266880}
    return [(hi-lo+1)*per[t]+(512 if hi<383 else 0) for t in route["weight_types"] for lo,hi in spans]
def profile(folder,run,route):
    results={}
    cbids=read(ROOT/"CUDA_API_NAMES.json")["names"]
    for s in (r for r in run["samples"] if r["phase"]=="CUPTI"):
        key=s["arm"];lo=s["start"];hi=s["end"];kern=[];dma=[];cpu=[];sync=[];hostcopies=[];bytes_by=collections.Counter();calls=collections.Counter();h2ds=collections.Counter()
        if s["cpu_begin"]:cpu.append((s["cpu_begin"],s["cpu_end"]))
        with (folder/(key+"-activity.tsv")).open() as f:
            for r in csv.DictReader(f,delimiter="\t"):
                a,b=int(r["start_ns"]),int(r["end_ns"]);k=int(r["kind"]);d=int(r["detail"])
                if a>=hi or b<=lo or b<=a:continue
                iv=(max(a,lo),min(b,hi))
                if k in (1,2):assert lo<=a and b<=hi,"device operation straddled ready boundary"
                if k==1:
                    dma.append(iv);name={1:"H2D",2:"D2H",8:"D2D"}.get(d,str(d))
                    bytes_by[name]+=int(r["bytes"]);calls[name]+=1
                    if name=="H2D":h2ds[int(r["bytes"])]+=1
                elif k==2:kern.append(iv)
                elif k in (4,5):
                    n=cbids.get(str(k),{}).get(str(d),"")
                    if "Synchronize" in n:sync.append(iv)
                    if "Memcpy" in n:hostcopies.append(iv)
        assert kern and dma
        wt=weights(route,key)
        required=collections.Counter(wt);missing=required-h2ds
        device=merge(dma+kern);covered=size(device+cpu)
        results[key]={"wall_ms":(hi-lo)/1e6,"transferred_bytes":dict(bytes_by),"transfer_counts":dict(calls),"kernels":len(kern),
          "weight_range_calls_from_route":len(wt),"weight_range_bytes_including_padding":sum(wt),
          "range_sizes_missing_from_observed_H2D":{str(n):v for n,v in missing.items()},"kernel_union_ms":size(kern)/1e6,"DMA_union_ms":size(dma)/1e6,
          "device_union_ms":size(device)/1e6,"device_inactive_ms":(hi-lo-size(device))/1e6,
          "CPU_graph_ms":size(cpu)/1e6,"CPU_device_overlap_ms":intersection(cpu,device)/1e6,"CPU_DMA_overlap_ms":intersection(cpu,dma)/1e6,
          "outside_CPU_graph_and_device_ms":(hi-lo-covered)/1e6,
          "host_sync_api_union_ms":size(sync)/1e6,"host_copy_api_union_ms":size(hostcopies)/1e6}
    assert len(results)==5 and run["cupti_dropped"]==run["cupti_errors"]==0
    return results
def analyze(root,label,matched=False):
    design=read(root/"DESIGN.json");routes={v["layer"]:v for v in read(root/"ROUTING.json")[label]}
    index=read(root/("READY_INDEX.json" if matched else "RUNS_INDEX.json"))[label]
    overall={}
    for l in design["layers"]:
        if str(l) not in index:continue
        pf=root/"runs"/index[str(l)]["primary"];df=root/"runs"/index[str(l)]["profile"]
        r=read(pf/"RUN.json");d=read(df/"RUN.json")
        for x in (r,d):
            assert x["passed"] and x["S_P_slots"]["bit_different"]==x["S_P_sum"]["bit_different"]==0 and x["histogram"]==r["histogram"]
        assert len(r["cpu_singleton_slots"])==routes[l]["singleton"]
        a={"native_matches_model":r["native_matches_model"] and d["native_matches_model"],
           "first_ms":{s["arm"]:(s["end"]-s["start"])/1e6 for s in r["samples"] if s["phase"]=="first"},
           "conditions":{},"per_block":{},"quality":r["quality"],"CUPTI":profile(df,d,routes[l])}
        advance=True
        for pressure in (False,True):
            k="LLC_pressure" if pressure else "warm";arr={};a["conditions"][k]=arr;a["per_block"][k]=[]
            for arm in ARMS:
                rows=[s for s in r["samples"] if s["phase"]=="primary" and s["llc_pressure"]==pressure and s["arm"]==arm]
                assert len(rows)==2*design["reps"]
                arr[arm]=stats([(s["end"]-s["start"])/1e6 for s in rows])
                if any(s["cpu_begin"] for s in rows):arr[arm]["CPU_span_median_ms"]=st.median((s["cpu_end"]-s["cpu_begin"])/1e6 for s in rows)
            for block in (0,1):
                m={arm:st.median((s["end"]-s["start"])/1e6 for s in r["samples"] if s["phase"]=="primary" and s["llc_pressure"]==pressure and s["block"]==block and s["arm"]==arm) for arm in ARMS}
                base=min(("G","C"),key=m.get);split=min(("S","P"),key=m.get);rel=m[split]/m[base]
                a["per_block"][k].append(dict(medians_ms=m,best_native=base,best_split=split,ratio=rel))
                advance &= rel<1
        a["passes_comparison_of_block_medians"]=advance and a["native_matches_model"]
        overall[str(l)]=a
    return overall
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");p.add_argument("--check",action="store_true");p.add_argument("--matched",action="store_true");args=p.parse_args()
    result=analyze(ROOT,args.label,args.matched);path=ROOT/(("MATCHED_" if args.matched else "RESULTS_")+args.label+".json")
    if args.check:assert result==read(path)
    else:path.write_text(json.dumps(result,indent=2)+"\n")
    for l,v in result.items():print(l,v["native_matches_model"],v["passes_comparison_of_block_medians"],{k:{arm:round(x["median"],3) for arm,x in val.items()} for k,val in v["conditions"].items()})
