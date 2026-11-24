"""Align synchronous CPU dispatch, CUDA device activity and blocking host APIs."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"tools"))
import summarize_trace
import parent_recheck
import xeon_imc


def read(p):return json.loads(p.read_bytes())
def table(p,name):
    with (p/("trace169-"+name+".tsv")).open() as f:return list(csv.DictReader(f,delimiter="\t"))


def merge(rows):
    out=[]
    for a,b in sorted(rows):
        assert a<=b
        if a==b:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out


def size(rows):return sum(b-a for a,b in merge(rows))


def intersection(a,b):
    x=merge(a);y=merge(b);i=j=0;out=[]
    while i<len(x) and j<len(y):
        lo=max(x[i][0],y[j][0]);hi=min(x[i][1],y[j][1])
        if lo<hi:out.append((lo,hi))
        if x[i][1]<y[j][1]:i+=1
        else:j+=1
    return out


def med(values):return statistics.median(values)


def trace(root):
    d=root/"runs/observe"; external=summarize_trace.summarize(d)
    narrow=parent_recheck.inspect(d,False)
    cp=read(root/"tools/CUDA_API_NAMES.json")["names"]
    corr={int(s["correlation"]):int(s["external_id"]) for s in table(d,"correlations")}
    ts={int(x["tensor"]):x for x in table(d,"tensors")}
    nodes=collections.defaultdict(list)
    for n in table(d,"nodes"):nodes[int(n["split"])].append(ts[int(n["tensor"])])
    splits=table(d,"splits");copies=table(d,"copies");kernels=table(d,"kernels");dma=table(d,"device-copies");api=table(d,"api")
    calls=table(d,"calls");run=read(d/"RUN.json");control=read(root/"runs/control/RUN.json")
    result=[]; layer_ids=collections.defaultdict(list); layer_device=collections.defaultdict(list)
    for info,prior in zip(calls,external["calls"]):
        cid=int(info["call"]);case=int(info["case_id"]);step=int(info["call_id"]);lo=int(info["start_ns"]);hi=int(info["end_ns"])
        assert (case,step)==(prior["case_id"],prior["call_id"])
        s=[v for v in splits if int(v["call"])==cid]; cs=[v for v in copies if int(v["call"])==cid]
        ids={int(x["split"]) for x in s}|{(2<<32)+int(x["copy"]) for x in cs}|{(1<<32)+cid}
        def span(row,a,b):return int(row[a]),int(row[b])
        def inside(row):return lo<=int(row["start_ns"]) and int(row["end_ns"])<=hi
        kernel=[span(v,"start_ns","end_ns") for v in kernels if inside(v)]
        transfers=[span(v,"start_ns","end_ns") for v in dma if inside(v)]
        for group in (kernels,dma):
            assert all((corr[int(v["correlation"])] in ids)==inside(v) for v in group)
        device=kernel+transfers
        cpu=[span(v,"dispatch_ns","return_ns") for v in s if v["backend"]=="CPU"]
        cuda_dispatch=[span(v,"dispatch_ns","return_ns") for v in s if v["backend"]=="CUDA0"]
        copy=[span(v,"start_ns","return_ns") for v in cs]
        metadata=[span(v,"metadata_start_ns","dispatch_ns") for v in s]
        sync=[];sync_api=collections.defaultdict(list)
        api_by_name=collections.defaultdict(list)
        for v in api:
            if not inside(v):continue
            name=cp[v["kind"]].get(v["cbid"],"unknown")
            api_by_name[name].append(span(v,"start_ns","end_ns"))
            if re.match(r"(?:cuda(?:Stream|Device|Event)|cu(?:Stream|Ctx|Event))Synchronize(?:_|$)",name):
                val=span(v,"start_ns","end_ns");sync.append(val);sync_api[name].append(val)
        layers={}
        for split in s:
            if split["backend"]!="CPU":continue
            tensors=nodes[int(split["split"])]
            gates=[re.fullmatch(r"ffn_moe_gate-(\d+)",v["name"]) for v in tensors if v["op"]=="MUL_MAT_ID"]
            found=[int(m[1]) for m in gates if m]
            if not found:continue
            assert len(found)==1
            layer=found[0];byname={v["name"]:v["op"] for v in tensors}
            for op in ("gate","up","down"):assert byname[f"ffn_moe_{op}-{layer}"]=="MUL_MAT_ID"
            assert byname[f"ffn_moe_swiglu_limited-{layer}"]=="GLU" and layer not in layers
            a,b=span(split,"dispatch_ns","return_ns");over=size(intersection([(a,b)],device))
            layers[layer]={"cpu_ns":b-a,"device_overlap_ns":over,"nodes":len(tensors)}
            layer_ids[layer].append(b-a);layer_device[layer].append(over)
        assert sorted(layers)==list(range(37)) and len(s)==int(info["n_splits"])
        overlap=size(intersection(cpu,device));cpu_ns=size(cpu);gpu_ns=size(device);outside=hi-lo-size(cpu+device)
        assert (cpu_ns,gpu_ns,overlap,outside)==(prior["cpu_split_span_union_ns"],prior["device_kernel_or_copy_union_ns"],prior["cpu_split_and_device_ns"],prior["outside_cpu_split_and_all_device_activity_ns"])
        assert len(kernel)==prior["device_activity_count"]["kernel"] and len(transfers)==prior["device_activity_count"]["dma"]
        def outside_active(interval):return size(interval)-size(intersection(interval,cpu+device))
        # Host spans below are overlapping annotations, not additive components.
        result.append({"case":case,"step":step,"trace_call":cid,"span_ns":hi-lo,"CPU_only_ns":cpu_ns-overlap,"device_only_ns":gpu_ns-overlap,
             "overlap_ns":overlap,"neither_ns":outside,"cpu_ns":cpu_ns,"kernel_ns":size(kernel),"dma_ns":size(transfers),
             "device_ns":gpu_ns,"routed37_ns":sum(v["cpu_ns"] for v in layers.values()),"by_layer":layers,
             "explicit_host_sync_ns":size(sync),"sync_ns_outside_cpu_device":outside_active(sync),
             "host_sync_api_ns":{k:size(v) for k,v in sync_api.items()},
             "cuda_api_top_host_unions":[{"name":k,"calls":len(v),"host_union_ns":size(v)}
                  for k,v in sorted(api_by_name.items(),key=lambda p:size(p[1]),reverse=True)[:12]],
             "host_copy_span_ns":size(copy),"host_copy_ns_outside_cpu_device":outside_active(copy),
             "cuda_submit_span_ns":size(cuda_dispatch),"cuda_submit_ns_outside_cpu_device":outside_active(cuda_dispatch),
             "observer_metadata_ns":size(metadata),"observer_metadata_ns_outside_cpu_device":outside_active(metadata),
             "outside_known_cpu_device_submit_copy_sync_metadata_ns":hi-lo-size(cpu+device+cuda_dispatch+copy+sync+metadata),
             "driver_trace_clock_ns":run["blocks"][case]["output_ready_ns"][step-1]-run["blocks"][case]["start_ns"][step-1],
             "driver_control_clock_ns":control["blocks"][case]["output_ready_ns"][step-1]-control["blocks"][case]["start_ns"][step-1],
             "copy_bytes":prior["device_copy_bytes_by_cupti_direction"]})
    median_ms={key:med([c[key] for c in result])/1e6 for key,value in result[0].items() if key.endswith("_ns") and isinstance(value,int)}
    means={key:statistics.mean(c[key] for c in result)/1e6 for key in median_ms}
    return {"calls":result,"median_ms":median_ms,"mean_ms":means,"layer_median_ms":{str(k):med(v)/1e6 for k,v in sorted(layer_ids.items())},
        "layer_median_device_overlap_ms":{str(k):med(v)/1e6 for k,v in sorted(layer_device.items())},
        "shared_prefix":{"median_busy_ms":narrow["median_prefix_busy_ms"],"median_overlap_ms":narrow["median_overlap_ms"],"layer_groups":len(narrow["per_layer"])},
        "observer_receipt":read(d/"trace169-summary.json"),"independent_CUPTI_recount_agrees":True,
        "definitions":"CPU dispatch is wall time including CPU team synchronization. Device time is the union of actual kernels and memcopies. Outside time includes scheduler and measurement work; explicit host sync/copy/submit durations can overlap each other or GPU execution."}


def traffic(d):
    meta=read(d/"IMC.json")
    assert not meta["open_error"] and not meta.get("error")
    events=meta["events"]; rows=[json.loads(s) for s in (d/"host-counters.jsonl").read_text().splitlines()]
    data=read(d/"RUN.json");out={}
    for b in data["blocks"]:
        target={}
        for label,lo,hi in (("prefill",b["prefill_begin_ns"],b["prefill_ready_ns"]),("decode_5_64",b["start_ns"][4],b["output_ready_ns"][-1])):
            samples=[]
            for a,z in zip(rows,rows[1:]):
                if a["raw"][0][3]*1e9<lo or z["raw"][-1][3]*1e9>hi:continue
                durations=[];sockets=collections.defaultdict(lambda:collections.defaultdict(float));fractions=[]
                for e,x,y in zip(events,a["raw"],z["raw"],strict=True):
                    rate=xeon_imc.interval_rate(x,y,e["bytes_per_count"])
                    secs=rate["interval_seconds"];durations.append(secs);fractions.append(rate["running_fraction"])
                    sockets[str(e["socket"])][e["direction"]]+=rate["bytes_per_second"]*secs
                samples.append((statistics.mean(durations),min(fractions),sockets))
            assert len(samples)>=3
            total_time=sum(x[0] for x in samples);coverage=min(x[1] for x in samples)
            assert coverage>=.99
            by={s:{kind:sum(x[2][s][kind] for x in samples)/total_time/1e9 for kind in ("read","write")} for s in ("0","1")}
            target[label]={"window_s":(hi-lo)/1e9,"sampled_s":total_time,"whole_intervals":len(samples),"min_running_fraction":coverage,
                 "socket_GB_s":by,"host_read_GB_s":sum(v["read"] for v in by.values()),"host_write_GB_s":sum(v["write"] for v in by.values())}
        out[b["case_id"]]=target
    return out


def analyze(root):
    result=trace(root)
    result["imc"]={s:traffic(root/"runs"/s) for s in ("control","observe")}
    result["control_output_timing"]={}
    for b in read(root/"runs/control/RUN.json")["blocks"]:
        ms=[(z-a)/1e6 for a,z in zip(b["start_ns"],b["output_ready_ns"])]
        result["control_output_timing"][b["case_id"]]={"calls":len(ms),"mean_ms":statistics.mean(ms),"median_ms":med(ms),"T1_per_s":1000/statistics.mean(ms),"p95_ms":sorted(ms)[60],"prefill_s":(b["prefill_ready_ns"]-b["prefill_begin_ns"])/1e9}
    return result


if __name__=="__main__":
    root=Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
    output=analyze(root)
    if "--check" in sys.argv:
        assert json.loads(json.dumps(output))==read(root/"TIMELINE.json")
    else:(root/"TIMELINE.json").write_text(json.dumps(output,indent=2)+"\n")
    print("trace medians (ms)",output["median_ms"]);print("L7",output["layer_median_ms"]["7"],output["layer_median_device_overlap_ms"]["7"]);print("control clocks",output["control_output_timing"])
