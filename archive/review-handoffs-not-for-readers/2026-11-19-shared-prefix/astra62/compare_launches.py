"""Compare full-device kernel identities and host launch kinds within each traced T1."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from analyze_trace import rows, sha

here=Path(__file__).resolve().parent
study=here.parent
keys=("name","grid_x","grid_y","grid_z","block_x","block_y","block_z")
graph_launch={311,312}
kernel_launch={211,214,430,431}
instantiate={310,418,436,437,443}

def extract(run):
    call_map={int(v["call"]):(int(v["case_id"]),int(v["call_id"])) for v in rows(run,"calls")}
    which={2**32+i:case for i,case in call_map.items()}
    which.update({int(s["split"]):call_map[int(s["call"])] for s in rows(run,"splits")})
    which.update({2*2**32+int(c["copy"]):call_map[int(c["call"])] for c in rows(run,"copies")})
    correlation={int(v["correlation"]):int(v["external_id"]) for v in rows(run,"correlations")}
    signatures=defaultdict(Counter);api=defaultdict(Counter);copy=defaultdict(Counter)
    for k in rows(run,"kernels"):
        token=which[correlation[int(k["correlation"])]]
        signatures[token][tuple(k[key] for key in keys)]+=1
    for k in rows(run,"api"):
        key=int(k["correlation"])
        if key not in correlation or int(k["kind"])!=5: continue
        token=which[correlation[key]];cb=int(k["cbid"])
        if cb in graph_launch: api[token]["graph_launch"]+=1
        if cb in kernel_launch: api[token]["direct_or_captured_kernel_API"]+=1
        if cb in instantiate: api[token]["graph_instantiate"]+=1
        if cb==335: api[token]["graph_exec_update"]+=1
    for k in rows(run,"device-copies"):
        ids=[int(k[n]) for n in ("correlation","runtime_correlation") if int(k[n]) in correlation]
        if not ids: raise ValueError(("copy correlation",k))
        owners={which[correlation[v]] for v in ids}
        assert len(owners)==1
        copy[owners.pop()][(k["direction"],k["bytes"])]+=1
    return signatures,api,copy

if __name__=="__main__":
    # CUPTI 12.9 identifiers in the shipped cupti_runtime_cbid.h. API count
    # includes capture-time calls, which do not dispatch twice on the GPU.
    base,ba,bc=extract(study/"main/runs/observe")
    test,ta,tc=extract(study/"trial/runs/observe-B")
    names=sorted(base)
    report=[]
    for key in names:
        missing=base[key]-test[key]; extra=test[key]-base[key]
        old_copy=bc[key]-tc[key];new_copy=tc[key]-bc[key]
        report.append(dict(case=key[0],call=key[1],matched_kernel_signature_multiset=not missing and not extra,
            baseline_kernel_count=sum(base[key].values()),candidate_kernel_count=sum(test[key].values()),
            baseline_only=[(list(k),n) for k,n in missing.items()],candidate_only=[(list(k),n) for k,n in extra.items()],
            baseline_runtime_launch_counts=ba[key],candidate_runtime_launch_counts=ta[key],
            same_physical_copy_direction_size_multiset=bc[key]==tc[key],
            different_copy_records_confined_to_device_to_device=all(direction=="8" for direction,size in list(old_copy)+list(new_copy)),
            baseline_only_copies=[(list(k),n) for k,n in old_copy.items()],candidate_only_copies=[(list(k),n) for k,n in new_copy.items()]))
    out=dict(source_sha256=sha(Path(__file__)),all_eight_kernel_multisets_equal=all(r["matched_kernel_signature_multiset"] for r in report),
             all_eight_copy_multisets_equal=all(r["same_physical_copy_direction_size_multiset"] for r in report),
             signature_fields=keys,per_call=report)
    with (here/"CUDA_LAUNCH_AND_COPY_AUDIT.json").open("x") as f: json.dump(out,f,indent=2);f.write("\n")
    print(json.dumps(out,indent=2))
