"""Compare exact early tensors, recorded router choices and independently reconstructed IQ2_S dot outputs."""
import array
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parent
B=ROOT/"precision"
def read(p):return json.loads(p.read_bytes())
def rmse(v):return math.sqrt(statistics.mean(x*x for x in v))
def read_ids(p):
    a=array.array("i");a.frombytes(p.read_bytes());return a
def dots(p):
    return list(csv.DictReader(p.open(),delimiter="\t"))


def analyze():
    results={}
    for case in ("JA96","PY96"):
        paths=[ROOT/"runs"/f"precision-{v}-{v}"/(case+"-early") for v in (64,128)]
        ea,eb=[read(p/"CAPTURE.json") for p in paths]
        assert ea["prefix_logits_byte_equal_to_no_callback"] and eb["prefix_logits_byte_equal_to_no_callback"]
        meta=[{v["file"]:v for v in e["observations"]} for e in (ea,eb)]
        assert set(meta[0])==set(meta[1]) and len(meta[0])==12
        tensors={}
        for n,m in meta[0].items():
            assert (m["type"],m["shape"])==(meta[1][n]["type"],meta[1][n]["shape"])
            aa,bb=(d/n for d in paths)
            if m["type"]=="f32":
                data=json.loads(subprocess.check_output([str(B/"compare_floats"),str(aa),str(bb)]))
                assert data["elements"]==math.prod(m["shape"])
            else:
                a,b=read_ids(aa),read_ids(bb);data={"elements":len(a),"different_bits":sum(x!=y for x,y in zip(a,b,strict=True))}
            tensors[n]={**data,"buffers":[m["buffer"],meta[1][n]["buffer"]]}
        for n in ("ffn_moe_gate-0","ffn_moe_up-0"):
            assert tensors[n+"-ids.bin"]["different_bits"]==tensors[n+"-input.bin"]["different_bits"]==0
        source=read(B/"WEIGHT_SOURCE.json");ss={v["shard"] for v in source.values()};assert len(ss)==1
        logs=[]
        for i,(d,label) in enumerate(zip(paths,("cuda","cpu"))):
            out=B/(case+"-"+label+"-dot.tsv")
            subprocess.run([str(B/"replay_dot"),next(iter(ss)),str(d),str(out)],check=True)
            logs.append(dots(out))
        groups=collections.defaultdict(list);expert_counts=collections.Counter(read_ids(paths[0]/"ffn_moe_gate-0-ids.bin"))
        for a,b in zip(*logs,strict=True):
            assert all(a[k]==b[k] for k in a if k!="observed")
            n=expert_counts[int(a["expert"])]
            # Volta, IQ2_S, 96-token sorted expert fallback: <=8 MMVQ; MMQ below 64; GEMM otherwise.
            mode="q32_fp16_scale" if n<=8 else "q32_f32_scale" if n<64 else "fp16_both"
            row={k:float(a[k]) for k in ("full","q8K_double","cpu_dot","cpu_corrected","q32_fp16_scale","q32_f32_scale","fp16_both")}
            row.update(cuda=float(a["observed"]),cpu=float(b["observed"]),mode_estimate=float(a[mode]))
            for group in ("all",a["projection"],mode):groups[group].append(row)
        metrics={}
        for group,records in groups.items():
            metrics[group]={"n":len(records),"vs_GGUF_F64":{k:rmse([v[k]-v["full"] for v in records]) for k in ("cuda","cpu","cpu_dot","cpu_corrected","mode_estimate")},
                "vs_observed_CUDA":{k:rmse([v[k]-v["cuda"] for v in records]) for k in ("full","cpu","cpu_dot","cpu_corrected","mode_estimate","q32_fp16_scale","q32_f32_scale","fp16_both")},
                "observed_CPU_vs_Q8K_F64":rmse([v["cpu"]-v["q8K_double"] for v in records]),
                "observed_CPU_vs_native_scalar_dot":rmse([v["cpu"]-v["cpu_dot"] for v in records])}
        routes=[]
        rp=[d.with_name(case+"-routes") for d in paths]
        for d in rp:assert read(d/"CAPTURE.json")["prefix_logits_byte_equal_to_no_callback"]
        for l in range(40):
            a,b=[read_ids(p/f"ffn_moe_down-{l}-ids.bin") for p in rp]
            assert len(a)==len(b)==6*96
            sets=orders=replacements=0;positions=[]
            for t in range(96):
                x,y=list(a[t*6:t*6+6]),list(b[t*6:t*6+6])
                assert len(set(x))==len(set(y))==6 and all(0<=z<384 for z in x+y)
                orders+=x!=y
                if set(x)!=set(y):
                    sets+=1;replacements+=len(set(x)-set(y));positions.append(t)
            routes.append({"layer":l,"token_sets_changed":sets,"ordering_or_set_changed":orders,
                           "experts_replaced":replacements,"token_positions_with_set_changes":positions})
        assert len(ea["placements"])==len(eb["placements"])==120
        moves=collections.Counter((n.split("-")[0],a[0],eb["placements"][n][0],a[1]) for n,a in ea["placements"].items())
        results[case]={"validated_callbacks":{"early":True,"routes":True},"tensors":tensors,"arithmetic_sample":metrics,
            "per_layer_routing":routes,"first_layer_set_change":next((x["layer"] for x in routes if x["token_sets_changed"]),None),
            "routed_token_layers_changed":sum(v["token_sets_changed"] for v in routes),
            "routed_token_layers":96*40,"expert_replacements":sum(v["experts_replaced"] for v in routes),
            "placement_transitions":[[*key,count] for key,count in sorted(moves.items())]}
    return results


if __name__=="__main__":
    r=analyze();(ROOT/"PRECISION_CAUSE.json").write_text(json.dumps(r,indent=2)+"\n")
    for k,v in r.items():
        print(k,"first changed routing",v["first_layer_set_change"],v["routed_token_layers_changed"],"/3840")
        print("early",[(n,u["different_bits"]) for n,u in v["tensors"].items()])
        print("arithmetic",v["arithmetic_sample"]["all"])
