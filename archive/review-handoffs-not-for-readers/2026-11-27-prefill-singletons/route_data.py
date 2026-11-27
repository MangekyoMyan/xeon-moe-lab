"""Recount routing from packed little-endian IDs and measured GGUF types."""
import collections
import struct
from build_capture import HERE,LAB,read,save,sha

sizes={"iq2_s":3778560,"iq3_xxs":4515840,"iq3_s":5068800,"mxfp4":6266880}
typ=read(LAB.parent/"xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-25-specialization-consult/SAVED_TRACE_RESULT.json")["cupti_graph_structure"]["cpu_layers_gate_up_down_types"]
plan={}
for line in (LAB/read(LAB/"current/runtime.json")["plan"]["file"]).read_text().splitlines()[1:]:
    fields=line.split(); plan[fields[0]]=set(map(int,fields[6:]))

def classify(path,t,l,types):
    raw=path.read_bytes();ids=struct.unpack(f"<{t*6}i",raw)
    for i in range(t):assert len(set(ids[i*6:i*6+6]))==6 and all(0<=e<384 for e in ids[i*6:i*6+6])
    count=collections.Counter(ids);triplet=sum(sizes[v] for v in types)
    groups=[{"id":e,"inputs":n,"slot_count":n,"native_B":triplet,
             "cpu_compact":[e in plan.get(f"blk.{l}.ffn_{p}_exps.weight",set()) for p in ("gate","up","down")],
             "device_in_split":"CPU" if n==1 else "CUDA"} for e,n in sorted(count.items())]
    single=sum(n==1 for n in count.values())
    return {"layer":l,"tokens":t,"slots":t*6,"used":len(count),"singleton":single,"logical_B":len(count)*triplet,
            "singleton_B":single*triplet,"weight_types":types,"id_sha256":sha(path),"histogram":dict(sorted(collections.Counter(count.values()).items())),
            "groups":groups}
def main():
    old={};plan_overlap={}
    for label in ("JA","PY"):
        base=LAB/"worker-balance-20261124/runs/precision-64-64"/(label+"96-routes")
        vals=[classify(base/f"ffn_moe_down-{l}-ids.bin",96,l,typ[str(l)]) for l in range(37)]
        old[label]={k:sum(v[k] for v in vals) for k in ("slots","used","singleton","logical_B","singleton_B")}
        tiny=[g for v in vals for g in v["groups"] if g["inputs"]<=2]
        old[label].update(at_most_two_slots=sum(g["inputs"] for g in tiny),at_most_two_B=sum(g["native_B"] for g in tiny))
        cp=LAB/"worker-balance-20261124/runs/precision-128-128"/(label+"96-routes")
        iq=[]
        for l in range(37):
            v=classify(cp/f"ffn_moe_down-{l}-ids.bin",96,l,typ[str(l)])
            iq.extend(h for g in v["groups"] if g["inputs"]==1 for t,h in zip(v["weight_types"],g["cpu_compact"]) if t.startswith("iq"))
        plan_overlap[label]={"selected_singleton_IQ_projections":len(iq),"in_CPU_plan":sum(iq),"actual_hit_count_measured":False}
    result={"historical_96":old,"historical_128_threshold_96_inputs":plan_overlap}
    for case in ("JA512","PY512"):
        cap=HERE/"runs"/("capture-"+case)
        if not (cap/"CHECKED.json").exists():continue
        r=read(cap/"RUN.json");actual=[]
        for l in (4,24):
            types=[v["type"] for v in r["weight_meta"][str(l)]]
            assert [v["nb"][2] for v in r["weight_meta"][str(l)]]==[sizes[t] for t in types]
            actual.append(classify(cap/f"ids-{l}.bin",512,l,types))
        result[case]=actual
    save(HERE/"ROUTING.json",result)
if __name__=="__main__":main()
