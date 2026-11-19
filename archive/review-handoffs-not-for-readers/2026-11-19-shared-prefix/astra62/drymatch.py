"""Replay observable prefix-candidate guards against the original live trace."""
import json
from collections import Counter
from pathlib import Path
from analyze_trace import rows, by_integer, group, exact, sha

here = Path(__file__).resolve().parent
run = here.parent / "main/runs/observe"
T = by_integer(rows(run,"tensors"),"tensor")
N = group(rows(run,"nodes"),"split")
S = by_integer(rows(run,"splits"),"split")
C = group(rows(run,"copies"),"call")
report = json.loads((run/"RUN.json").read_text())
pairs = json.loads((here/"TIMELINE.json").read_text())["per_layer"]
weights = {w["name"]:w for l in report["shared_inventory"] for w in l["weights"]}
assert [b["name"] for b in report["active_backend_providers"]] == ["CUDA0","CPU"]

def tensor(idx):
    return T[int(idx)]

def dims(t, typ, a, b=1):
    assert t["type"] == typ and [int(t["ne"+str(i)]) for i in range(4)] == [a,b,1,1], t
    assert all(int(t[k],16) for k in ("data_hex","buffer_hex")), t
    # ggml_is_contiguous_m_n skips strides for dimensions of extent one.
    block, size = {"f32":(1,4),"i32":(1,4),"q6_K":(256,210),"q8_0":(32,34)}[typ]
    assert int(t["nb0"]) == size and (b == 1 or int(t["nb1"]) == (a//block)*size), t

uses = Counter(int(n[f"src{i}"]) for split in N.values() for n in split for i in range(10) if n[f"src{i}"] != "0")
cpu_node_counts = Counter()
split_inputs = Counter()
for p in pairs:
    layer = p["layer"]
    cpu, gpu = S[p["cpu_split"]], S[p["gpu_split"]]
    ci, gi = int(cpu["ordinal"]), int(gpu["ordinal"])
    assert (cpu["backend"],gpu["backend"]) == ("CPU","CUDA0") and gi == ci+1
    all_cpu = N[p["cpu_split"]]
    c = [n for n in all_cpu if tensor(n["tensor"])["op"] not in ("VIEW","RESHAPE","TRANSPOSE","PERMUTE")]
    cpu_node_counts[(len(all_cpu),len(c))] += 1
    assert len(c) == 4, (p,c)
    for j,n in enumerate(c):
        t=tensor(n["tensor"]); dims(t,"f32",5120 if j==3 else 2304,6)
        assert t["name"] == ("ffn_moe_gate","ffn_moe_up","ffn_moe_swiglu_limited","ffn_moe_down")[j]+f"-{layer}"
        assert t["op"] == ("MUL_MAT_ID","MUL_MAT_ID","GLU","MUL_MAT_ID")[j]
        assert all(int(tensor(n[f"src{i}"])["buffer_hex"],16) for i in range(10) if n[f"src{i}"]!="0")
    x,ids=c[0]["src1"],c[0]["src2"]
    dims(tensor(x),"f32",5120);dims(tensor(ids),"i32",6)
    assert (c[1]["src1"],c[1]["src2"],c[3]["src2"]) == (x,ids,ids)
    assert (c[2]["src0"],c[2]["src1"],c[3]["src1"]) == (c[0]["tensor"],c[1]["tensor"],c[2]["tensor"])
    copies = C[int(cpu["call"])]
    hc=[d for d in copies if int(d["before_split_ordinal"]) == ci]
    dc=[d for d in copies if int(d["before_split_ordinal"]) == gi]
    split_inputs[(len(hc),len(dc))] += 1
    assert len(hc)>=2 and len(dc)>=1
    nx=tensor(exact([d for d in hc if d["dst"]==x],p)["src"])
    assert len([d for d in hc if d["dst"]==ids])==1
    g=N[p["gpu_split"]];w=g[0]
    assert tensor(w["tensor"])["name"] == f"ffn_moe_weighted-{layer}" and tensor(w["tensor"])["op"]=="MUL"
    assert any(d["src"]==c[3]["tensor"] and d["dst"]==w["src0"] for d in dc)
    at=exact([i for i,n in enumerate(g) if tensor(n["tensor"])["name"]==f"ffn_gate-{layer}"],p)
    a=g[at:at+5];assert at>0 and len(a)==5
    for j,n in enumerate(a):
        t=tensor(n["tensor"]);dims(t,"f32",2304 if j<3 else 5120)
        assert t["name"]==("ffn_gate","ffn_up","ffn_swiglu_limited","ffn_shexp","ffn_out")[j]+f"-{layer}"
        assert t["op"]==("MUL_MAT","MUL_MAT","GLU","MUL_MAT","ADD")[j] and int(t["flags"])&16
        if j<4: assert uses[int(n["tensor"])]==1 and t["view_hex"]=="0" and int(t["flags"])&2==0
        if j<3: assert t["view_offset"]=="0" and int(t["flags"])&3==0
    norm=tensor(a[0]["src1"]);dims(norm,"f32",5120)
    assert norm["name"]==f"ffn_norm-{layer}" and nx["data_hex"]==norm["data_hex"]
    assert nx["self_hex"]==norm["self_hex"] or (nx["view_hex"]==norm["self_hex"] and nx["view_offset"]=="0")
    assert (a[1]["src1"],a[2]["src0"],a[2]["src1"],a[3]["src1"],a[4]["src1"]) == (
        a[0]["src1"],a[0]["tensor"],a[1]["tensor"],a[2]["tensor"],a[3]["tensor"])
    assert tensor(a[4]["src0"])["name"]==f"ffn_moe_out-{layer}"
    assert int.from_bytes(bytes.fromhex(tensor(a[2]["tensor"])["params_hex"])[:4],"little")==6
    for j,n in enumerate((a[0],a[1],a[3])):
        t=tensor(n["src0"]);dims(t,"q8_0" if j==1 else "q6_K",2304 if j==2 else 5120,5120 if j==2 else 2304)
        assert t["name"]==f"blk.{layer}.ffn_"+("gate","up","down")[j]+"_shexp.weight"
        assert weights[t["name"]]["buffer"]=="CUDA0"
assert len(pairs)==296
out=dict(matched_pairs=len(pairs),cases=2,calls=8,layers_per_call=37,
    cpu_all_nodes_and_nonviews={str(k):v for k,v in cpu_node_counts.items()},
    cpu_and_cuda_sync_copy_counts={str(k):v for k,v in split_inputs.items()},
    guarded_helper_sha256=sha(here.parent/"prefix-engine/shared_prefix.inc"),
    verified="All recorded tensor identities, direct edges, global direct-use counts, copy endpoints, shapes, strides, types, flags, view roots/offsets, clamp type, CPU/CUDA splits, shared weight names and owner placement.",
    not_recorded="CUDA/host buffer methods and usage enum for every tensor, tensor.extra, internal scheduler uid, CPU backend function pointer, callback state, private allocation and actual enabled dispatch remain runtime checks.")
(here/"DRYMATCH.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
