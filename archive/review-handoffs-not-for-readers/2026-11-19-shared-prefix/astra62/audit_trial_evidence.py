"""Recheck pinned runtime identities, all quality rows, and inclusive clock contributions."""
import array
import hashlib
import json
import math
from functools import lru_cache
from fractions import Fraction
from pathlib import Path

here=Path(__file__).resolve().parent
study=here.parent
trial=study/"trial"
read=lambda p:json.loads(p.read_text())
@lru_cache(maxsize=None)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

orig=read(study/"INITIAL_RUNTIME.json")
all_pins=dict(orig["mapped_runtime_hashes"])
engine=read(study/"prefix-engine/BUILD.json")
candidate=engine["arms"]["candidate"]
all_pins[candidate["library"]]=candidate["library_sha256"]
assert engine["arms"]["control"]["bit_identical_deployed"] and engine["arms"]["control"]["bit_identical_object"]
assert all(sha(path)==digest for path,digest in all_pins.items())
frozen_cases=read(study/"main/CASES.json")["cases"]
ref_dir=study/"main/runs/baseline"
ref=read(ref_dir/"RUN.json")
base_symbol_paths={p["symbol"]:p["path"] for p in ref["providers"]}
assert base_symbol_paths["ggml_backend_sched_graph_compute_async"] in all_pins
ref_backends={p["name"]:p["graph_compute_provider"] for p in ref["active_backend_providers"]}
assert set(ref_backends)=={"CUDA0","CPU"}
receipts={}
quality_digests={}
regenerations={}
timing={0:[],1:[]}
for spec in read(trial/"DESIGN.json")["stages"]:
    name=spec["name"]
    directory=trial/"runs"/name
    actual=read(directory/"RUN.json");expected=read(directory/"PROVIDER_EXPECTED.json");l=read(directory/"LAUNCH.json")
    assert actual["passed"] and expected["model_loaded"]==False
    assert actual["context"]==ref["context"] and actual["placement"]==ref["placement"] and actual["shared_inventory"]==ref["shared_inventory"]
    assert actual["preparation"]==ref["preparation"] and actual["compact_bytes"]==127401984000 and actual["vision_loaded"]
    assert (l["environment"].get("LD_PRELOAD") is not None)==(spec["stage"]=="observe")
    assert l["environment"]["XEON_169_SHARED_PREFIX"]==("1" if spec["arm"]=="B" else "0")
    assert l["baseline_config_sha256"]==orig["config_sha256"] and l["reserve_GiB"]==64
    assert l["driver_sha256"]==sha(trial/"driver") and l["cases_sha256"]==sha(trial/"CASES.json")
    assert expected["providers"]==actual["providers"] and expected["active_backend_providers"]==actual["active_backend_providers"]
    wanted=dict(base_symbol_paths)
    if spec["arm"]=="B":
        wanted["ggml_backend_sched_graph_compute_async"]=candidate["library"]
        assert actual["prefix_provider"]==candidate["library"]
    else:
        assert "prefix_provider" not in actual
    assert {v["symbol"]:v["path"] for v in actual["providers"]}==wanted
    for provider in actual["providers"]:
        path=Path(provider["path"]);st=path.stat()
        assert provider["bytes"]==st.st_size and provider["device"]==st.st_dev and provider["inode"]==st.st_ino
        assert sha(str(path))==all_pins[str(path)]
    assert {p["name"]:p["graph_compute_provider"] for p in actual["active_backend_providers"]}==ref_backends
    assert all(sha(str(p))==all_pins[str(p)] for p in ref_backends.values())
    health=read(directory/"health.json")
    assert health["child_returncode"]==0 and health["status"]=="exited"
    for b in actual["blocks"]:
        c=b["case_index"];frozen=frozen_cases[c]
        assert b["prefix_ids"]==frozen["prefix_ids"] and b["input_tokens"]==frozen["used_T1_ids"]
        assert b["positions_begin"]==len(frozen["prefix_ids"]) and b["status"]==[0]*64
        out=directory/b["file"];orig_out=ref_dir/b["file"]
        if spec["stage"] in ("quality","observe"):
            assert out.stat().st_size==orig_out.stat().st_size==65*129280*4 and sha(out)==sha(orig_out)
            # Finite verification of every float in the original full-output artifact.
            # The compared candidate files are byte-identical.
            if b["case_id"] not in quality_digests:
                vals=array.array("f")
                with orig_out.open("rb") as f:vals.fromfile(f,65*129280)
                assert vals.itemsize==4 and all(map(math.isfinite,vals))
                quality_digests[b["case_id"]]=dict(sha256=sha(orig_out),rows=65,elements=len(vals))
            prev=b["prefix_stats_before_T1"]
            assert b["prefix_stats_before_prefill"][2:5]==prev[2:5]
            change=[]
            for ci,cur in enumerate(b["per_call_prefix_stats"],1):
                assert cur[2]-prev[2]==cur[3]-prev[3]==37 and cur[4]==0 and cur[5]==27648 and cur[6]==37
                if cur[1]!=prev[1]:change.append(ci)
                prev=cur
            regenerations[(name,c)]=change
        else:
            timing[c].append([e-s for e,s in zip(b["output_ready_ns"],b["start_ns"])])
    receipts[name]=dict(run_sha256=sha(directory/"RUN.json"),launch_sha256=sha(directory/"LAUNCH.json"),
        base=wanted["ggml_backend_sched_graph_compute_async"],active_backends=ref_backends)

descriptive={}
for c in (0,1):
    # First two T1 calls of each case. Python positions 256 and 257 are
    # calls 17 and 18, the recorded rebuild and its successor in quality runs.
    selected=[1,2]+([17,18] if c==1 else [])
    a0,b1,b2,a3=timing[c]
    delta=[Fraction(y+z-x-w,2) for x,y,z,w in zip(a0,b1,b2,a3)]
    total=sum(delta)/64
    contrib=sum(delta[i-1] for i in selected)/64
    descriptive[str(c)]=dict(order=["A0","B1","B2","A3"],
        full_inclusive_64_mean_BminusA_ms=float(total/1_000_000),
        focus_calls=selected,
        process_call_ms={str(i):[v[i-1]/1_000_000 for v in (a0,b1,b2,a3)] for i in selected},
        contribution_to_inclusive_64_mean_ms={str(i):float(delta[i-1]/(64*1_000_000)) for i in selected},
        focus_calls_total_contribution_ms=float(contrib/1_000_000),
        other_calls_total_contribution_ms=float((total-contrib)/1_000_000))
result=dict(pinned_backing_files=all_pins,verified_processes=receipts,
    verified_F32_per_case=quality_digests,finite_elements_per_full_set=sum(q["elements"] for q in quality_digests.values()),
    candidate_quality_graph_rebuild_calls={str(k):v for k,v in regenerations.items()},
    inclusive_descriptive_timing=descriptive,
    scope="All 64 samples and all four process means enter the decision. Timing calls contain no CUPTI. Quality counters identify graph generations; launch counts come from the separate observed first four calls.")
with (here/"TRIAL_EVIDENCE_AUDIT.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
print(json.dumps(descriptive,indent=2))
