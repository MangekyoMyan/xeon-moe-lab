"""Recheck the published build receipts, logits, clock arithmetic and NUMA records."""
import argparse
import array
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from analyze import calculate

def read(path):
    return json.loads(path.read_text())

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def verify(root):
    # The archive uses a renamed guard status, and omits private per-run config files.
    manifest=read(root/"FILES_SHA256.json")
    for p,entry in manifest.items():
        file=root/p
        assert file.stat().st_size==entry["bytes"] and digest(file)==entry["sha256"],p
    design=read(root/"trial/DESIGN.json")
    golden=read(root/"GOLDEN_OUTPUTS.json")
    plan=read(root/"PLAN_MEMBERSHIP.json")
    build=read(root/"build/BUILD.json")
    assert build["control_object_and_library_exact"] and build["row_local_patch_roundtrip"]
    assert build["control"]["sha256"]==read(root/"INITIAL_RUNTIME.json")["mapped_runtime_hashes"][build["original_cpu"]]
    assert build["ordinary"]["cache_api"]==build["control"]["cache_api"]
    corpus=read(root/"trial/CASES.json")["cases"]
    for stage in design["stages"]:
        directory=root/"trial/runs"/stage["name"]
        spec=read(directory/"SPEC.json")
        assert spec==stage
        rows=65 if stage["stage"]=="quality" else 1
        result=read(directory/"RUN.json")
        guard=read(directory/"guard-health.json")
        cfg=read(directory/"LAUNCH.json")
        pin=read(directory/"PROVIDER_EXPECTED.json")
        assert result["passed"] and guard["child_returncode"]==0 and guard["status"]=="exited"
        assert result["providers"]==pin["providers"] and result["active_backend_providers"]==pin["active_backend_providers"]
        assert cfg["libraries"]["libggml-cpu.so.0.28.0"]["sha256"]==build[stage["arm"]]["sha256"]
        assert result["compact_bytes_after_unload"]==0 and result["compact_bytes"]==127401984000
        assert [p["routed_host"] for p in result["placement"]]==[True]*37+[False]*3
        if rows==65:
            reg=result["registration"]
            assert {k:v for k,v in reg["experts"].items() if v}==plan and reg["sampled_vma_policies"]=={"interleave:0-1":1440}
        for name,h in read(directory/"RESULTS_SHA256.json").items():
            if name=="server-config.json":
                data={"command":cfg["command"],"environment":cfg["environment"]}
                actual=hashlib.sha256((json.dumps(data,indent=2)+"\n").encode()).hexdigest()
            else:actual=digest(directory/("guard-health.json" if name=="health.json" else name))
            assert actual==h,(stage["name"],name)
        for block,ci in zip(result["blocks"],stage["cases"],strict=True):
            assert block["status"]==[0]*64 and block["prefix_ids"]==corpus[ci]["prefix_ids"] and block["input_tokens"]==corpus[ci]["used_T1_ids"]
            path=directory/block["file"]
            assert path.stat().st_size==rows*129280*4
            raw=path.read_bytes()
            values=array.array("f")
            values.frombytes(raw)
            if sys.byteorder!="little":values.byteswap()
            assert all(map(math.isfinite,values))
            assert hashlib.sha256(raw).hexdigest()==golden[block["file"]]["full" if rows==65 else "last"]
    recomputed=calculate(root)
    assert recomputed==read(root/"ANALYSIS.json")
    for name,rec in read(root/"compact-check/RESULT.json")["arms"].items():
        assert digest(root/"compact-check"/(name+".f32"))==rec["raw_sha256"]
    assert (root/"compact-check/control.f32").read_bytes()==(root/"compact-check/ordinary.f32").read_bytes()
    return {"passed":True,"files_checked":len(manifest),"model_processes":len(design["stages"]),
            "timed_calls":recomputed["timed_T1_calls"],"raw_quality_rows_per_prompt":65,"model_executed":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    print(json.dumps(verify(p.parse_args().archive)))
