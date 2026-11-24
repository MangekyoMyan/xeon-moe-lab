"""Offline evidence check: recreate file layout from deduplicated F32 blobs in a temp directory."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def safe(rel):
    p=PurePosixPath(rel)
    return not p.is_absolute() and ".." not in p.parts and bool(p.parts)


def check(root):
    files=read(root/"MANIFEST_SHA256.json")["files"];names=set()
    for e in files:
        rel=e["path"];assert safe(rel) and rel not in names;names.add(rel)
        p=root/rel
        assert p.is_file() and p.stat().st_size==e["bytes"] and sha(p)==e["sha256"],rel
    blobs=read(root/"F32_INDEX.json")["captures"]
    with tempfile.TemporaryDirectory() as tmp:
        tree=Path(tmp)
        for rel in names:
            dst=tree/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(root/rel)
        for rel,entry in blobs.items():
            assert safe(rel) and rel not in names and entry["object"] in names and entry["object"]=="objects/"+entry["sha256"]+".f32"
            obj=root/entry["object"]
            assert obj.stat().st_size==entry["bytes"]
            dst=tree/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(obj)
        control=read(tree/"runs/control/VERIFIED.json");trace=read(tree/"runs/observe/VERIFIED.json")
        assert control["output_sha256"]==trace["output_sha256"] and trace["compared_byte_for_byte_to_matched_control"]
        for arm in ("control","observe"):
            run=tree/"runs"/arm
            assert read(run/"RUN.json")["passed"]
            for filename,digest in read(run/"VERIFIED.json")["output_sha256"].items():assert sha(run/filename)==digest
        for key in ("TIMELINE","OPERATOR","VALIDATION","CONTINUED"):
            assert read(tree/("WINDOW_"+key+".json"))["restore_required"] is False
        scripts=["analyze_timeline.py","analyze_operator.py","analyze_whole.py","analyze_prefill.py","analyze_edges.py","summary_data.py"]
        for name in scripts:
            subprocess.run([sys.executable,"-B",str(root/name),str(tree),"--check"],stdout=subprocess.DEVNULL,check=True)
        subprocess.run([sys.executable,"-B",str(root/"replay_cpu.py"),str(tree)],check=True)
        e=read(tree/"PREFIX_QUALITY.json")
        if e["ran"]:
            assert read(tree/"WINDOW_PREFIX_QUALITY.json")["restore_required"] is False
        final = read(tree/"FINAL_STATE.json")
        live = read(tree/"FINAL_API.json")
        assert live["API_health"]=="ok" and live["config_sha256"]=="5fd9aeb7ac83bffb7a250880cfe7e106824a450f39de9634f085eb85fdb6cf15"
        assert final["normal_pid"] == live["pid"] and final["production_prefill_thresholds"] == [64,64]
        assert final["normal_profile_at_start_and_end"] and len(final["processes"]) == 13
        assert all(entry["child_returncode"] == 0 for entry in final["processes"].values())
        config = read(tree/"PREFILL_CANDIDATE.json")
        assert config["normal_config_sha256"] == live["config_sha256"]
        assert [row["value"] for row in config["patch"]] == ["64","128","64","128"]
    print("Verified",len(files),"manifest entries,",len(blobs),"F32 file references; 8 traced T1, 296 routed layer windows, 576 local operator timings, 512 full-model timings and 56 native prompt measurements.")
    if e["ran"]:print("Verified extra boundary quality rows per policy:",e["total_rows_per_arm"])


if __name__=="__main__":check(Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve())
