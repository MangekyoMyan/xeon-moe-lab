"""Verify the stand-alone handoff on Linux using only the Python standard library."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def verify(root):
    files=read(root/"MANIFEST_SHA256.json")["files"]
    for item in files:
        p=root/item["path"]
        assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],item["path"]
    assert read(root/"trial/STATE.json")["completed"]
    assert read(root/"API_WINDOW_POSTSWAP.json")["restore_required"] is False
    assert read(root/"RESTORED_POSTSWAP.json")["normal_API_restored"]
    assert read(root/"FINAL_API.json")["API_health"]=="ok"
    case_dirs=sorted((root/"trial/runs").iterdir()); assert len(case_dirs)==6
    for d in case_dirs:
        for name,digest in read(d/"RESULTS_SHA256.json").items():
            p=d/("guard-health.json" if name=="health.json" else name)
            assert sha(p)==digest,(d.name,name)
    old=root/"reference/OLD_P12"
    def check(name,*args):
        subprocess.run([sys.executable,"-B",str(root/name),*[str(a) for a in args]],check=True)
    check("replay_results.py",root,old,"--check")
    check("host_summary.py",root,old,"--check")
    check("analyze_stream_tasks.py",root,root/"reference/old_normal_stream","--check")
    for tag,folder in (("OLD",root/"reference/old_normal_stream"),("NEW",root/"stream"),("PILOT",root/"pilot24/stream")):
        if not folder.is_dir():continue
        got=json.loads(subprocess.check_output([sys.executable,"-I",str(root/"tools/replay_stream.py"),str(folder)]))
        assert got==read(root/("SSE_REPLAY_"+tag+".json"))
    fixed=read(root/"FIXED_COMPARISON.json")
    for name in ("NEW_P12","NEW_D24_B12"):
        check2=read(root/("COMPARE_"+name+".json"))
        assert check2["status"]=="exact_on_saved_outputs"
        for case,meta in fixed["conditions"][name]["cases"].items():
            other=check2["candidate_timing"]["cases"][case]
            assert abs(other["T1_per_second"]-meta["T1_s"])<1e-10
    if (root/"PRACTICAL_COMPARISON.json").exists():
        check("pilot_result.py",root,"--check")
        assert read(root/"API_WINDOW_PILOT24.json")["restore_required"] is False
    print("Verified",len(files),"manifest payloads, all six actual stages and their restoration evidence")


if __name__=="__main__":
    verify(Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve())
