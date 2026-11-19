"""Rebuild an exact base control and isolated scheduler candidate; never load either."""
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
lab = here.parent.parent
receipt_file = lab / "gpu-stream-20261103/SCHEDULER.json"
recipe = json.loads(receipt_file.read_text())
cwd = lab / "deepseek41-bringup-20261101/build/ggml/src"
original = lab / "gpu-stream-20261103/ggml-backend.cpp"
old_object = original.with_suffix(".cpp.o")
deployed = Path(recipe["library"])
frozen_hash = "2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()

assert not (here/"BUILD.json").exists(), "Preserve existing build before explicit revision"
assert sha(original) == recipe["source_sha256"] and sha(deployed) == recipe["library_sha256"] == frozen_hash
subprocess.run(["python3", str(here/"prepare.py")], cwd=here, check=True)
report = dict(original_recipe={"path":str(receipt_file),"sha256":sha(receipt_file)},
              original_base={"path":str(deployed),"sha256":frozen_hash}, cwd=str(cwd),
              source_sha256={str(p):sha(p) for p in (original,here/"prepare.py",here/"build.py",here/"ggml-backend.cpp",
                                                      here/"shared_prefix.inc",here/"shared-prefix.patch")},
              original_link_inputs={str((cwd/arg).resolve()):sha(cwd/arg) for arg in recipe["link"] if arg.endswith(".o")},
              original_object_sha256=sha(old_object), arms={})
for arm, source in (("control", original), ("candidate", here/"ggml-backend.cpp")):
    out = here / arm
    out.mkdir(exist_ok=True)
    obj = out / "ggml-backend.cpp.o"
    library = out / "libggml-base.so.0.28.0"
    compile_cmd = list(recipe["compile"])
    compile_cmd[compile_cmd.index(str(original))] = str(source)
    compile_cmd[compile_cmd.index("-o")+1] = str(obj)
    link = [str(obj) if v == str(old_object) else (f"-Wl,--dependency-file={out/'link.d'}" if v.startswith("-Wl,--dependency-file=") else v)
            for v in recipe["link"]]
    link[link.index("-o")+1] = str(library)
    commands=[compile_cmd,link]
    result=[]
    for label, cmd in zip(("compile","link"),commands):
        with (out/(label+".log")).open("wb") as log:
            proc = subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        result.append(proc.returncode)
        if proc.returncode: break
    report["arms"][arm] = dict(commands=commands, result=result, object_sha256=sha(obj) if obj.exists() else None)
    if result == [0,0]:
        d=report["arms"][arm]
        d.update(library=str(library),library_sha256=sha(library))
        if arm=="control":
            d["bit_identical_deployed"] = d["library_sha256"] == frozen_hash
            d["bit_identical_object"] = d["object_sha256"] == sha(old_object)
        else:
            syms=subprocess.check_output(["nm","-D","--defined-only",str(library)],text=True)
            d["stats_exported"]=any(line.split()[-1]=="xeon_shared_prefix_stats_v1" for line in syms.splitlines() if line.split())
    (here/"BUILD.json").write_text(json.dumps(report,indent=2)+"\n")
    if result != [0,0]:
        raise SystemExit(result[-1])
    if arm=="control":
        assert d["bit_identical_deployed"] and d["bit_identical_object"]
    else:
        assert d["stats_exported"]
