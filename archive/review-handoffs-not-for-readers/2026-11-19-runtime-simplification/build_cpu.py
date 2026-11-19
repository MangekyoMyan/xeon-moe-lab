"""Rebuild the pinned two-TU control and replace only the cache TU in the comparison."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

STUDY = Path(__file__).resolve().parent
LAB = STUDY.parent
HIST = LAB / "be05-followup-20261119/cpu08"
ORIGINAL = LAB / "exploration-20261112/mx-prepack/release-candidate-a/libggml-cpu.so.0.28.0"
CPU_SHA = "33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def run(command, cwd, log):
    with log.open("w") as stream:
        subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)

def compile_unit(unit, output, source=None):
    command = list(unit["argv"])
    if source:
        old = Path(command[command.index("-c")+1])
        command[command.index("-c")+1] = str(source)
        command.insert(1, "-I"+str(source.parent))
        flag = next(v for v in command if v.startswith("-ffile-prefix-map="+str(old)+"="))
        command.append("-ffile-prefix-map="+str(source)+"="+flag.split("=",2)[2])
    command[command.index("-o")+1] = str(output)
    deps=output.with_suffix(".d")
    command.extend(["-MMD","-MF",str(deps)])
    run(command,output.parent,output.with_suffix(".build.log"))
    dependency_names=deps.read_text().replace("\\\n"," ").split(":",1)[1].split()
    return {"command":command,"sha256":digest(output),"dependencies":{str(Path(p).resolve()):digest(p) for p in dependency_names}}

def link(control, folder, cpu_obj, cache_obj):
    folder.mkdir(exist_ok=True,parents=True)
    library=folder / ORIGINAL.name
    original_folder=HIST/"control"
    command=list(control["link"])
    for old,new in [(original_folder/"cpu.o",cpu_obj),(original_folder/"cache.o",cache_obj)]:
        assert command.count(str(old))==1
        command[command.index(str(old))]=str(new)
    command[command.index("-o")+1]=str(library)
    command=[f"-Wl,--dependency-file={folder/'link.d'}" if v.startswith("-Wl,--dependency-file=") else v for v in command]
    cwd=Path(control["link_cwd"])
    run(command,cwd,folder/"link.log")
    inputs={}
    for i,arg in enumerate(command):
        if i==command.index("-o")+1 or arg.startswith("-"):continue
        p=(cwd/arg).resolve()
        if p.is_file() and arg.endswith((".o",".a",".so",".0.28.0")):inputs[str(p)]=digest(p)
    for alias in ["libggml-cpu.so.0","libggml-cpu.so"]:(folder/alias).symlink_to(library.name)
    return {"file":str(library),"sha256":digest(library),"link_command":command,"cwd":str(cwd),"link_inputs_sha256":inputs}

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    receipt=HIST/"BUILD.json"
    control=json.loads(receipt.read_text())["variants"]["control"]
    assert digest(ORIGINAL)==CPU_SHA==control["CPU_sha256"]
    for p,h in control["link_inputs"].items():assert digest(p)==h,p
    for n,h in control["sources"].items():assert digest(HIST/"control/source"/n)==h,n
    units=control["commands"]
    cpu=out/"control-cpu.o"
    cache=out/"control-cache.o"
    baseline_units=[compile_unit(u,o) for u,o in zip(units,[cpu,cache],strict=True)]
    for u,build in zip(units,baseline_units,strict=True):
        assert build["sha256"]==u["original_sha256"]==digest(u["original_object"])
    original=link(control,out/"control",cpu,cache)
    assert original["sha256"]==CPU_SHA
    source=STUDY/"current/cpu/iq3-cache.cpp"
    candidate_obj=out/"ordinary-cache.o"
    clean_unit=compile_unit(units[1],candidate_obj,source)
    assert not any(Path(p).name=="placement.inc" for p in clean_unit["dependencies"])
    ordinary=link(control,out/"ordinary",cpu,candidate_obj)
    assert ordinary["sha256"]!=CPU_SHA
    opts=(b"GGML_CPU_IQ_ROW_LOCAL",b"GGML_CPU_IQ_AUDIT_DIR")
    for receipt_for_arm,required in [(original,True),(ordinary,False)]:
        elf=Path(receipt_for_arm["file"]).read_bytes()
        assert all((v in elf)==required for v in opts)
        syms=subprocess.check_output(["nm","-D","--defined-only",receipt_for_arm["file"]],text=True)
        receipt_for_arm["cache_api"]=sorted(l.split()[-1] for l in syms.splitlines() if l.split() and l.split()[-1].startswith("xeon_iq"))
    assert original["cache_api"]==ordinary["cache_api"]
    test_source=out/"restore-experiment";shutil.copytree(source.parent,test_source)
    subprocess.run(["git","apply","--check",str(STUDY/"experiments/row-local/enable.patch")],cwd=test_source,check=True)
    subprocess.run(["git","apply",str(STUDY/"experiments/row-local/enable.patch")],cwd=test_source,check=True)
    for name in ["iq3-cache.cpp","placement.inc"]:
        assert digest(test_source/name)==digest(HIST/"control/source/xeon-local"/name)
    result={"original_cpu":str(ORIGINAL),"baseline_recipe_sha256":digest(receipt),"original_recipe":str(receipt),
            "control":original,"ordinary":ordinary,"control_units":baseline_units,"ordinary_unit":clean_unit,
            "compilers":{u["argv"][0]:digest(u["argv"][0]) for u in units},
            "row_local_patch_roundtrip":True,"control_object_and_library_exact":True}
    (out/"BUILD.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"control":original["sha256"],"ordinary":ordinary["sha256"],"same_cache_ABI":True,"patch_roundtrip":True}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=STUDY/"build")
    main(ap.parse_args().output.resolve())
