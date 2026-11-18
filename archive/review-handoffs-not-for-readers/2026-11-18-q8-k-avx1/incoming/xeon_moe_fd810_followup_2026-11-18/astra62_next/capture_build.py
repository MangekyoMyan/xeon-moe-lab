#!/usr/bin/env python3
"""Record sources, compilation/linkage, loaded providers and representative ELF evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("build",type=Path)
p.add_argument("out",type=Path)
a=p.parse_args()
run=a.build.resolve()
out=a.out.resolve()
out.mkdir(parents=True,exist_ok=True)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def output(*args): return subprocess.check_output(args,text=True)
inputs=json.loads((here/"inputs.json").read_text())
protocol=json.loads((here/"protocol.json").read_text())
files=[r["target"] for r in inputs["archived_files"]]+["ggml/src/ggml-cpu/arch/x86/quants.c","ggml/src/ggml-quants.c","CMakeLists.txt"]
manifest=dict(system=platform.platform(),compiler=output("gcc","--version").splitlines()[0],cpu_threads_visible=len(os.sched_getaffinity(0)),
              model=next(l.split(":",1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),
              cpus_allowed=sorted(os.sched_getaffinity(0)),
              memory_nodes_allowed=next(l.split(":",1)[1].strip() for l in Path("/proc/self/status").read_text().splitlines() if l.startswith("Mems_allowed_list")),
              runtime_env={"OMP_DYNAMIC":"FALSE","OMP_PROC_BIND":"FALSE","GGML_CPU_IQ_ROW_LOCAL":"0","GGML_CPU_TILED_MM":"unset","GGML_CPU_TILED_MM_FORCE":"unset",
                           "LD_PRELOAD":"count_paths.so only for numerical runs; unset for timing"},
              probe_sha256=digest(run/"graph_probe"),counter_sha256=digest(run/"count_paths.so"),
              protocol_sha256=digest(here/"protocol.json"),run_trials_sha256=digest(here/"run_trials.py"),variants={})
for v in protocol["variants"]:
    src=run/("src-"+v)
    build=run/("build-"+v)
    cp=(run/("lib-"+v)/"libggml-cpu.so.0").resolve(strict=True)
    bp=(run/("lib-"+v)/"libggml-base.so.0").resolve(strict=True)
    env=dict(os.environ,LD_LIBRARY_PATH=str(run/("lib-"+v)),TEST_EXPERT_MASK="64",GGML_CPU_IQ_ROW_LOCAL="0",OMP_DYNAMIC="FALSE",OMP_PROC_BIND="FALSE")
    env.pop("LD_PRELOAD",None)
    tiny=subprocess.run(["taskset","-c","2",str(run/"graph_probe"),"bench","256","1","1","1","1","1","1007"],
                         env=env,capture_output=True,text=True,check=True)
    maps={Path(x.split(",",1)[1]).resolve(strict=True) for x in tiny.stderr.splitlines() if x.startswith("map,")}
    if maps!={cp,bp}: raise RuntimeError(tiny.stderr)
    (out/("maps-"+v+".txt")).write_text(tiny.stderr)
    asm=output("objdump","-d","--disassemble=quantize_row_q8_K",cp)
    if "quantize_row_q8_K_ref@plt>" not in asm: raise RuntimeError("expected reference wrapper")
    (out/("q8-wrapper-"+v+".txt")).write_text(asm)
    symbols=[x for x in output("nm","-D",cp).splitlines() if any(k in x for k in ("xeon_iq_input","xeon_iq3_cache_add_selected","xeon_iq3_cache_lookup_experts","xeon_mxfp4_rhs","quantize_row_q8_K"))]
    (out/("symbols-"+v+".txt")).write_text("\n".join(symbols)+"\n")
    selected=[]
    for entry in json.loads((build/"compile_commands.json").read_text()):
        f=Path(entry["file"])
        if any(str(f).endswith(suffix) for suffix in ("/ggml-quants.c","/ggml-cpu/arch/x86/quants.c","/ggml-cpu/ggml-cpu.c","/xeon-local/iq3-cache.cpp","/xeon-local/mx-rhs.cpp")):
            argv=shlex.split(entry["command"])
            obj=Path(entry["directory"])/argv[argv.index("-o")+1]
            selected.append(dict(source=str(f.relative_to(src)),source_sha256=digest(f),object=entry["output"],object_sha256=digest(obj),command=entry["command"]))
    assert len(selected)==5
    (out/("compile-"+v+".json")).write_text(json.dumps(selected,indent=2)+"\n")
    link=build/"ggml/src/CMakeFiles/ggml-cpu.dir/link.txt"
    shutil.copyfile(link,out/("link-"+v+".txt"))
    shutil.copyfile(run/"raw"/("build-"+v+".log"),out/("build-"+v+".log"))
    shutil.copyfile(run/"raw"/("sources-"+v+".log"),out/("sources-"+v+".log"))
    src_files=list(files)
    if v!="current": src_files.append("xeon-local/folded-input.inc")
    manifest["variants"][v]=dict(cpu_elf_sha256=digest(cp),cpu_loaded_path=str(cp),base_elf_sha256=digest(bp),base_loaded_path=str(bp),
        patch_sha256={patch:digest(here/patch) for patch in (["08-current-mixed.patch"] if v in ("08","09") else [])+(["09-selected-input-only.patch"] if v=="09" else [])},
        compiled_source_sha256={f:digest(src/f) for f in src_files},
        maps="maps-"+v+".txt",compiled_objects="compile-"+v+".json",link="link-"+v+".txt",q8_wrapper="q8-wrapper-"+v+".txt",symbols="symbols-"+v+".txt")
assert len({x["cpu_elf_sha256"] for x in manifest["variants"].values()})==3
assert len({x["base_elf_sha256"] for x in manifest["variants"].values()})==1
(out/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
