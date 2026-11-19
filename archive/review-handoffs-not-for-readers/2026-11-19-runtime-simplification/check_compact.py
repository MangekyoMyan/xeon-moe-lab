"""Verify cache hit, miss and release on synthetic weights with both CPU libraries."""
import subprocess
import json
import os
from pathlib import Path
from live import digest, read, LAB

ROOT=Path(__file__).resolve().parent
STUDY=LAB/"exploration-20261112/sparse-cache"
BUILD=ROOT/"build"
OUTPUT=ROOT/"compact-check"

def main():
    OUTPUT.mkdir()
    cmd=read(STUDY/"TEST_BUILD.json")
    src=ROOT/"astra-review/compact_quality.cpp"
    cmd[cmd.index(str(STUDY/"test-sparse.cpp"))]=str(src)
    exe=OUTPUT/"compact_quality"
    cmd[cmd.index("-o")+1]=str(exe)
    cmd.insert(cmd.index("-o"),"-ldl")
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    base=next(Path(p) for p in read(ROOT/"INITIAL_RUNTIME.json")["mapped_runtime_hashes"] if "/libggml-base." in p)
    wrapper=LAB/"moe-implementation-20261026/release/without_thp.py"
    report={"source_sha256":digest(src),"command":cmd,"executable_sha256":digest(exe),"arms":{}}
    for arm in ["control","ordinary"]:
        cpu=Path(read(BUILD/"BUILD.json")[arm]["file"])
        env={k:v for k,v in os.environ.items() if not k.startswith(("LLAMA_","GGML_","OMP_","GOMP_","LD_","CUDA_"))}
        env.update(LD_LIBRARY_PATH=":".join(str(p) for p in [cpu.parent,base.parent,LAB/"deepseek41-bringup-20261101/build/bin"]),LD_BIND_NOW="1",OMP_DYNAMIC="FALSE",OMP_THREAD_LIMIT="12")
        f32=OUTPUT/(arm+".f32")
        args=["/usr/bin/python3",str(wrapper),"numactl","--physcpubind=0-11","--interleave=0,1",str(exe),str(cpu),str(base),str(f32)]
        with (OUTPUT/(arm+".log")).open("w") as out,(OUTPUT/(arm+".err")).open("w") as err:
            subprocess.run(args,env=env,stdout=out,stderr=err,check=True,timeout=120)
        rows=[json.loads(v) for v in (OUTPUT/(arm+".log")).read_text().splitlines()]
        assert rows[-1]["pass"] and f32.stat().st_size==414720
        report["arms"][arm]={"cpu_sha256":digest(cpu),"base_sha256":digest(base),"raw_sha256":digest(f32),"command":args,"checks":rows}
    assert (OUTPUT/"control.f32").read_bytes()==(OUTPUT/"ordinary.f32").read_bytes()
    report["outputs_byte_equal"]=True
    (OUTPUT/"RESULT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"compact_test_passed":True,"finite_floats_per_arm":103680,"numa_pages":report["arms"]["ordinary"]["checks"][0]["resident_pages_by_node"]}))
if __name__=="__main__":
    main()
