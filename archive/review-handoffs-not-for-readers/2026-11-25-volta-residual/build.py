"""Build host-side diagnostics against the deployed CUDA backend and freeze the input hashes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
PRIOR=LAB/"worker-balance-20261124"
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+"\n")


def main():
    cfg=LAB/"deepseek41-bringup-20261101/server-config.json"
    profile=read(LAB/"current/runtime.json")
    assert sha(cfg)==profile["service_config"]["sha256"]==read(PRIOR/"FINAL_API.json")["config_sha256"]
    mapped=read(PRIOR/"FINAL_API.json")["mapped_runtime_hashes"]
    libs={k:(LAB/v["file"]).resolve() for k,v in profile["libraries"].items()}
    for k,p in libs.items():assert sha(p)==profile["libraries"][k]["sha256"]==mapped[str(p)]
    engine=LAB/"deepseek41-bringup-20261101";toolkit=LAB/"v100-prep-20261024/cuda-12.9.2"
    source=engine/"source"
    capture=PRIOR/"runs/precision-64-64"
    refs={}
    for name in ("JA96","PY96"):
        leaf=capture/(name+"-early")
        assert read(leaf/"CAPTURE.json")["prefix_logits_byte_equal_to_no_callback"]
        for p in [leaf/"CAPTURE.json",*(leaf/("ffn_moe_"+role+"-0"+suffix) for role in ("gate","up") for suffix in (".bin","-ids.bin","-input.bin"))]:
            refs[str(p)]=sha(p)
    original=read(PRIOR/"precision/WEIGHT_SOURCE.json");shards={v["shard"] for v in original.values()};assert len(shards)==1
    spec={"normal_config_sha256":sha(cfg),"cuda_library":str(libs["cuda"]),"cpu_library":str(libs["cpu"]),
          "base_library":str(libs["base"]),"mapped_runtime":mapped,"capture_root":str(capture),"shard":shards.pop(),
          "frozen_captures":refs,"target":"real L0 gate/up IQ2_S, two 96-token prompts, natural expert occupancy",
          "sample":"Six pre-existing positions per prompt, six expert slots, 32 original rows per projection.",
          "timing":"Three GPU-resident paths in all six orders per expert. Output-ready wall time includes residual preparation, stream ordering, native matmul(s), scale and addition.",
          "arms":["ordinary native n-column dot","residual computed on GPU, two native n-column dots","native 2n-column dot, fold both halves; residual matches 2n dispatch"]}
    save(ROOT/"DESIGN.json",spec)
    nvcc=[str(toolkit/"bin/nvcc"),"-std=c++17","-O3","-use_fast_math","-Wno-deprecated-gpu-targets",
          "--generate-code=arch=compute_70,code=sm_70","-Xcompiler=-fPIC","-c",str(ROOT/"residual.cu"),"-o",str(ROOT/"residual.o")]
    env=read(cfg)["environment"]
    paths=env["LD_LIBRARY_PATH"].split(":")
    link=["g++","-std=c++17","-O2","-ffp-contract=off","-march=ivybridge",
          "-I"+str(source/"ggml/include"),"-I"+str(source/"ggml/src"),"-I"+str(source/"vendor/nlohmann"),
          "-I"+str(toolkit/"include"),str(ROOT/"probe.cpp"),str(ROOT/"residual.o"),
          *["-L"+d for d in paths],"-Wl,-rpath,"+env["LD_LIBRARY_PATH"],"-Wl,-rpath-link,"+env["LD_LIBRARY_PATH"],
          "-lggml-cuda","-lggml-cpu","-lggml-base","-lggml","-lcudart","-ldl","-o",str(ROOT/"probe")]
    with (ROOT/"build.log").open("w") as f:
        for cmd in (nvcc,link):subprocess.run(cmd,stdout=f,stderr=f,timeout=180,check=True)
    save(ROOT/"BUILD.json",{"sources":{p:sha(ROOT/p) for p in ("residual.cu","probe.cpp")},
                             "commands":[nvcc,link],"probe_sha256":sha(ROOT/"probe"),"spec_sha256":sha(ROOT/"DESIGN.json")})
    print("built",sha(ROOT/"probe"))

if __name__=="__main__":main()
