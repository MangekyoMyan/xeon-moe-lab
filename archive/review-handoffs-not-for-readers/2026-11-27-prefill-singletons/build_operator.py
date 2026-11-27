from build_capture import HERE,LAB,read,save,sha
import subprocess

D=HERE/"operator";src=LAB/"deepseek41-bringup-20261101/source";bin=src.parent/"build/bin"
cuda=LAB/"v100-prep-20261024/cuda-12.9.2"
cupti=LAB/"strata-tc-study-20261031/cupti/cuda_cupti-linux-x86_64-12.9.79-archive"
runtime=read(LAB/"current/runtime.json")
dirs=[LAB/runtime["libraries"][k]["file"] for k in ("cpu","cuda","llama","base")]
libdirs=[p.parent for p in dirs]+[bin,cuda/"lib",cupti/"lib"]
include=[src/"include",src/"ggml/include",src/"ggml/src",src/"src",src/"vendor/nlohmann",
         LAB/"exploration-20261112/row-local",cuda/"include",cupti/"include"]
cmd1=[str(cuda/"bin/nvcc"),"-O2","-std=c++17","-arch=sm_70","-Xcompiler","-fPIC","-c",str(D/"gather.cu"),"-o",str(D/"gather.o")]
cmd2=["g++","-std=c++17","-O2","-g","-march=ivybridge","-mno-avx2","-mno-fma","-ffp-contract=off","-fno-fast-math","-pthread","-fopenmp",
      *["-I"+str(p) for p in include],str(D/"replay.cpp"),str(D/"gather.o"),*["-L"+str(p) for p in libdirs],
      "-Wl,-rpath,"+":".join(map(str,libdirs)),"-Wl,--no-as-needed","-lllama","-lggml-cpu","-lggml-cuda","-lggml-base","-lggml",
      "-lcudart","-lcupti","-ldl","-o",str(D/"replay")]
if __name__=="__main__":
    with (D/"build.log").open("w") as f:
        for cmd in (cmd1,cmd2):subprocess.run(cmd,stdout=f,stderr=f,check=True)
    save(D/"BUILD.json",{"commands":[cmd1,cmd2],"source_sha256":{p.name:sha(p) for p in D.iterdir() if p.suffix in (".cpp",".h",".cu")},
         "binary_sha256":sha(D/"replay"),"libraries":{k:{"path":v["file"],"sha256":v["sha256"]} for k,v in runtime["libraries"].items()}})
