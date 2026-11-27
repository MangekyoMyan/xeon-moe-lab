import hashlib
import json
from pathlib import Path
import subprocess

R=Path(__file__).resolve().parent
LAB=R.parent
D=R/"observer"
src=LAB/"deepseek41-bringup-20261101/source"
def read(p):return json.loads(Path(p).read_bytes())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

if __name__=="__main__":
    cmd=["/usr/bin/g++","-std=c++17","-O2","-march=ivybridge","-fPIC","-shared","-Wall","-Wextra",
         *["-I"+str(src/p) for p in ("include","src","ggml/include","ggml/src","vendor/nlohmann")],
         str(D/"observe.cpp"),"-ldl","-o",str(D/"observe.so")]
    with (D/"build.log").open("w") as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
    save(D/"BUILD.json",{"command":cmd,"source_sha256":sha(D/"observe.cpp"),"binary_sha256":sha(D/"observe.so")})
