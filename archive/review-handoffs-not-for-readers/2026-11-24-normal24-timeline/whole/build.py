"""One changed TU in a frozen diagnostic library; compile the matched fixed-input caller."""
from pathlib import Path
import hashlib
import json
import subprocess

D=Path(__file__).resolve().parent
LAB=D.parent.parent
OLD=LAB/"parallel-pilot-20261117/integrated"
DRIVER=LAB/"preupgrade-baseline-20261123/trial"
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(Path(p).read_bytes())


def build():
    assert not (D/"BUILD.json").exists()
    old=read(OLD/"ENGINE_BUILD.json"); objects={}
    assert sha(OLD/"runtime/libllama.so.0.8.0")==old["libllama_sha256"]
    for name,h in old["new_objects"].items():
        assert sha(name)==h
    for name,h in old["reused_objects"].items():
        assert sha(name)==h
    recipes=old["revision_rebuild_commands"]
    cc=next(x for x in recipes if "-c" in x["argv"] and any(a==str(OLD/"hot-backend.cpp") for a in x["argv"]))
    link=next(x for x in reversed(recipes) if "-c" not in x["argv"])
    commands=[]
    def invoke(cmd,cwd,label):
        commands.append({"argv":cmd,"cwd":str(cwd)})
        with (D/(label+".log")).open("x") as out:
            subprocess.run(cmd,cwd=cwd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
    obj=D/"hot-backend.cpp.o"
    cmd=cc["argv"].copy();cmd[cmd.index("-c")+1]=str(D/"hot-backend.cpp");cmd[cmd.index("-o")+1]=str(obj)
    cmd.insert(1,"-I"+str(D))
    if "-MF" in cmd:cmd[cmd.index("-MF")+1]=str(D/"hot-backend.d")
    invoke(cmd,cc["cwd"],"compile")
    lib=D/"runtime/libllama.so.0.8.0";cmd=link["argv"].copy();cmd[cmd.index("-o")+1]=str(lib)
    cmd[cmd.index(str(OLD/"hot-backend.cpp.o"))]=str(obj)
    for i,v in enumerate(cmd):
        if v.startswith("-Wl,--dependency-file="):cmd[i]="-Wl,--dependency-file="+str(D/"link.d")
        if v.startswith("-Wl,-Map="):cmd[i]="-Wl,-Map="+str(D/"link.map")
        if v.endswith(".o"):
            path=Path(v);path=path if path.is_absolute() else Path(link["cwd"])/path
            objects[str(path)]=sha(path)
    invoke(cmd,link["cwd"],"link")
    oldlibs=read(LAB/"parallel-pilot-20261117/replay/BUILD.json")["engine_libraries"]["diagnostic"]
    libs={}
    for name,e in oldlibs.items():
        src=lib if name.startswith("libllama.") else Path(e["path"])
        if src!=lib:assert sha(src)==e["sha256"]
        stem=name.split(".so")[0]
        for alias in {stem+".so",stem+".so.0", name}:
            dst=D/"runtime"/alias
            if dst!=src:dst.symlink_to(src if src!=lib else lib.name)
        libs[name]={"path":str(src),"sha256":sha(src)}
    cmd=read(DRIVER/"BUILD.json")["command"].copy()
    for i,v in enumerate(cmd):
        if str(DRIVER) in v:cmd[i]=v.replace(str(DRIVER),str(D))
    invoke(cmd,D,"caller")
    srcs=["driver.cpp","hot-backend.cpp","hot-backend.h","physical_team.inc","affinity_check.h","cache_inspection.h","CASES.json"]
    (D/"BUILD.json").write_text(json.dumps({"returncode":0,"commands":commands,"reused_engine_object_hashes":objects,
        "historical_engine_receipt":sha(OLD/"ENGINE_BUILD.json"),"source_sha256":{n:sha(D/n) for n in srcs},
        "libraries":libs,"driver_sha256":sha(D/"driver")},indent=2)+"\n")
    print("whole-model diagnostic built",sha(lib),sha(D/"driver"))

if __name__=="__main__":build()
