"""Connect each actual linker object with its recorded compilation command."""
import json
import shlex
from pathlib import Path
from live import LAB, digest, read

ROOT=Path(__file__).resolve().parent
build=read(ROOT/"build/BUILD.json")
tree=LAB/"mxfp4-hybrid-20261102/hybrid/build-cpu-ivybridge"
cmake_db=tree/"compile_commands.json"
database={}
for entry in read(cmake_db):
    directory=Path(entry["directory"])
    argv=shlex.split(entry["command"])
    target=(directory/argv[argv.index("-o")+1]).resolve()
    assert target==(tree/entry["output"]).resolve()
    database[str(target)]={"cwd":str(directory),"argv":argv,"source":entry["file"],"recipe":str(cmake_db),"status":"existing CMake compile record"}
def add(argv,cwd,record,status):
    if "-c" not in argv or "-o" not in argv:return
    source=str((Path(cwd)/argv[argv.index("-c")+1]).resolve())
    obj=str((Path(cwd)/argv[argv.index("-o")+1]).resolve())
    database[obj]={"cwd":str(cwd),"argv":argv,"source":source,"recipe":str(record),"status":status}
for path in [LAB/"exploration-20261112/sparse-cache/CPU_BUILD.json", LAB/"exploration-20261112/mx-prepack/release-candidate-a/BUILD.json"]:
    data=read(path)
    commands=data.get("commands",[data.get("cpp",[])])
    for argv in commands:add(argv,path.parent,path,"existing custom compile record with absolute source and output")
for u in build["control_units"]+[build["ordinary_unit"]]:
    cmd=u["command"]
    add(cmd,Path(cmd[cmd.index("-o")+1]).parent,ROOT/"build/BUILD.json","recompiled in this study")
rows=[]
for arm in ["control","ordinary"]:
    for p,h in build[arm]["link_inputs_sha256"].items():
        if not p.endswith(".o"):continue
        assert p in database and digest(p)==h,p
        rec=dict(database[p])
        rec.update(arm=arm,object=p,object_sha256=h,source_bytes_sha256_now=digest(rec["source"]),record_sha256=digest(rec["recipe"]))
        rows.append(rec)
out={"scope":"Known backing objects, recorded compile commands and current source digests. Unchanged CMake objects were reused in the link. Only the rebuilt units have a fresh compilation receipt. Link cwd/argv and non-object libraries are in build/BUILD.json.","objects_by_arm":rows}
(ROOT/"CPU_OBJECT_PROVENANCE.json").write_text(json.dumps(out,indent=2)+"\n")
print("objects per link",len(rows)//2)
