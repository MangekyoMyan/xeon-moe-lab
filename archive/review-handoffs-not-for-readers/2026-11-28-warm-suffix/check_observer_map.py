"""Verify the diagnostic DSO against the actual mapping of a ready trial."""
import argparse
import json
import os
from pathlib import Path
from analyze import R, read, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("period",choices=["B2","A2"])
    run = R/"runs"/p.parse_args().period
    ready = read(run/"READY.json")
    proc = Path("/proc")/str(ready["pid"])
    assert ready["observer"] and proc.joinpath("comm").read_text().strip() == "llama-server"
    def tick():
        return int(proc.joinpath("stat").read_text().rsplit(")",1)[1].split()[19])
    assert tick() == ready["start_ticks"]
    lib = R/"observer/observe.so"
    st = lib.stat()
    rows = [r.split(maxsplit=5) for r in proc.joinpath("maps").read_text().splitlines()]
    matched = [r for r in rows if len(r)==6 and r[5]==str(lib)]
    assert matched and any("x" in r[1] for r in matched)
    for r in matched:
        major,minor = [int(s,16) for s in r[3].split(":")]
        assert os.makedev(major,minor)==st.st_dev and int(r[4])==st.st_ino
    h = sha(lib)
    assert h == read(R/"DESIGN.json")["observer_sha256"] and tick()==ready["start_ticks"]
    rec = {"pid":ready["pid"],"start_ticks":tick(),"mapped_dev":st.st_dev,"inode":st.st_ino,
           "executable_mapping_verified":True,"filename":lib.name,"bytes":st.st_size,"sha256":h}
    (run/"OBSERVER_MAP.json").write_text(json.dumps(rec,indent=2)+"\n")
    print("Observer mapping verified",run.name,rec["pid"],h[:12])


if __name__ == "__main__":
    main()
