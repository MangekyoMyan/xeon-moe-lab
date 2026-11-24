"""Decode existing quality-only perf records, without starting another measurement."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def run(name):
    d=ROOT/"trial/runs"/name
    assert name in ("new-p12-quality-01","new-d24-b12-quality-01")
    assert json.loads((d/"SPEC.json").read_text())["stage"]=="quality"
    h=d/"quality-host"; record=h/"worker.local.data"
    result=h/"PC_DECODE.json"
    if result.exists():
        raise ValueError(f"Existing decoded result in {h}")
    command=["/usr/bin/perf","script","--ns","-i",str(record),"-F","pid,tid,cpu,time,ip,sym,dso"]
    with (h/"pcs.local.txt").open("xb") as text, (h/"script.stderr").open("xb") as err:
        subprocess.run(command,stdout=text,stderr=err,check=True,timeout=120)
    subprocess.run([sys.executable,str(ROOT/"host_evidence.py"),str(d),"--parse",str(h/"pcs.local.txt")],check=True,timeout=120)
    parts=[record,h/"pcs.local.txt"]
    local={p.name:{"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest()} for p in parts}
    (h/"LOCAL_INPUTS.json").write_text(json.dumps({"command":command,"kept_on_host":local},indent=2)+"\n")
    counts=json.loads(result.read_text())
    assert counts["unparsed_rows"]==0 and counts["kept"]>0


if __name__=="__main__":
    for item in sys.argv[1:]:
        run(item)
