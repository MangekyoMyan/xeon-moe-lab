"""Capture the normal server with read-only checks; stream action sends one fixed request."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OLD = LAB / "ecc9-simplification-20261119"
sys.path.insert(0, str(OLD))
from live import ENGINE, api, digest, read, snapshot

REFERENCE = ROOT / "intake/reference/current_runtime_633e.json"

def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

def capture_snapshot(label):
    prior = read(OLD / "FINAL_LIVE.json")
    current = snapshot(prior)
    pid = current["pid"]
    folder = ROOT / label
    assert not folder.exists()
    tool = ROOT / "intake/tools/capture_snapshot.py"
    assert digest(tool) == "47bb87b648f0bc50b288383525f580f181c181271b1b184b8ca2f92f03dcfbaf"
    assert digest(REFERENCE) == digest(LAB / "current/runtime.json")
    config = read(ENGINE / "server-config.json")
    log = read(ENGINE / "health.json")
    assert log["pid"] == pid
    current["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    current["os_release"] = Path("/etc/os-release").read_text()
    current["model_alias"] = config["model_alias"]
    current["monitor"] = {k:log.get(k) for k in ("status","limits","initial_fans")}
    current["hostname"] = os.uname().nodename
    save(ROOT / (label+"-live.json"), current)
    with (ROOT/(label+"-capture.log")).open("x") as log_file:
        result = subprocess.run([sys.executable,"-I",str(tool),"--profile",str(REFERENCE),
                                 "--lab-root",str(LAB),"--pid",str(pid),"--out",str(folder),
                                 "--label",label],stdout=log_file,stderr=subprocess.STDOUT,timeout=90)
    result_data = read(folder / "snapshot.json")
    print(json.dumps({"label":label,"pid":pid,"health":current["API_health"],"idle":current["slots_idle"],
                      "snapshot":result_data["summary"]},ensure_ascii=False))
    return result.returncode

def capture_stream():
    # Resolve the local recipient using the verified ordinary config and served model.
    expected=read(ROOT/"production-before-live.json")
    config=read(ENGINE/"server-config.json")
    assert digest(ENGINE/"server-config.json")==expected["config_sha256"]
    assert read(ENGINE/"health.json")["pid"]==expected["pid"]
    assert config["host"]=="10.10.10.2" and int(config["port"])==8000
    assert Path(config["api_key_file"]).resolve()==(LAB/"api-server-20261030/api-key.txt").resolve()
    route=json.loads(subprocess.check_output(["/usr/sbin/ip","-j","route","get",config["host"]],text=True))
    assert len(route)==1 and route[0].get("type")=="local" and route[0]["dev"]=="lo", "Direct benchmark recipient must be local"
    slots=api(config,"/slots")
    assert slots and all(slot.get("is_processing") is False for slot in slots)
    models=api(config,"/v1/models")
    served=[row["id"] for row in models["data"]]
    assert config["model_alias"] in served
    request=read(ROOT/"intake/requests/stream_ja.template.json")
    assert request["model"]=="__FILL_ACTUAL_SERVED_MODEL_ID_ONCE__" and request["n"]==1 and request["stream"] is True
    request["model"]=config["model_alias"]
    save(ROOT/"request.json",request)
    save(ROOT/"STREAM_CONTEXT.json",{"observation_point":"x9ram direct to its own ordinary OpenAI-compatible endpoint","started_utc":datetime.now(timezone.utc).isoformat(),
        "served_model":request["model"],"pid":expected["pid"],"slots_idle":True,"prefix_cache_state":"unknown","previous_use":"unknown","automatic_retry":False,
        "max_tokens":request["max_tokens"],"seed":request["seed"],"temperature":request["temperature"]})
    tool=ROOT/"intake/tools/capture_stream.py"
    assert digest(tool)=="1fd4259d44a9f5c96efc89ad810854fefdc04f22422e1fee115d02883503d08d"
    key_name="XEON_BENCH_STREAM_KEY"
    uid=LAB.stat().st_uid
    env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8",key_name:Path(config["api_key_file"]).read_text().strip()}
    args=[sys.executable,"-I",str(tool),"--url",f"http://{config['host']}:{config['port']}/v1/chat/completions","--request",str(ROOT/"request.json"),
          "--out",str(ROOT/"stream"),"--label","old-production-direct-01","--api-key-env",key_name]
    # The capture records a newly created, non-private benchmark prompt and its response only.
    options={"user":uid,"group":LAB.stat().st_gid,"extra_groups":[]} if os.geteuid()==0 else {}
    return subprocess.run(args,env=env,**options,timeout=340).returncode

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("before","after","stream"))
    args=parser.parse_args()
    sys.exit(capture_stream() if args.action=="stream" else capture_snapshot("production-"+args.action))
