"""Read-only identity check for this machine's normal API."""
import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

LAB = Path("/home/codex/projects/llm-lab")
ENGINE = LAB / "deepseek41-bringup-20261101"
PREVIOUS = LAB / "169a-exploration-20261119/FINAL_RUNTIME_RECHECK.json"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def api(config, route, payload=None):
    url = f"http://{config['host']}:{config['port']}{route}"
    headers = {"Authorization": "Bearer " + Path(config["api_key_file"]).read_text().strip()}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as response:
        return json.load(response)

def snapshot(expected, smoke=False):
    config_file = ENGINE / "server-config.json"
    config = read(config_file)
    assert digest(config_file) == expected["config_sha256"], "Unexpected normal configuration"
    pid = read(ENGINE / "health.json")["pid"]
    proc = Path("/proc") / str(pid)
    cmd = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    start = config["command"].index(str(ENGINE / "build/bin/llama-server"))
    assert cmd == config["command"][start:], "API command differs from the profile"
    env = dict(v.split("=", 1) for v in (proc / "environ").read_bytes().decode().split("\0") if "=" in v)
    assert all(env.get(k) == v for k, v in config["environment"].items()), "API environment differs from the profile"
    switches = {k: env.get(k) for k in ("GGML_CPU_IQ_ROW_LOCAL", "GGML_CPU_IQ_AUDIT_DIR", "LD_PRELOAD", "XEON_169_SHARED_PREFIX")}
    assert all(v is None for v in switches.values()), "Experimental options present"
    rows = [line.split(None,5) for line in (proc / "maps").read_text().splitlines()]
    paths = {r[5] for r in rows if len(r)==6 and r[5].startswith("/")}
    targets = expected["mapped_runtime_hashes"]
    names = {Path(p).name for p in targets}
    actual = {p for p in paths if Path(p.removesuffix(" (deleted)")).name in names}
    assert actual == set(targets), "Active inference library set differs from expected"
    assert all("libtrace169" not in p for p in paths)
    for path, h in targets.items():
        st = os.stat(path)
        assert digest(path) == h
        assert all(int(row[4]) == st.st_ino and tuple(int(v,16) for v in row[3].split(":")) == (os.major(st.st_dev),os.minor(st.st_dev)) for row in rows if len(row)==6 and row[5]==path)
    model_dir = Path(config["model"]).parent.parent
    model_receipt = model_dir / "download-verified.json"
    recorded = read(model_receipt)
    for row in recorded["files"]:
        file = model_dir / row["name"]
        assert file.stat().st_size == row["size"]
    plan = Path(env["GGML_CPU_IQ_PLAN"])
    assert digest(plan) == "cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6"
    health = api(config, "/health")
    assert health.get("status") == "ok"
    slots = api(config,"/slots")
    assert slots
    services = {u:subprocess.check_output(["systemctl","is-active",u],text=True).strip() for u in
                ("llm-deepseek41-api.service","xeon-watch.service","xeon-watch-notify.service")}
    mon = read("/var/lib/xeon-watch/latest.json")
    assert 0 <= time.time()-mon["unix"] <=45 and not mon["errors"] and not mon["active_alerts"]
    external = sorted(p for p in paths if Path(p).name.startswith(("libcuda","libcublas","libgomp","libstdc++","libggml","libllama","libmtmd","libc.so.")))
    result = dict(unix=time.time(),pid=pid,API_health="ok",slots_idle=all(not x["is_processing"] for x in slots),
                  config_sha256=expected["config_sha256"],mapped_runtime_hashes=targets,backing_inodes_verified=True,
                  command_and_configured_environment_match=True,experimental_environment=switches,ld_library_path=env["LD_LIBRARY_PATH"].split(":"),
                  dependencies_mapped=external,model_download_manifest_sha256=digest(model_receipt),model_revision=recorded["revision"],
                  mapped_gguf_files=sorted(p for p in paths if p.endswith(".gguf")),plan_sha256=digest(plan),services=services,
                  monitor_errors=mon["errors"],active_alerts=mon["active_alerts"],monitor_timestamp=mon["unix"],
                  CPU_model=next(l.split(":",1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")),
                  CPU_socket_core_node=subprocess.check_output(["lscpu","-p=CPU,SOCKET,CORE,NODE"],text=True).splitlines(),
                  gpu_watts_limit=mon["gpu"]["power_limit_W"],gpu_ecc=mon["gpu"]["ecc_mode"])
    if smoke and result["slots_idle"]:
        j=api(config,"/v1/chat/completions",dict(model=config["model_alias"],messages=[dict(role="user",content="Reply with only one digit: what is 2 + 3?")],temperature=0,max_tokens=8,stream=False))
        answer=j["choices"][0]["message"]["content"].strip()
        assert answer=="5", "API smoke result"
        result["openai_smoke"]={"expected":"5","received":answer}
    assert read(ENGINE/"health.json")["pid"] == pid and digest(config_file) == expected["config_sha256"]
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--expected",type=Path,default=PREVIOUS)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    data=snapshot(read(args.expected),args.smoke)
    with args.output.open("x") as stream:
        json.dump(data,stream,indent=2)
        stream.write("\n")
    print(json.dumps({k:data[k] for k in ("pid","CPU_model","API_health","slots_idle","experimental_environment","config_sha256")}))
