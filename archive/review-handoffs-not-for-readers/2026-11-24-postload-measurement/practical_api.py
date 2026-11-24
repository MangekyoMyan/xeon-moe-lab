"""One conditional 24/24 text-API confirmation, locally bound and automatically restored."""
import argparse
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

from api_window import active, clear_monitor, window
from stream_tasks import capture, stat

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
ENGINE=LAB/"deepseek41-bringup-20261101"
PILOT=ROOT/"pilot24"
SCRIPT=Path(__file__).resolve()
SOURCE=ENGINE/"server-config.json"
HOST="127.0.0.1"
PORT=8000
KEY=LAB/"api-server-20261030/api-key.txt"
DRIVER_CFG_SHA="963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43"
CLIENT_SHA="1fd4259d44a9f5c96efc89ad810854fefdc04f22422e1fee115d02883503d08d"
REQUEST_SHA="8c7dd777627326870076b7b4ec19fca9db085f84bfd461a2321a78d3363bfc39"


def read(p):
    return json.loads(p.read_bytes())


def save(p,data):
    with p.open("x") as f:
        json.dump(data,f,indent=2)
        f.write("\n")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def config():
    assert sha(SOURCE)==DRIVER_CFG_SHA
    c=read(SOURCE);args=c["command"]
    for flag,old,new in (("-t","12","24"),("--host","10.10.10.2",HOST)):
        assert args.count(flag)==1 and args[args.index(flag)+1]==old
        args[args.index(flag)+1]=new
    assert c["host"]=="10.10.10.2" and c["port"]==PORT and Path(c["api_key_file"]).resolve()==KEY
    assert args[args.index("-tb")+1]=="24" and args[args.index("--cpu-mask")+1]==args[args.index("--cpu-mask-batch")+1]=="0xffffff"
    c["host"]=HOST
    return c


def child_identity(supervisor,pid,c):
    p=Path("/proc")/str(pid)
    s=(p/"stat").read_text().rsplit(")",1)[1].split()
    assert int(s[1])==supervisor
    exe=str(ENGINE/"build/bin/llama-server");args=c["command"]
    assert (p/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")==args[args.index(exe):]
    return int(s[19])


def owns_listener(pid):
    proc=Path("/proc")/str(pid)
    inodes=set()
    for fd in (proc/"fd").iterdir():
        try:link=os.readlink(fd)
        except FileNotFoundError:continue
        if link.startswith("socket:["):
            inodes.add(link[8:-1])
    for row in (proc/"net/tcp").read_text().splitlines()[1:]:
        fields=row.split();address,p=fields[1].split(":")
        if fields[3]=="0A" and int(p,16)==PORT and socket.inet_ntoa(bytes.fromhex(address)[::-1])==HOST and fields[9] in inodes:
            return int(fields[9])
    return None


def get(route,auth=False):
    con=http.client.HTTPConnection(HOST,PORT,timeout=4)
    try:
        headers={"Authorization":"Bearer "+KEY.read_text().strip()} if auth else {}
        con.request("GET",route,headers=headers);resp=con.getresponse()
        return resp.status,json.loads(resp.read(1024**2))
    finally:
        con.close()


def wait_ready(proc,c):
    deadline=time.monotonic()+7200
    first=time.time()
    while proc.poll() is None and time.monotonic()<deadline:
        clear_monitor(ROOT)
        if (PILOT/"health.json").is_file():
            h=read(PILOT/"health.json");pid=h.get("pid")
            if pid and h["status"]=="running":
                # The supervisor publishes the child's PID immediately after
                # Popen. The Python/numactl launch wrappers exec the server next.
                exe=Path("/proc")/str(pid)/"exe"
                if not exe.exists() or exe.resolve()!=ENGINE/"build/bin/llama-server":
                    time.sleep(1)
                    continue
                start=child_identity(proc.pid,pid,c);ino=owns_listener(pid)
                if ino:
                    try:status,body=get("/health")
                    except (OSError,ValueError):status,body=0,{}
                    if status==200 and body.get("status")=="ok":
                        code,slots=get("/slots",True)
                        assert code==200 and slots and all(s.get("is_processing") is False for s in slots)
                        code,models=get("/v1/models",True)
                        assert code==200 and c["model_alias"] in {m["id"] for m in models["data"]}
                        return dict(pid=pid,start_ticks=start,supervisor=proc.pid,socket_inode=ino,ready_unix=time.time(),
                                    wait_started_unix=first,host=HOST,port=PORT,slots_idle=True,alias=c["model_alias"])
        time.sleep(4)
    raise RuntimeError(f"Candidate did not reach ready; supervisor return={proc.poll()}")


def verify_backing(pid,expected):
    matches={}
    expected_names={Path(k).name for k in expected}
    for line in (Path("/proc")/str(pid)/"maps").read_text().splitlines():
        v=line.split(None,5)
        if len(v)!=6 or Path(v[5]).name not in expected_names:continue
        assert v[5] in expected, "Different runtime mapped with the same soname"
        path=Path(v[5]);st=path.stat()
        assert st.st_dev==os.makedev(*[int(n,16) for n in v[3].split(":")]) and st.st_ino==int(v[4])
        if v[5] not in matches:
            assert sha(path)==expected[v[5]]
            matches[v[5]]={"dev":st.st_dev,"inode":st.st_ino,"sha256":expected[v[5]]}
    assert set(matches)==set(expected)
    return matches


def serve():
    assert read(PILOT/"server-config.json")==config() and not active("llm-deepseek41-api.service")
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    source=ENGINE/"supervise.py"
    assert sha(source)==read(ROOT/"trial/runs/new-p12-quality-01/LAUNCH.json")["guard_source_sha256"][str(source)]
    sys.path.insert(0,str(ENGINE))
    spec=importlib.util.spec_from_file_location("practical_native_guard",source)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.ROOT=PILOT
    old=guard.check_host
    def check(initial):
        cpu,avail=old(initial)
        if avail<64*1024**2:
            raise RuntimeError("Candidate API RAM reserve below 64 GiB")
        return cpu,avail
    guard.check_host=check
    return guard.main()


def send():
    ready=read(PILOT/"READY.json")
    assert child_identity(ready["supervisor"],ready["pid"],config())==ready["start_ticks"]
    assert owns_listener(ready["pid"])==ready["socket_inode"]
    request=ROOT/"request.json";client=ROOT/"intake/tools/capture_stream.py"
    assert sha(client)==CLIENT_SHA and sha(request)==REQUEST_SHA
    cmd=[sys.executable,"-I",str(client),"--url",f"http://{HOST}:{PORT}/v1/chat/completions",
         "--request",str(request),"--out",str(PILOT/"stream"),"--label","pilot-normal-24-24","--api-key-env","XEON_BENCH_STREAM_KEY"]
    env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","XEON_BENCH_STREAM_KEY":KEY.read_text().strip()}
    return subprocess.run(cmd,env=env,user=LAB.stat().st_uid,group=LAB.stat().st_gid,extra_groups=[],timeout=340).returncode


def justified():
    fixed=read(ROOT/"FIXED_COMPARISON.json")["conditions"]
    for key,p in fixed["NEW_P12"]["cases"].items():
        d=fixed["NEW_D24_B12"]["cases"][key]
        if min(d["per_process_T1_s"])<=max(p["per_process_T1_s"]) or d["p95_ms"]>=p["p95_ms"]:
            return False
    return True


def run():
    assert read(ROOT/"trial/STATE.json")["completed"] and read(ROOT/"RESTORED_POSTSWAP.json")["normal_API_restored"]
    baseline=read(ROOT/"production-after-live.json")
    assert baseline["API_health"]=="ok" and baseline["config_sha256"]==DRIVER_CFG_SHA
    assert active("llm-deepseek41-api.service") and baseline["pid"]==read(ENGINE/"health.json")["pid"]
    assert read(ROOT/"HOST_AND_THERMALS.json")["watch_warnings_or_errors"]==0
    if not justified():
        print("No extra API trial selected");return
    c=config()
    with (ROOT/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clear_monitor(ROOT)
        PILOT.mkdir()
        os.chown(PILOT,LAB.stat().st_uid,LAB.stat().st_gid)
        save(PILOT/"server-config.json",c)
        (PILOT/"server-config.json").chmod(0o600)
        before=read(ENGINE/"health.json")
        save(PILOT/"BASELINE.json",{"normal_pid":baseline["pid"],"normal_start_unix":before["started_unix"],"config_sha256":DRIVER_CFG_SHA,
            "effective_differences":{"-t":["12","24"],"--host":["10.10.10.2",HOST]},"projector_initialization_argument_from_server_source":24})
        def interrupted(sig,frame):
            raise KeyboardInterrupt(f"Pilot interrupted with signal {sig}")
        for sig in (signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,interrupted)
        with window(ROOT,"PILOT24"):
            assert not active("llm-deepseek41-api.service") and not active("llm-openai-api.service")
            mem={k:int(v.split()[0]) for k,v in (s.split(":",1) for s in Path("/proc/meminfo").read_text().splitlines())}
            assert mem["MemAvailable"]>=360*1024**2 and mem["Mlocked"]<=32*1024**2
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((HOST,PORT));s.listen(1)
            with (PILOT/"supervisor.out").open("x") as log:
                proc=subprocess.Popen([sys.executable,str(SCRIPT),"serve"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    ready=wait_ready(proc,c)
                    environ=(Path("/proc")/str(ready["pid"])/"environ").read_bytes().split(b"\0")
                    assert all((k+"="+v).encode() in environ for k,v in c["environment"].items())
                    ready["configured_environment_matches"]=True
                    ready["backing"]=verify_backing(ready["pid"],baseline["mapped_runtime_hashes"])
                    save(PILOT/"READY.json",ready)
                    capture(ready["pid"],ready["start_ticks"],PILOT/"normal-scheduling",[sys.executable,str(SCRIPT),"send"])
                    assert child_identity(proc.pid,ready["pid"],c)==ready["start_ticks"]
                    summary=read(PILOT/"stream/summary.json")
                    assert summary["status"]=="complete" and sha(PILOT/"stream/request.effective.json")==REQUEST_SHA
                    print("Candidate API measurement completed",flush=True)
                finally:
                    if proc.poll() is None:proc.terminate()
                    code=proc.wait(timeout=90)
                    save(PILOT/"STOP.json",{"supervisor_exit":code,"final_guard_state":read(PILOT/"health.json")["status"] if (PILOT/"health.json").exists() else None})
                    assert code==0


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=("run","serve","send"));action=ap.parse_args().action
    if os.geteuid()!=0 or not __debug__:raise SystemExit("Requires the existing root guard and enabled checks")
    if action=="serve":raise SystemExit(serve())
    if action=="send":raise SystemExit(send())
    run()
