"""Install the measured 24/24 profile on the normal service, retaining its rollback bytes."""
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
ENGINE=LAB/"deepseek41-bringup-20261101"
REVIEW=LAB/"results-review-20261124"
PREVIOUS=LAB/"postload-measurement-20261124"
CONFIG=ENGINE/"server-config.json"
CANDIDATE=PREVIOUS/"profiles/normal-24-24.json"
BACKUP=ROOT/"private/normal-12-24.json"
UNIT="llm-deepseek41-api.service"
HOST,PORT="10.10.10.2",8000
KEY=LAB/"api-server-20261030/api-key.txt"
OLD="963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43"
NEW="5fd9aeb7ac83bffb7a250880cfe7e106824a450f39de9634f085eb85fdb6cf15"
CHECKER="f2f1ab36d3af16ca12e2a27e8e992c99d75fa6d9e62a40d8c91ade2cc88e3450"
WATCH=Path("/var/lib/xeon-watch/latest.json")
BOOT=Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def read(p):return json.loads(p.read_bytes())
def digest(b):return hashlib.sha256(b).hexdigest()


def save(name,obj):
    dest=ROOT/name
    with dest.open("x") as f:
        json.dump(obj,f,indent=2)
        f.write("\n")


def event(phase,**data):
    with (ROOT/"events.jsonl").open("a") as f:
        f.write(json.dumps(dict(unix=time.time(),mono=time.monotonic(),phase=phase,**data))+"\n")
        f.flush();os.fsync(f.fileno())
    print(phase,flush=True)


def unit():
    out=subprocess.check_output(["systemctl","show",UNIT,"-p","ActiveState,MainPID,InvocationID"],text=True)
    return dict(l.split("=",1) for l in out.splitlines())


def monitor():
    for name in ("xeon-watch.service","xeon-watch-notify.service"):
        if subprocess.run(["systemctl","is-active","--quiet",name]).returncode:
            raise RuntimeError(f"monitor inactive: {name}")
    if any(p.exists() for p in (ENGINE/"STOP_REQUESTED",LAB/"moe-engine-20261029/STOP_REQUESTED")):
        raise RuntimeError("hardware stop marker present")
    w=read(WATCH);g=w["gpu"];b=w["bmc"]
    assert w["boot_id"]==BOOT and 0<=time.time()-w["unix"]<=45 and 0<=time.time()-b["observed_unix"]<=75
    assert not w["active_alerts"] and not w["errors"] and not b["bad_sensor_status"]
    assert g["power_limit_W"]==150 and g["ecc_mode"]=="Enabled" and g["ecc_double"]==0
    assert b["fan_mode"]=="00" and max(b["fan_percent"])<=70
    return w


def snapshot_health():
    w=monitor()
    return {"unix":w["unix"],"boot":w["boot_id"],"age_s":time.time()-w["unix"],
            "cpu_package_C":{k:v for k,v in w["cpu"].items() if "Package" in k},
            "host_counters":w["host_counters"],"gpu":w["gpu"],"dimm_peak_C":max(w["bmc"]["dimms_C"].values()),
            "fan_percent":w["bmc"]["fan_percent"],"alerts":w["active_alerts"],"errors":w["errors"]}


def process(pid):
    path=Path("/proc")/str(pid)
    data=(path/"stat").read_text()
    v=data[data.rfind(")")+2:].split()
    return {"pid":pid,"ppid":int(v[1]),"start_ticks":int(v[19]),"comm":data[data.find("(")+1:data.rfind(")")]}


def listener(pid):
    p=Path("/proc")/str(pid)
    handles=set()
    for fd in (p/"fd").iterdir():
        try:link=os.readlink(fd)
        except FileNotFoundError:continue
        if link.startswith("socket:["):handles.add(link[8:-1])
    for line in (p/"net/tcp").read_text().splitlines()[1:]:
        fields=line.split()
        ip,port=fields[1].split(":")
        if (int(port,16)==PORT and socket.inet_ntoa(bytes.fromhex(ip)[::-1])==HOST and
                fields[3]=="0A" and fields[9] in handles):
            return int(fields[9])
    return None


def get(path,auth=False,plain=False):
    con=http.client.HTTPConnection(HOST,PORT,timeout=5)
    try:
        headers={"Authorization":"Bearer "+KEY.read_text().strip()} if auth else {}
        con.request("GET",path,headers=headers)
        resp=con.getresponse();raw=resp.read(4*1024**2+1)
        assert len(raw)<=4*1024**2
        return resp.status,raw.decode() if plain else json.loads(raw)
    finally:con.close()


def identity(expected):
    u=unit(); assert u["ActiveState"]=="active" and int(u["MainPID"])>0
    h=read(ENGINE/"health.json"); assert h["status"]=="running" and h["boot_id"]==BOOT and not h.get("error")
    pid=h["pid"];p=Path("/proc")/str(pid);model=process(pid);assert model["ppid"]==int(u["MainPID"])
    raw=CONFIG.read_bytes();assert digest(raw)==expected
    cfg=json.loads(raw);cmd=cfg["command"];exe=str(ENGINE/"build/bin/llama-server");i=cmd.index(exe)
    assert (p/"exe").resolve()==Path(exe) and (p/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")==cmd[i:]
    assert cfg["host"]==HOST and cfg["port"]==PORT and Path(cfg["api_key_file"]).resolve()==KEY
    ino=listener(pid);assert ino
    return {**model,"boot":BOOT,"supervisor":int(u["MainPID"]),"invocation":u["InvocationID"],"listen_inode":ino,
        "host":HOST,"port":PORT,"decode":int(cmd[cmd.index("-t")+1]),"prefill":int(cmd[cmd.index("-tb")+1]),
        "decode_mask":cmd[cmd.index("--cpu-mask")+1],"prefill_mask":cmd[cmd.index("--cpu-mask-batch")+1],
        "guard_start_unix":h["started_unix"]}


def idle(expected):
    deadline=time.monotonic()+7200; streak=0; observed_busy=False
    while time.monotonic()<deadline:
        monitor(); who=identity(expected)
        code,slots=get("/slots",True);assert code==200 and slots
        code,metrics=get("/metrics",True,True);assert code==200
        counts={k:float(v) for k,v in re.findall(r"(?m)^llamacpp:(requests_processing|requests_deferred)\s+([0-9.eE+-]+)\s*$",metrics)}
        assert set(counts)=={"requests_processing","requests_deferred"}
        empty=all(s.get("is_processing") is False for s in slots) and not any(counts.values())
        streak=streak+1 if empty else 0
        observed_busy|=not empty
        if streak==2:
            return who,{"all_slots_idle":True,"queue":counts,"observed_busy_while_waiting":observed_busy,"unix":time.time()}
        time.sleep(1 if empty else 3)
    raise TimeoutError("API has outstanding requests")


def backup(raw,st):
    fd=os.open(BACKUP,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    assert digest(BACKUP.read_bytes())==OLD
    save("BACKUP.json",{"private_path":str(BACKUP),"sha256":OLD,"uid":st.st_uid,"gid":st.st_gid,
        "file_mode":stat.S_IMODE(st.st_mode),"source_inode":st.st_ino,"source_device":st.st_dev})


def replace_config(expected,new_bytes,original):
    assert not CONFIG.is_symlink() and stat.S_ISREG(CONFIG.stat().st_mode)
    assert digest(CONFIG.read_bytes())==expected
    new_hash=digest(new_bytes);assert {new_hash,expected}=={OLD,NEW}
    fd,n=tempfile.mkstemp(dir=CONFIG.parent,prefix=".server-config-profile-")
    temp=Path(n)
    try:
        with os.fdopen(fd,"wb") as out:
            out.write(new_bytes);out.flush()
            os.fchown(out.fileno(),original.st_uid,original.st_gid)
            os.fchmod(out.fileno(),stat.S_IMODE(original.st_mode));os.fsync(out.fileno())
        assert digest(CONFIG.read_bytes())==expected
        os.replace(temp,CONFIG)
        directory=os.open(CONFIG.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temp.unlink(missing_ok=True)
    st=CONFIG.stat()
    assert digest(CONFIG.read_bytes())==new_hash and (st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode))==(original.st_uid,original.st_gid,stat.S_IMODE(original.st_mode))


def released(old_pid=None):
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        w=monitor();disappeared=True
        if old_pid:
            try:disappeared=process(old_pid["pid"])["start_ticks"]!=old_pid["start_ticks"]
            except (FileNotFoundError,ProcessLookupError):pass
        if (unit()["ActiveState"] in ("inactive","failed") and disappeared and w["gpu"]["used_MiB"]<=128):
            return
        time.sleep(2)
    raise TimeoutError("previous service resources not yet released")


def launch(expected):
    assert digest(CONFIG.read_bytes())==expected
    monitor()
    subprocess.run(["/usr/bin/python3",str(ENGINE/"start.py")],check=True,timeout=90)
    deadline=time.monotonic()+7200;last=None
    while time.monotonic()<deadline:
        monitor();assert unit()["ActiveState"] in ("active","activating"),"supervisor stopped during startup"
        try:
            who=identity(expected);code,body=get("/health")
        except (OSError,AssertionError,ValueError,KeyError):
            who=None;code=None;body={}
        if code!=last:
            event("startup HTTP state",code=code);last=code
        if code==200 and body.get("status")=="ok":return who
        time.sleep(5)
    raise TimeoutError("ready deadline; keep existing live loader for inspection")


def verify(which,label):
    tool=REVIEW/"tools/check_live_identity.py"
    assert digest(tool.read_bytes())==CHECKER
    subprocess.run([sys.executable,"-I",str(tool),str(ROOT/label),"--expected",
                    str(REVIEW/f"reference/EXPECTED_NORMAL_{which}.json")],check=True,timeout=90)
    rec=read(ROOT/label)
    assert rec["gpu_watts_limit"]==150 and rec["gpu_ecc"]=="Enabled" and rec["backing_inodes_verified"]
    return rec


def apply():
    assert os.geteuid()==0 and __debug__
    before=read(ROOT/"BEFORE.json");old_bytes=CONFIG.read_bytes();candidate=CANDIDATE.read_bytes()
    assert digest(old_bytes)==OLD==before["config_sha256"] and digest(candidate)==NEW
    cfg=json.loads(old_bytes);cmd=cfg["command"];assert cmd[8:10]==["-t","12"] and cmd.count("-t")==1
    cmd[9]="24";assert cfg==json.loads(candidate)
    assert cmd[cmd.index("-tb")+1]=="24" and cmd[cmd.index("--cpu-mask")+1]==cmd[cmd.index("--cpu-mask-batch")+1]=="0xffffff"
    route=json.loads(subprocess.check_output(["/usr/sbin/ip","-j","route","get",HOST]))
    assert len(route)==1 and route[0]["dev"]=="lo" and route[0]["type"]=="local"
    original=CONFIG.stat()
    who,free=idle(OLD);assert who["pid"]==before["pid"] and who["decode"]==12 and who["prefill"]==24
    save("HEALTH_BEFORE.json",snapshot_health())
    backup(old_bytes,original)
    save("BEFORE_IDENTITY.json",{**who,"idle":free,"unix":time.time()})
    touched=False;retaining_loader=False
    try:
        who,last=idle(OLD);assert who["pid"]==before["pid"] and digest(CONFIG.read_bytes())==OLD
        event("stop idle original service",pid=who["pid"],idle=last);touched=True
        subprocess.run(["systemctl","stop",UNIT],check=True,timeout=110)
        released(who);event("original service released")
        replace_config(OLD,candidate,original);event("installed candidate atomically",sha256=NEW)
        try:after=launch(NEW)
        except TimeoutError:
            if unit()["ActiveState"]=="active":retaining_loader=True
            raise
        assert after["decode"]==after["prefill"]==24
        checked=verify("24_24","AFTER.json");assert after==identity(NEW) and checked["pid"]==after["pid"]
        save("AFTER_IDENTITY.json",{**after,"unix":time.time(),"http_ready":True})
        event("ordinary 24/24 ready and backing verified",pid=after["pid"])
    except BaseException as failure:
        event("cutover needs attention",error=repr(failure),attempted_stop=touched)
        if not touched or retaining_loader:raise
        # Roll back a software/start failure only while the original protections
        # are healthy and the current config is still one of the two known hashes.
        monitor();current=digest(CONFIG.read_bytes());assert current in (OLD,NEW)
        if unit()["ActiveState"]=="active":
            if current==OLD:raise
            # A successfully started service may already be serving other clients.
            # In that case drain before rollback rather than disrupting their request.
            idle(NEW)
            subprocess.run(["systemctl","stop",UNIT],check=True,timeout=110)
        released()
        if current==NEW:replace_config(NEW,BACKUP.read_bytes(),original)
        restored=launch(OLD);verify("12_24","ROLLBACK_VERIFIED.json")
        event("rolled back to original profile",pid=restored["pid"])
        raise


if __name__=="__main__":
    def interrupted(sig,frame):
        raise KeyboardInterrupt(f"signal {sig}")
    for sig in (signal.SIGHUP,signal.SIGTERM):signal.signal(sig,interrupted)
    with (ROOT/".cutover.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apply()
