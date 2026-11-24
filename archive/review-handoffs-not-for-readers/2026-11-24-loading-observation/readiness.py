"""Probe only the health resource of this PID's verified local listening socket."""
import datetime
import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
START=json.loads((ROOT/"WINDOW_START.json").read_text())
PID=START["model"]["pid"]
PROC=Path("/proc")/str(PID)


def ident():
    st=(PROC/"stat").read_text()
    return (int(st[st.rfind(")")+2:].split()[19])==START["model"]["start_ticks"] and
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()==START["boot"])


def probe():
    rec={"wall":time.time(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "monotonic":time.monotonic(),"pid":PID,"request_sent":False,"ready":None}
    if not PROC.exists() or not ident():
        return {**rec,"state":"original_identity_unavailable"}
    arg=(PROC/"cmdline").read_bytes().split(b"\0")
    host=arg[arg.index(b"--host")+1].decode()
    port=int(arg[arg.index(b"--port")+1])
    addr=ipaddress.ip_address(host)
    if addr.version!=4 or addr.is_unspecified:
        return {**rec,"state":"unverified_ipv4_binding"}
    local={a["local"] for interface in json.loads(subprocess.check_output(["ip","-j","-4","address","show"]))
           for a in interface.get("addr_info",[]) if "local" in a}
    owns=set()
    for file in (PROC/"fd").iterdir():
        try:
            link=os.readlink(file)
        except FileNotFoundError:
            continue
        if link.startswith("socket:["):
            owns.add(link[8:-1])
    listener=[]
    for line in (PROC/"net/tcp").read_text().splitlines()[1:]:
        fields=line.split()
        ip,hexport=fields[1].split(":")
        if (int(hexport,16)==port and fields[3]=="0A" and fields[9] in owns and
                socket.inet_ntoa(bytes.fromhex(ip)[::-1])==host):
            listener.append(int(fields[9]))
    rec.update(local_address_confirmed=host in local,listener_inode_owned_by_pid=listener,port=port)
    if host not in local or not listener:
        return {**rec,"state":"no_verified_listener_or_not_bound_yet"}
    con=http.client.HTTPConnection(host,port,timeout=2)
    try:
        rec["request_sent"]=True
        con.request("GET","/health",headers={"Connection":"close"})
        resp=con.getresponse()
        raw=resp.read(4097)
        rec["http_status"]=resp.status
        try:
            parsed=json.loads(raw) if len(raw)<=4096 else {}
        except (ValueError,UnicodeError):
            parsed={}
        status=parsed.get("status") if isinstance(parsed,dict) else None
        err=parsed.get("error") if isinstance(parsed,dict) else None
        # Never persist free-form HTTP responses or request data.
        loading=(isinstance(err,dict) and str(err.get("message","")).lower()=="loading model")
        rec["status_field"]=status if status in ("ok","loading model","loading") else None
        rec["loading_error_recognized"]=loading
        if (resp.status,status)==(200,"ok"):
            rec.update(state="ready",ready=True)
        elif resp.status==503 and (loading or status in ("loading model","loading")):
            rec.update(state="loading",ready=False)
        else:
            rec["state"]="unclassified_health_response"
    except OSError as exc:
        rec["state"]="health_request_unavailable"
        rec["error_type"]=type(exc).__name__
    finally:
        con.close()
    if not PROC.exists() or not ident():
        rec.update(ready=None,state="identity_changed_during_health_check")
    return rec


if __name__=="__main__":
    if sys.argv[1] not in ("early","end","followup"):
        raise SystemExit("use a timepoint label")
    destination=ROOT/("READINESS_"+sys.argv[1].upper()+".json")
    with destination.open("x") as f:
        value=probe()
        json.dump(value,f,indent=2)
    print(value)
