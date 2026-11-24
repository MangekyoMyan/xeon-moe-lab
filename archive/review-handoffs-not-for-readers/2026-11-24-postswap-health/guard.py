"""Bounded observation for one requested health phase; no fan or service writes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from host import API, BOOT, CPU, ERROR, EV, ROOT, WATCH, command, counters, get, now, save

LOCK = ROOT.parent / "results/glm-air-q8-20261020/.experiment.lock"


def real_state():
    active = command(["systemctl","show",API,"-p","ActiveState","--value"]).strip()
    workers = command(["ps","-C","llama-server,llama-cli,llama-bench,ollama,vllm","-o","pid=,comm="],allow=(0,1)).strip()
    monitors = [command(["systemctl","is-active",name]).strip() for name in ("xeon-watch","xeon-watch-notify")]
    if active != "inactive" or workers or monitors != ["active","active"]:
        raise RuntimeError(f"API/monitor state: {active} {workers} {monitors}")


def memory_kib(path):
    return {s.split(":")[0].split()[-1]:int(s.split(":")[1].split()[0]) for s in path.read_text().splitlines()
            if any(k+":" in s for k in ("MemFree","MemAvailable","SwapFree","SwapTotal"))}


class Guard:
    def __init__(self, phase):
        self.dir = EV / phase
        self.lock = LOCK.open("r")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.before = json.loads((EV/"before.json").read_text())
        self.hw = json.loads((EV/"hardware.json").read_text())
        self.spec = self.hw["sensors"]
        self.sensor_paths = {k:(Path(v["path"]+"_input"), Path(v["path"]+"_crit_alarm")) for k,v in self.spec.items()}
        self.basec = self.before["counters"]
        self.before_sel = (EV/"before-sel.txt").read_text()
        self.last_slow = -100.0
        self.last_sel = -100.0
        self.zone = {}
        for z in sorted(Path("/sys/class/powercap").glob("intel-rapl:*")):
            name = get(z/"name")
            if name.startswith("package-"):
                self.zone[name]=z
            if name=="dram":
                pkg=get((z.parent/z.name.rsplit(":",1)[0])/"name")
                self.zone[pkg+"-dram"]=z
        real_state()
        if self.hw["boot"] != get(BOOT) or len(self.spec)!=26 or not self.basec:
            raise RuntimeError("boot, core sensor coverage or initial counters")
        self.dir.mkdir()
        self.output = (self.dir/"samples.jsonl").open("x",buffering=1)
        self.info = dict(phase=phase,start=now(),boot=get(BOOT),uid=os.getuid(),
                         rapl={k:{"path":str(p),"range_uj":int(get(p/"max_energy_range_uj"))} for k,p in self.zone.items()},
                         stop_mC={k:min(self.before["monitor"]["monitor"]["limits"]["cpu_C"]*1000,80000,v["crit_mC"]-10000) for k,v in self.spec.items()})
        if self.info["uid"]==0:
            raise RuntimeError("use ordinary user for stressors")
        save(self.dir/"start.json",self.info)
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,self.interrupted)

    @staticmethod
    def interrupted(signum,frame):
        raise InterruptedError(f"phase interrupted by signal {signum}")

    def tick(self, label, force=False):
        row = {**now(),"test":label}
        t=row["monotonic_s"]
        temps={k:int(get(v[0])) for k,v in self.sensor_paths.items()}
        alarms=[k for k,v in self.sensor_paths.items() if int(get(v[1]))]
        row["temp_mC"]=temps
        watch=json.loads(WATCH.read_text())
        b=watch["bmc"]
        row.update(watch_unix=watch["unix"],bmc_unix=b["observed_unix"],fan=b["fan_rpm"],
                   fan_duty=b["fan_percent"],dimm=b["dimms_C"],pch_C=b["pch_C"],peripheral_C=b["peripheral_C"],
                   gpu=watch["gpu"],monitor_errors=watch["errors"],alerts=watch["active_alerts"])
        mem=memory_kib(Path("/proc/meminfo"))
        row["mem"]=mem
        if self.zone:
            before=time.monotonic()
            vals=command(["sudo","-n","cat",*[str(z/"energy_uj") for z in self.zone.values()]],timeout=3).split()
            row["energy_uj"]=dict(zip(self.zone,map(int,vals),strict=True))
            row["energy_t"]=(before+time.monotonic())/2
        reason=[]
        if any(v>=self.info["stop_mC"][k] for k,v in temps.items()) or alarms:
            reason.append(f"temperature or alarm: max {max(temps.values())}, alarm {alarms}")
        if get(BOOT)!=self.hw["boot"] or watch["boot_id"]!=self.hw["boot"] or not 0<=time.time()-watch["unix"]<35 or not 0<=time.time()-b["observed_unix"]<75:
            reason.append("boot change or stale monitoring")
        if watch["errors"] or watch["active_alerts"] or b["bad_sensor_status"]:
            reason.append("watch errors, alerts or BMC sensor fault")
        if (not b["fan_rpm"] or any(v<=0 for v in b["fan_rpm"].values()) or
                not b["fan_percent"] or max(b["fan_percent"])>70 or
                any(k not in b["fan_rpm"] for k in self.before["monitor"]["bmc"]["fan_rpm"])):
            reason.append("fan unavailable or outside existing policy")
        lim=watch["monitor"]["limits"]
        if max(b["dimms_C"].values())>=lim["dimm_C"] or b["pch_C"]>=lim["pch_C"] or b["peripheral_C"]>=lim["peripheral_C"]:
            reason.append("peripheral/DIMM threshold")
        if watch["gpu"]["used_MiB"]!=0 or watch["gpu"]["power_limit_W"]!=150 or watch["gpu"]["ecc_mode"]!="Enabled":
            reason.append("GPU/idle state changed")
        if mem["MemAvailable"]<64*1024**2 or mem["SwapFree"]<self.before["meminfo"]["SwapFree"]:
            reason.append("memory reserve or swap changed")
        cur=counters()
        if cur!=self.basec:
            reason.append("EDAC/thermal counters changed")
            row["changed_counters"]={k:[self.basec.get(k),cur.get(k)] for k in self.basec.keys()|cur.keys() if self.basec.get(k)!=cur.get(k)}
        if force or t-self.last_slow>=6:
            new=command(["sudo","-n","journalctl","-k","-b","--after-cursor="+self.before["journal_cursor"],"--no-pager","-o","cat"],timeout=6)
            events=[s for s in new.splitlines() if ERROR.search(s)]
            if events:
                reason.append("new kernel hardware event")
                row["kernel_events"]=events
            real_state()
            self.last_slow=t
        if force or t-self.last_sel>=12:
            sel=command(["sudo","-n","ipmitool","-I","open","sel","elist"],timeout=6)
            if sel!=self.before_sel:
                reason.append("new or changed BMC event log")
                row["sel"]=sel
            self.last_sel=t
        row["stop_reasons"]=reason
        self.output.write(json.dumps(row,separators=(",",":"))+"\n")
        if reason:
            raise RuntimeError("; ".join(reason))
        return row

    def finish(self,status,details=None):
        self.output.close()
        save(self.dir/"end.json",{"status":status,"end":now(),"details":details})
        self.lock.close()


def idle(phase):
    if phase not in ("P0","P6"):
        raise ValueError(phase)
    g=Guard(phase)
    try:
        until=time.monotonic()+180
        while True:
            start=time.monotonic()
            g.tick(phase)
            if start>=until:
                break
            time.sleep(max(0,min(start+2,until)-time.monotonic()))
        g.finish("pass")
        print(phase,"180s completed")
    except BaseException as exc:
        g.finish("stopped",str(exc))
        raise


if __name__=="__main__":
    idle(sys.argv[1])
