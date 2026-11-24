"""Read the swap host. Full kernel and SMBIOS output stays in private/."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EV = ROOT / "evidence"
PRIVATE = ROOT / "private"
WATCH = Path("/var/lib/xeon-watch/latest.json")
CPU = Path("/sys/devices/system/cpu")
MC = Path("/sys/devices/system/edac/mc")
NODE = Path("/sys/devices/system/node")
BOOT = Path("/proc/sys/kernel/random/boot_id")
API = "llm-deepseek41-api.service"
ENV = {**os.environ, "LC_ALL": "C"}
UNIT_FIELDS = "Id,LoadState,ActiveState,SubState,MainPID,UnitFileState,Restart,TriggeredBy"
ERROR = re.compile(r"\[hardware error\]|machine check (?:exception|events logged)|mce:.*(?:bank [0-9]|hardware error)|EDAC .*\b(?:CE|UE)\b|temperature above threshold|critical temperature|cpu.*throttled|NVRM: Xid|PCIe Bus Error|out of memory:|oom-kill:|general protection fault|BUG:|watchdog: BUG|kernel panic", re.I)


def command(args, allow=(0,), timeout=25):
    p = subprocess.run(args, text=True, errors="replace", capture_output=True,
                       env=ENV, timeout=timeout)
    if p.returncode not in allow:
        raise RuntimeError(f"{args[:4]} rc={p.returncode} {p.stderr[:250]}")
    return p.stdout + p.stderr


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic()}


def get(p):
    return p.read_text().strip()


def cpu_map():
    data = []
    allowed = os.sched_getaffinity(0)
    for p in sorted(CPU.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        i = int(p.name[3:])
        topo = p / "topology"
        data.append(dict(cpu=i, socket=int(get(topo/"physical_package_id")),
                         core=int(get(topo/"core_id")), node=int(next(p.glob("node[0-9]*")).name[4:]),
                         siblings=get(topo/"thread_siblings_list"), allowed=i in allowed,
                         online=not (p/"online").exists() or get(p/"online") == "1",
                         microcode=get(p/"microcode/version"),
                         governor=get(p/"cpufreq/scaling_governor")))
    return data


def sensors():
    result = {}
    for dev in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        if get(dev/"name") != "coretemp":
            continue
        package = get(dev/"temp1_label").split()[-1]
        for p in sorted(dev.glob("temp*_label")):
            stem = p.name.split("_")[0]
            result[f"S{package}/{get(p)}"] = dict(path=str(dev/stem), crit_mC=int(get(dev/(stem+"_crit"))),
                                                  max_mC=int(get(dev/(stem+"_max"))))
    return result


def counters():
    paths = set()
    for pattern in ("mc*/?e_count", "mc*/?e_noinfo_count", "mc*/dimm*/dimm_?e_count",
                    "mc*/csrow*/?e_count", "mc*/csrow*/ch*_ce_count"):
        paths.update(MC.glob(pattern))
    paths.update(CPU.glob("cpu[0-9]*/thermal_throttle/*_throttle_count"))
    return {str(p): int(get(p)) for p in sorted(paths)}


def dimm_blocks(dmi, marker, include):
    blocks = []
    for section in dmi.split("\n\n"):
        if "\n" + marker + "\n" in section:
            items = dict(s.strip().split(": ", 1) for s in section.splitlines() if ": " in s)
            blocks.append({k: items[k] for k in include if k in items})
    return blocks


def public_hw():
    model = command(["lscpu"])
    (EV/"lscpu.txt").write_text(model)
    (EV/"lscpu-extended.txt").write_text(command(["lscpu", "-e=CPU,SOCKET,CORE,NODE,ONLINE"]))
    (EV/"numa.txt").write_text(command(["numactl", "--hardware"]))
    cpu = cpu_map()
    dmi = {}
    for t in (4, 17):
        dmi[t] = command(["sudo", "-n", "dmidecode", "-t", str(t)])
        (PRIVATE / f"dmi-{t}.txt").write_text(dmi[t])
    base = Path("/sys/class/dmi/id")
    return dict(boot=get(BOOT), kernel=os.uname().release, os=get(Path("/etc/os-release")),
                board={k:get(base/k) for k in ("board_vendor","board_name","bios_version","bios_date")},
                cpus=cpu, cpu_online=get(CPU/"online"), cpu_present=get(CPU/"present"), cpu_offline=get(CPU/"offline"),
                no_turbo=get(CPU/"intel_pstate/no_turbo") if (CPU/"intel_pstate/no_turbo").exists() else None,
                processors=dimm_blocks(dmi[4], "Processor Information", ["Socket Designation","Version","Status","Core Count","Core Enabled","Thread Count"]),
                dimms=dimm_blocks(dmi[17], "Memory Device", ["Locator","Bank Locator","Size","Type","Type Detail","Rank","Speed","Configured Memory Speed","Configured Clock Speed","Configured Voltage"]),
                edac_dimm_labels={str(p.parent):get(p) for p in MC.glob("mc*/dimm*/dimm_label")},
                sensors=sensors())


def host_snapshot(tag):
    if tag not in ("before","after"):
        raise ValueError(tag)
    if (EV/f"{tag}.json").exists():
        raise RuntimeError(f"{tag} snapshot already exists")
    at=now()
    unit = command(["systemctl","show",API,"xeon-watch.service","xeon-watch-notify.service","rasdaemon.service","-p",UNIT_FIELDS])
    journalargs = ["sudo","-n","journalctl","-k","-b","--no-pager","--show-cursor","-o","short-iso-precise"]
    if tag == "after":
        ref=json.loads((EV/"before.json").read_text())
        if ref["boot"] != get(BOOT):
            raise RuntimeError("boot changed since before")
        journalargs += ["--after-cursor="+ref["journal_cursor"]]
    journal=command(journalargs)
    (PRIVATE/f"kernel-{tag}.txt").write_text(journal)
    cursor=next((s.split(": ",1)[1] for s in reversed(journal.splitlines()) if s.startswith("-- cursor: ")),None)
    if tag=="before" and not cursor:
        raise RuntimeError("no initial journal cursor")
    data=dict(time=at, boot=get(BOOT), units=unit, models=command(["ps","-C","llama-server,llama-cli,llama-bench,ollama,vllm","-o","pid=,comm="],allow=(0,1)).splitlines(),
              journal_cursor=cursor, journal_args=journalargs, journal_hardware_matches=[line for line in journal.splitlines() if ERROR.search(line)],
              journal_text_records=sum(bool(s) and not s.startswith("--") for s in journal.splitlines()),
              meminfo={s.split(":")[0]:int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines()
                       if s.split(":")[0] in ("MemTotal","MemAvailable","SwapTotal","SwapFree")},
              counters=counters(), monitor=json.loads(WATCH.read_text()))
    for name, words in (("sel-info",["sel","info"]),("sel-time",["sel","time","get"]),("sel",["sel","elist"]),("bmc-sensors",["sensor","list"])):
        out=command(["sudo","-n","ipmitool","-I","open",*words])
        (EV/f"{tag}-{name}.txt").write_text(out)
    (EV/f"units-{tag}.txt").write_text(unit)
    save(EV/f"{tag}.json",data)
    if tag=="before":
        save(EV/"hardware.json",public_hw())
        dependencies=command(["systemctl","list-dependencies","--reverse",API])
        (EV/"api-reverse-dependencies.txt").write_text(dependencies)
    print(tag,"model PIDs",data["models"],"kernel hardware matches",len(data["journal_hardware_matches"]),
          "nonzero counters",sum(v != 0 for v in data["counters"].values()),"sel",get(EV/f"{tag}-sel.txt"))


if __name__ == "__main__":
    host_snapshot(sys.argv[1])
