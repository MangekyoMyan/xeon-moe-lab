"""Check exactly the processes seen in the test trees. No signals are sent."""
import json
from pathlib import Path
from host import BOOT, EV, command, get, now, save


def main():
    original=json.loads((EV/"before.json").read_text())
    if get(BOOT)!=original["boot"]:
        raise RuntimeError("boot changed")
    observed=set()
    for path in EV.glob("P*/s*-cpu*.json"):
        for sample in json.loads(path.read_text())["snapshots"]:
            observed.update((p["pid"],p["start_tick"]) for p in sample["tree"])
    surviving=[]
    for pid,start in sorted(observed):
        try:
            st=get(Path(f"/proc/{pid}/stat")).split(") ",1)[1].split()
            if int(st[19])==start:
                surviving.append({"pid":pid,"start_tick":start,"state":st[0]})
        except (FileNotFoundError,ProcessLookupError):
            pass
    related=command(["ps","-C","stress-ng,stress-ng-cpu,stress-ng-vm","-o","pid=,comm="],allow=(0,1)).strip()
    if surviving or related or not observed:
        raise RuntimeError((surviving,related))
    save(EV/"worker-cleanup.json",dict(time=now(),boot=get(BOOT),observed_pid_start_pairs=len(observed),
             observed_survivors=surviving,system_matching_stress_processes=related.splitlines(),all_terminated=True))
    print("All",len(observed),"observed process identities exited")


if __name__=="__main__":
    main()
