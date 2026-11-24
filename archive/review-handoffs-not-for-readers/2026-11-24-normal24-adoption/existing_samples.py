"""Retain the native guard's rolling samples during cutover and acceptance. No probes."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parent/"deepseek41-bringup-20261101"


def read(p):
    return json.loads(p.read_bytes())


def main():
    baseline=read(ROOT/"HEALTH_BEFORE.json")
    old=read(ROOT/"BEFORE_IDENTITY.json")["pid"];seen=set();last=None;deadline=time.monotonic()+7300
    hw=read(ROOT.parent/"postswap-health-20261124/evidence/hardware.json")
    pk={k.split("/")[0]:f"{v['path']}_input" for k,v in hw["sensors"].items() if "/Package " in k}
    assert len(pk)==2
    with (ROOT/"GUARD_SAMPLES.jsonl").open("x",buffering=1) as g,(ROOT/"WATCH_SAMPLES.jsonl").open("x",buffering=1) as watch:
        while time.monotonic()<deadline:
            w=read(Path("/var/lib/xeon-watch/latest.json"))
            if w["unix"]!=last:
                last=w["unix"];b=w["bmc"];c=w["host_counters"];base=baseline["host_counters"]
                diff={k:[base.get(k),c.get(k)] for k in c.keys()|base.keys() if c.get(k)!=base.get(k)}
                watch.write(json.dumps({"unix":last,"boot":w["boot_id"],"dimm_max_C":max(b["dimms_C"].values()),
                        "bmc_unix":b["observed_unix"],"fans":b["fan_percent"],"errors":bool(w["errors"] or b["bad_sensor_status"]),
                        "alert":bool(w["active_alerts"]),"counter_diff":diff,"gpu_ecc_single":w["gpu"].get("ecc_single"),
                        "gpu_ecc_double":w["gpu"].get("ecc_double")},separators=(",",":"))+"\n")
            h=read(ENGINE/"health.json")
            if h["boot_id"]==baseline["boot"] and h.get("pid") not in (None,old):
                for row in h.get("recent_samples",[]):
                    k=h["pid"],row["unix"]
                    if k in seen:continue
                    seen.add(k);gpu=row["gpu"];t=row["cpu_C"]
                    g.write(json.dumps({"pid":k[0],"unix":k[1],"packages_C":{k:t[v] for k,v in pk.items()},
                            "cpu_max_C":max(t.values()),"gpu_C":gpu["gpu_C"],"hbm_C":gpu["hbm_C"],"gpu_W":gpu["power_W"],
                            "limit_W":gpu["limit_W"],"ecc":gpu["ecc"],"gpu_util_pct":gpu["gpu_util"],
                            "fan_percent":row["fans"]["percent"],"available_kib":row["available_kib"]},separators=(",",":"))+"\n")
            if (ROOT/"ACCEPTANCE.json").is_file() and last>=read(ROOT/"ACCEPTANCE.json")["finish_unix"]:break
            if (ROOT/"ROLLBACK_VERIFIED.json").exists() or (ENGINE/"STOP_REQUESTED").exists() or w["boot_id"]!=baseline["boot"]:break
            time.sleep(5)
    print("Native guard samples saved:",len(seen))


if __name__=="__main__":main()
