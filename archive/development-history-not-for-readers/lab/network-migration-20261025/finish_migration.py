#!/usr/bin/env python3
"""Retire only the X540 interface after its live TCP connections are gone."""
import argparse
import datetime
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / "results" / ROOT.name
CONFIG = Path("/etc/network/interfaces")
OLD_INTERFACE = "enp130s0f0"
NEW_INTERFACE = "enp4s0f0"
OLD_IP = "10.10.10.2"
NEW_IP = "192.168.0.14"
BLOCK = "# 10GbE direct link to main PC\nauto enp130s0f0\niface enp130s0f0 inet static\n    address 10.10.10.2/24\n"


def command(args):
    return subprocess.check_output(args, text=True, timeout=30)


def connections():
    rows = []
    for line in command(["ss", "-Htnp", "state", "established"]).splitlines():
        fields = line.split()
        if len(fields) >= 4:
            rows.append({"local": fields[2], "peer": fields[3], "process": " ".join(fields[4:])})
    return rows


def preflight():
    rows = connections()
    old = [r for r in rows if r["local"].rsplit(":", 1)[0] == OLD_IP]
    new = [r for r in rows if r["local"] == NEW_IP + ":22"]
    route = json.loads(command(["ip", "-j", "-4", "route", "show", "default"]))
    addresses = json.loads(command(["ip", "-j", "address", "show", "dev", NEW_INTERFACE]))
    ready = any(a.get("local") == NEW_IP for i in addresses for a in i["addr_info"])
    assert new and ready and (Path("/sys/class/net") / NEW_INTERFACE / "operstate").read_text().strip() == "up", "Onboard SSH path is not established"
    assert route and all(r.get("dev") == NEW_INTERFACE for r in route), "Unexpected default route"
    return {"old_connections": old, "new_ssh_connections": new, "default_routes": route}


def replace_config(text):
    assert not CONFIG.is_symlink()
    metadata = CONFIG.stat()
    descriptor, name = tempfile.mkstemp(prefix=".interfaces-x540-", dir=CONFIG.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), stat.S_IMODE(metadata.st_mode))
            os.fchown(stream.fileno(), metadata.st_uid, metadata.st_gid)
        os.replace(name, CONFIG)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() == 0
    if args.apply and (ROOT / "HOLD_OLD_ROUTE").exists():
        print("STOP: the user explicitly requires the old route to remain. Nothing changed.")
        return 4
    before = (OUT / "interfaces.before").read_text()
    proposed = (OUT / "interfaces.proposed").read_text()
    assert before.count(BLOCK) == 1 and proposed == before.replace("\n" + BLOCK, "\n")
    assert CONFIG.read_text() == before, "Configuration changed since inspection; review again"
    command(["ifquery", "--interfaces", str(OUT / "interfaces.proposed"), NEW_INTERFACE])
    checks = preflight()
    print(json.dumps(checks, indent=2), flush=True)
    if checks["old_connections"]:
        print("STOP: the old interface still carries live TCP connections. Nothing changed.")
        return 3
    if not args.apply:
        print("Ready. Check only; nothing changed.")
        return 0
    record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "before": checks}
    attempted = False
    try:
        # Check again immediately before stopping the old interface.
        assert not preflight()["old_connections"]
        attempted = True
        command(["ifdown", OLD_INTERFACE])
        replace_config(proposed)
        time.sleep(2)
        after = preflight()
        old_addresses = json.loads(command(["ip", "-j", "address", "show", "dev", OLD_INTERFACE]))
        assert not any(a.get("local") == OLD_IP for i in old_addresses for a in i["addr_info"])
        before_pairs = {(r["local"], r["peer"]) for r in checks["new_ssh_connections"]}
        after_pairs = {(r["local"], r["peer"]) for r in after["new_ssh_connections"]}
        assert before_pairs <= after_pairs, "An onboard SSH connection disappeared; restore old path"
        assert after["default_routes"] == checks["default_routes"]
        record.update(status="complete", after=after, old_interface_addresses=old_addresses)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        if attempted:
            current = CONFIG.read_text()
            if current == proposed:
                replace_config(before)
            elif current != before:
                record["rollback_error"] = "Unexpected concurrent configuration change"
            if CONFIG.read_text() == before:
                try:
                    record["rollback_ifup"] = command(["ifup", OLD_INTERFACE])
                except Exception as rollback_error:
                    record["rollback_error"] = repr(rollback_error)
        raise
    finally:
        target = OUT / "migration-result.json"
        target.write_text(json.dumps(record, indent=2) + "\n")
        owner = LAB.stat()
        os.chown(target, owner.st_uid, owner.st_gid)
    print("Completed. Only X540 configuration/interface was retired; onboard settings were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
