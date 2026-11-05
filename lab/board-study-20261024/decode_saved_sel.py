#!/usr/bin/env python3
"""Decode the archived records without changing the BMC or assuming a failed DIMM."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
raw = (OUT / "sel-before.bin").read_bytes()
rows = []
for offset in range(0, len(raw), 16):
    record = raw[offset:offset + 16]
    assert len(record) == 16 and record[2] == 2
    assert record[10] == 0x0c and (record[12] & 0x7f) == 0x6f
    data1, data2, data3 = record[13:16]
    assert (data1 & 15) == 0
    cpu = (data3 & 3) + 1
    channel = chr((data2 >> 4) + 0x40 + (data3 & 3) * 4)
    slot = chr((data2 & 15) + 0x27)
    rows.append({"record_id": int.from_bytes(record[:2], "little"), "timestamp_raw": int.from_bytes(record[3:7], "little"),
                 "event_data_hex": record[13:].hex(), "conditional_romely_location": f"P{cpu}-DIMM{channel}{slot}"})
report = {
    "observed_product_id": "0x0628", "observed_manufacturer_id": 0,
    "rule": "ipmitool Supermicro Romely rule; 0x0628 is in its product table",
    "certainty": "Conditional OEM interpretation, not proof of a currently defective module; manufacturer 0 prevents normal OEM identification",
    "source_code": "https://github.com/ipmitool/ipmitool/blob/master/lib/ipmi_sel.c",
    "product_table": "https://sources.debian.org/src/ipmitool/1.8.19-4%2Bdeb12u2/include/ipmitool/ipmi_sel_supermicro.h",
    "counts": dict(collections.Counter(row["conditional_romely_location"] for row in rows)), "records": rows,
}
(OUT / "sel-conditional-location.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
