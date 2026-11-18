"""Prepare two review-only H1 diagnostic plans from the pinned original35 plan.

No engine is loaded and no repository file is modified. This is a mechanism
experiment, not placement training or a production candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = args.repo / "archive/review-handoffs-not-for-readers/2026-11-17-parallel-pilot"
    source = base / "integrated/capacity.plan"
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    assert source_sha == "4b5a8063ae1e61c55d30f957798a7cc44d7f462cf22767d2596105ed7322c596"
    lines = raw.decode().splitlines()
    head = lines[0].split()
    assert head == ["XEON_HOT_FFN_PLAN_V1", "40", "384", "5120", "2304", "35", "475453440"]
    rows = [line.split() for line in lines[1:]]
    assert len(rows) == 40
    original = {int(r[0]): list(map(int, r[5:])) for r in rows}
    replay = json.loads((base / "replay/runs/20261117T053052Z-U1024-1563d936/REPLAY.json").read_text())
    variants = {
        "H1_COMPACT": {7: [19,154,331], 35: [0,1,78]},
        "H1_NONCOMPACT": {7: [19,85,154], 35: [0,1,100]},
    }
    record = {
        "status": "prepared offline; NOT run on engine; not a production bank",
        "commit": "37f6390c9a5bc594658127651e204ab11420cd71",
        "source_plan_sha256": source_sha,
        "active_mask_exists_in_current_API": False,
        "comparison": "two fixed memberships, equal per-layer counts/types/native bytes/initialization shapes; NOT identical bank membership",
        "fresh_process_order": ["H1_COMPACT", "H1_NONCOMPACT", "H1_NONCOMPACT", "H1_COMPACT"],
        "primary_cells": [1,4],
        "total_experts": 35,
        "native_payload_bytes": 475453440,
        "plans": {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, changes in variants.items():
        selected = dict(original)
        selected.update(changes)
        assert sum(map(len, selected.values())) == 35
        assert all(len(selected[i]) == len(original[i]) for i in range(40))
        assert all(v == sorted(set(v)) and all(0 <= e < 384 for e in v) for v in selected.values())
        output_lines = [lines[0]]
        for i, row in enumerate(rows):
            assert int(row[0]) == i and int(row[4]) == len(selected[i])
            output_lines.append(" ".join(row[:5] + list(map(str, selected[i]))))
        data = ("\n".join(output_lines) + "\n").encode()
        path = args.out / (name + ".plan")
        path.write_bytes(data)
        cells = []
        for cell in replay["cells"]:
            ids = cell["ids"]
            gpu = [e for e in ids if e in selected[cell["layer"]]]
            cells.append({"cell":cell["cell"],"layer":cell["layer"],"global_top6":ids,"GPU_global_ids":gpu,"H":len(gpu)})
        expected = {1:331 if name=="H1_COMPACT" else 85, 4:78 if name=="H1_COMPACT" else 100}
        for idx, expert in expected.items():
            c = next(c for c in cells if c["cell"] == idx)
            assert c["GPU_global_ids"] == [expert]
        record["plans"][name] = {"file":path.name,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"changed_layers":{str(i):{"original":original[i],"diagnostic":selected[i]} for i in range(40) if original[i]!=selected[i]},"all_eight_cells":cells}
    (args.out / "H1_PLAN_VERIFICATION.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k!='plans'},ensure_ascii=False))
    for name, p in record["plans"].items():
        print(json.dumps({"plan":name,"sha256":p["sha256"],"bytes":p["bytes"],"primary":[c for c in p["all_eight_cells"] if c['cell'] in [1,4]]},ensure_ascii=False))


if __name__ == "__main__":
    main()
