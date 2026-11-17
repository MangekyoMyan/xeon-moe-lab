#!/usr/bin/env python3
"""Create a review-only patch; never modify the supplied repository.

Usage: python make_empty_bypass_proposal.py REPOSITORY OUTPUT_DIRECTORY
Uses only the Python standard library. It does not execute repository code.
"""
import difflib
import hashlib
import json
from pathlib import Path
import sys

PIN = "d98a17c56fadc92cf79e21ecbe80732dedeed899"
REL = Path("archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace/bank-capacity/hot-backend.cpp")
EXPECTED_SHA256 = "9f647622e0578b86b2042b9d63f77e6bdcae80b72ac20b69a1e79d24672cd767"


def main():
    repo = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    source = (repo / REL).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"source hash differs from reviewed pin: {digest}")
    original = source.decode("utf-8")
    edited = original
    replacements = [
        (
            "        for(auto & ptr:layers){if(!ptr)continue;auto & l=*ptr;if(!l.eligible)continue;\n",
            "        // Empty selections stay on the ordinary model graph, including CPU execution.\n"
            "        for(auto & ptr:layers){if(!ptr)continue;auto & l=*ptr;if(!l.eligible||l.selected.empty())continue;\n",
        ),
        ("            if(l.selected.empty())continue;\n", ""),
        (
            "    if(!l.eligible)return nullptr;\n",
            "    // A layer with no resident experts cannot benefit from the private dispatcher.\n"
            "    if(!l.eligible||l.selected.empty())return nullptr;\n",
        ),
    ]
    for before, after in replacements:
        if edited.count(before) != 1:
            raise SystemExit(f"expected unique source anchor: {before!r}")
        edited = edited.replace(before, after, 1)
    output.mkdir(parents=True, exist_ok=True)
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), edited.splitlines(keepends=True),
        fromfile="a/" + REL.as_posix(), tofile="b/" + REL.as_posix(),
    ))
    (output / "empty_selection_bypass.patch").write_text(patch, encoding="utf-8")
    receipt = {
        "kind": "source-only proposal; not a deployed change",
        "repository_pin": PIN,
        "source": REL.as_posix(),
        "source_sha256": digest,
        "proposed_source_sha256": hashlib.sha256(edited.encode()).hexdigest(),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "repository_modified": False,
        "compiled": False,
        "executed": False,
        "numeric_equivalence_measured": False,
        "performance_measured": False,
        "preserved": [
            "native plan validation and payload accounting",
            "Gpre/Gpost inequalities and initialization phase",
            "nonempty-layer CPU/GPU graph arithmetic and runtime partition",
            "GPU upload/scatter, routing weights, and reduction order",
        ],
        "behavior_change": [
            "No private CPU graph or warmup for eligible layers with an empty selection",
            "xeon_hot_build returns nullptr for empty selections, preserving the caller's ordinary graph",
            "An entirely empty plan no longer exercises dispatcher all-miss behavior",
        ],
        "not_included": [
            "main context/scheduler/graph integration",
            "route37 logical route versus private CPU execution separation",
            "benchmark modes or timing hooks",
            "runtime H0 bypass for layers with a nonempty static selection",
        ],
    }
    (output / "EMPTY_SELECTION_PATCH_MANIFEST.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"patch": str(output / "empty_selection_bypass.patch"),
                      "source_sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
