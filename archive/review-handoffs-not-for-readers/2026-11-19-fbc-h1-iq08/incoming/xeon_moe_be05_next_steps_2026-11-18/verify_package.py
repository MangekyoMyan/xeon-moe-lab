#!/usr/bin/env python3
"""Check this handoff's file inventory and SHA256 without running its code."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
manifest = json.loads((root / "FILES_SHA256.json").read_text(encoding="utf-8"))
expected = {entry["path"]: entry for entry in manifest["files"]}
actual = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
if any(p.is_symlink() for p in root.rglob("*")):
    raise SystemExit("Unexpected symbolic link")
if set(actual) != set(expected) | {"FILES_SHA256.json"}:
    raise SystemExit("File inventory differs from the manifest")
for name, entry in expected.items():
    data = actual[name].read_bytes()
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit("File identity mismatch: " + name)
print(json.dumps({"status": "pass", "verified_entries": len(expected),
                  "total_files": len(actual), "manifest_excludes_itself": True}))
