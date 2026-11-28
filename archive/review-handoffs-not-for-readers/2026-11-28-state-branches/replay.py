"""Verify the complete release without an inference server, model or GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/"MANIFEST.json").read_bytes())
    entries={**manifest["record"],**manifest["software"],**manifest["native_binary_data"]}
    for name,spec in entries.items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and path.stat().st_size==spec["bytes"],name
        with path.open("rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()==spec["sha256"],name
    for script in ("verify_rows.py","assess.py"):
        subprocess.run([sys.executable,"-B",str(ROOT/script),"--check"],check=True)
    subprocess.run([sys.executable,"-B","-m","unittest","discover","-s",str(ROOT/"cli"),"-p","test_*.py"],check=True)
    print("Verified",len(entries),"hashed files.")


if __name__=="__main__":
    main()
