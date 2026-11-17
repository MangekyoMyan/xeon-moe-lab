"""Read-only offline replay of a public export; no model/hardware/network access.

The public archive renames guard health and excludes the runtime config filename.
Reconstruct those two names in a temporary copy, verify the original hashes, and
run the same analysis. The runtime command is DATA and is never executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
from analyze import one

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists()
    manifest=json.loads((a.root/'FILES_SHA256.json').read_text())
    for name,receipt in manifest.items():
        data=(a.root/name).read_bytes();assert len(data)==receipt['bytes'] and hashlib.sha256(data).hexdigest()==receipt['sha256']
    results=[]
    with tempfile.TemporaryDirectory(prefix='route37-offline-') as tmp:
        for src in sorted((a.root/'runs').iterdir()):
            if not src.is_dir():continue
            dst=Path(tmp)/src.name;shutil.copytree(src,dst)
            (dst/'guard-health.json').rename(dst/'health.json')
            launch=json.loads((dst/'LAUNCH_MANIFEST.json').read_text())
            (dst/'server-config.json').write_text(json.dumps(dict(command=launch['command'],environment=launch['environment']),indent=2)+'\n')
            results.append(one(dst))
    a.output.write_text(json.dumps(dict(passed=True,public_files=len(manifest),runs=results),indent=2)+'\n')
    print(json.dumps(dict(passed=True,model_started=False,public_files=len(manifest),runs=len(results))))

if __name__=='__main__':main()
