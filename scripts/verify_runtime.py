#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json
def main():
    p=argparse.ArgumentParser();p.add_argument('runtime',type=Path);a=p.parse_args();root=a.runtime.resolve();m=json.loads((root/'RUNTIME_MANIFEST.json').read_text())
    for row in m['files']:
        path=root/row['file']
        if not path.resolve().is_relative_to(root):raise SystemExit('Unsafe runtime path')
        if row.get('symlink') is not None:
            if not path.is_symlink() or str(path.readlink())!=row['symlink']:raise SystemExit('Alias mismatch: '+row['file'])
        elif path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Hash mismatch: '+row['file'])
    print('Runtime files verified:',len(m['files']))
if __name__=='__main__':main()
