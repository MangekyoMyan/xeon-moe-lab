"""Publish source and complete mixed-run evidence, excluding executables and credentials."""
from pathlib import Path
import gzip,hashlib,json,shutil
R=Path(__file__).resolve().parent
D=Path('/home/codex/projects/xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-17-mixed-operator')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p):
    target=D/p.relative_to(R);target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,target)
def main():
    D.mkdir(exist_ok=False,parents=True)
    for p in R.iterdir():
        if p.is_file() and p.suffix in ['.txt','.py','.json']:copy(p)
    for folder in ['incoming','outer-guard']:
        for p in (R/folder).rglob('*'):
            if p.is_file():
                assert p.suffix in ['.txt','.py','.json','.jsonl','.log','.cpp','.h','.bin'],p
                if p.suffix=='.bin':assert p.name=='inputs.bin' and p.stat().st_size==163840
                copy(p)
    copy(R/'build/BUILD.json')
    payloads={}
    for p in (R/'mixed-run').rglob('*'):
        if not p.is_file() or p.name=='RUN.lock':continue
        if p.suffix!='.bin':copy(p);continue
        assert p.name in ['expected-C0.bin','expected-C6.bin','expected-Cplan.bin'] and p.stat().st_size==442368
        digest=sha(p)
        if digest not in payloads:
            out=D/'correctness'/(digest+'.gz');out.parent.mkdir(exist_ok=True)
            out.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
            payloads[digest]={'gzip':str(out.relative_to(D)),'bytes':p.stat().st_size,'targets':[]}
        payloads[digest]['targets'].append(str(p.relative_to(R)))
    (D/'correctness/INDEX.json').write_text(json.dumps(payloads,indent=2)+'\n')
    manifest={str(p.relative_to(D)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(manifest)+1,'bytes':sum(x['bytes'] for x in manifest.values()),'correctness_blobs':len(payloads)}))
if __name__=='__main__':main()
