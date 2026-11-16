"""Export complete source/receipts and original review, no executables or credentials."""
from pathlib import Path
import gzip,hashlib,json,shutil
R=Path(__file__).resolve().parent
D=Path('/home/codex/projects/xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-17-route-observation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(p):
    q=D/p.relative_to(R);q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
def main():
    D.mkdir(exist_ok=False,parents=True)
    for p in R.iterdir():
        if p.is_file() and p.suffix in ['.json','.txt','.py']:cp(p)
    for name in ['incoming','execution','route37']:
        for p in (R/name).rglob('*'):
            if p.is_file() and p.suffix in ['.json','.txt','.py','.log','.cpp','.c','.h','.patch','.png','.svg']:
                cp(p)
    source=R/'route37/golden.bin';q=D/'correctness/golden.bin.gz';q.parent.mkdir(exist_ok=True);q.write_bytes(gzip.compress(source.read_bytes(),compresslevel=6,mtime=0))
    (D/'CORRECTNESS.json').write_text(json.dumps({'path':'correctness/golden.bin.gz','uncompressed_bytes':source.stat().st_size,'uncompressed_sha256':sha(source),'scope':'Frozen CPU artificial graph outputs; not model weights or model routes'},indent=2)+'\n')
    m={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'directory':str(D),'files':len(m)+1,'bytes':sum(x['bytes'] for x in m.values())}))
if __name__=='__main__':main()
