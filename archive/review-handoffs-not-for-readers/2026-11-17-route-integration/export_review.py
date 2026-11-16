"""Export complete source and evidence, excluding executables/objects and private runtime config."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent
D=Path('/home/codex/projects/xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-17-route-integration')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(p):
    q=D/p.relative_to(R);q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
def main():
    if D.exists():
        old=json.loads((D/'FILES_SHA256.json').read_text())
        for name,receipt in old.items():
            if name!='FILES_SHA256.json':assert sha(D/name)==receipt['sha256'],'Export file edited externally: '+name
    D.mkdir(exist_ok=True,parents=True)
    allow={'.py','.h','.cpp','.c','.cxx','.txt','.json','.jsonl','.log','.patch','.map','.d'}
    for p in R.iterdir():
        if p.is_file() and p.suffix in allow:cp(p)
    for folder in ['incoming','route37','provenance','probe-results','probe-results-v2','execution','checkpoint-before-exception','checkpoint-first-clients-build','checkpoint-before-retry-test']:
        for p in (R/folder).rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            if p.suffix in allow:cp(p)
            elif p.name=='RESTORED_ARTIFICIAL_GOLDEN.bin':assert p.stat().st_size==17760;cp(p)
    m={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'FILES_SHA256.json'}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'directory':str(D),'files':len(m)+1,'bytes':sum(x['bytes'] for x in m.values())}))
if __name__=='__main__':main()
