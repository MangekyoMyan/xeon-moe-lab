"""Export the isolated caller, review inputs and complete non-secret evidence."""
from pathlib import Path
import hashlib
import json
import shutil

R=Path(__file__).resolve().parent
D=Path('/home/codex/projects/xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-17-route-calibration')
ALLOW={'.py','.h','.cpp','.txt','.json','.patch','.log','.f32'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not D.exists(),'Fresh immutable export required'
    D.mkdir(parents=True)
    mapping=[]
    for p in sorted(R.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(R)
        if any(part.startswith('runtime-') or part=='__pycache__' for part in rel.parts):continue
        if p.name in ['server-config.json','SHARING.json'] or p.suffix not in ALLOW:continue
        dest=D/rel
        if p.name=='health.json':dest=dest.with_name('guard-health.json')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        mapping.append(dict(original=str(p),shared=str(dest.relative_to(D)),sha256=sha(p),byte_identical=True))
    (D/'RAW_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    m={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(directory=str(D),files=len(m)+1,bytes=sum(x['bytes'] for x in m.values()))))

if __name__=='__main__':main()
