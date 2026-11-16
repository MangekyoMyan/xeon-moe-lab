"""Export complete text evidence and deduplicated compressed correctness bytes to Git."""
from pathlib import Path
import gzip,hashlib,json,shutil
R=Path(__file__).resolve().parent
D=Path('/home/codex/projects/xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-16-abs-operator')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,rel):
    q=D/rel;q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
def main():
    D.mkdir(parents=True,exist_ok=False)
    for p in R.iterdir():
        if p.is_file() and p.suffix in ['.py','.cpp','.c','.h','.patch','.txt','.json']:copy(p,p.name)
    for folder in ['execution','checkpoint-screen-r1r2']:
        for p in (R/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.cpp','.c','.h','.patch','.txt','.json','.log']:copy(p,p.relative_to(R))
    copy(R/'sanitized/BUILD.json','sanitized/BUILD.json')
    incoming=R.parent/'pro-review-round2'
    for p in incoming.rglob('*'):
        if p.is_file():
            assert p.suffix in ['.json','.txt','.py','.asm','.cpp','.inc','.png']
            copy(p,Path('incoming')/p.relative_to(incoming))
    payloads={}
    def compress(source,targets):
        digest=sha(source)
        if digest not in payloads:
            q=D/'correctness'/(digest+'.gz');q.parent.mkdir(exist_ok=True)
            q.write_bytes(gzip.compress(source.read_bytes(),compresslevel=6,mtime=0))
            payloads[digest]={'gzip':str(q.relative_to(D)),'bytes':source.stat().st_size,'targets':[]}
        payloads[digest]['targets']+=targets
    for name in ['golden.bin','golden-r2.bin']:compress(R/name,[name])
    for run in json.loads((R/'FIXED_RECEIPT.json').read_text())['runs']:
        for name in run['files']:
            rel=Path('fixed-outputs')/run['run']/name
            compress(R/rel,[str(rel)])
    (D/'correctness/INDEX.json').write_text(json.dumps(payloads,indent=2)+'\n')
    manifest={str(p.relative_to(D)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'directory':str(D),'files_before_manifest':len(manifest),'bytes':sum(x['bytes'] for x in manifest.values()),'deduplicated_correctness_payloads':len(payloads)}))
if __name__=='__main__':main()
