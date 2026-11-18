"""Export reviewed sources/raw records, without runtime binaries or model weights."""
from pathlib import Path
import hashlib,json,re,shutil
R=Path(__file__).resolve().parent
D=R.parent.parent/'xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-18-q8-k-avx1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not D.exists();assert json.loads((R/'RESTORED_RUNTIME.json').read_text())['normal_API_restored']
    key=(R.parent/'api-server-20261030/api-key.txt').read_bytes().strip();assert len(key)>16
    pats=[rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',rb'\bhf_[A-Za-z0-9]{24,}\b',rb'\bsk-[A-Za-z0-9_-]{24,}\b',rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}']
    items=[];omitted=[]
    for p in sorted(R.rglob('*')):
        if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(R)
        if p.name in ['server-config.json','SHARING.json'] or p.suffix in ['.o','.a','.lock']:
            omitted.append(str(rel));continue
        raw=p.read_bytes()
        if raw.startswith(b'\x7fELF'):omitted.append(str(rel));continue
        assert len(raw)<50*1024**2 and p.suffix not in ['.gguf','.safetensors','.pem','.key'],rel
        assert key not in raw and not any(re.search(pattern,raw) for pattern in pats),'credential pattern: '+str(rel)
        public=rel.with_name('guard-health.json') if p.name=='health.json' else rel
        items.append((p,public,raw))
    assert sum(len(raw) for p,rel,raw in items)<512*1024**2
    for p,rel,raw in items:
        out=D/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    mapping=[dict(original=str(p),shared=str(rel),sha256=hashlib.sha256(raw).hexdigest(),byte_identical=True) for p,rel,raw in items]
    (D/'RAW_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (D/'EXPORT_SCOPE.json').write_text(json.dumps(dict(omitted=omitted,reason='No ELF/object/model weights/private guard configs. guard-health is byte-identical rename. Per-run config is reconstructed offline from LAUNCH command/environment.',secret_scan_passed=True),indent=2)+'\n')
    m={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(m)+1,bytes=sum(x['bytes'] for x in m.values()),directory=str(D),secret_scan_passed=True)))
if __name__=='__main__':main()
