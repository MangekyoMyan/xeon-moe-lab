"""Export source and raw evidence after verified restoration; exclude binaries/secrets."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
D=R.parent.parent/'xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-19-shared-prefix'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not D.exists()
    assert json.loads((R/'RESTORED_SHARED_PREFIX_TRIAL.json').read_text())['normal_API_restored']
    assert json.loads((R/'STATE.json').read_text())['completed']
    key=(R.parent/'api-server-20261030/api-key.txt').read_bytes().strip();assert len(key)>16
    patterns=[rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',rb'\bhf_[A-Za-z0-9]{24,}\b',rb'\bsk-[A-Za-z0-9_-]{24,}\b',rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}']
    mapping=[];omitted=[];total=0
    for p in sorted(R.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        rel=p.relative_to(R)
        if rel.parts[0]=='incoming':
            if 'previous_handoff' in rel.parts or 'r169a_review_2026-11-19/source/' in str(rel) or p.name=='verify_package.py':
                omitted.append(dict(path=str(rel),reason='unchanged previous published evidence; incoming package itself is not repackaged'));continue
        if p.name in ['server-config.json','SHARING.json','PUBLIC_REPLAY.json','FINAL_RUNTIME_RECHECK.json'] or p.suffix in ['.o','.a','.lock']:
            omitted.append(dict(path=str(rel),reason='private config, generated binary, lock or post-publication receipt'));continue
        raw=p.read_bytes()
        if raw.startswith(b'\x7fELF'):
            omitted.append(dict(path=str(rel),reason='ELF binary; source/build/hash retained'));continue
        assert len(raw)<50*1024**2 and p.suffix not in ['.gguf','.safetensors','.pem','.key'],rel
        assert key not in raw and not any(re.search(x,raw) for x in patterns),'credential pattern: '+str(rel)
        shared=rel.with_name('guard-health.json') if p.name=='health.json' else rel
        if rel.parts[0]=='incoming' and p.name=='FILES_SHA256.json':shared=rel.with_name('ORIGINAL_PACKAGE_FILES_SHA256.json')
        out=D/shared;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw);total+=len(raw)
        mapping.append(dict(original=str(rel),shared=str(shared),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),byte_identical=True))
    assert total<512*1024**2
    (D/'RAW_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    scope=dict(omitted=omitted,secret_scan_passed=True,prior_fixed_commit='169a2f7eab8d52071661f0da6859d38df4206f8b',prior_archive='archive/review-handoffs-not-for-readers/2026-11-19-fbc-h1-iq08',incoming_notice='Selected new review documents, audits and reference sources only; previous_handoff and byte-identical source mirror are already public at the prior commit. ORIGINAL_PACKAGE_FILES_SHA256 is historical, not the export manifest. Use root FILES_SHA256 to verify this export.',scope='No binaries/model/private guard config. guard-health is a byte-identical rename. All actual new raw outputs, source, negative results and corrected drafts remain.')
    (D/'EXPORT_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
    m={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(m)+1,bytes=sum(x['bytes'] for x in m.values()),directory=str(D),secret_scan_passed=True)))
if __name__=='__main__':main()
