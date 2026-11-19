from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
D=R.parent.parent/'xeon-moe-review-20261116/archive/review-handoffs-not-for-readers/2026-11-19-fbc-h1-iq08'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not D.exists();assert json.loads((R/'RESTORED_CPU08.json').read_text())['normal_API_restored']
    key=(R.parent/'api-server-20261030/api-key.txt').read_bytes().strip();assert len(key)>16
    pats=[rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',rb'\bhf_[A-Za-z0-9]{24,}\b',rb'\bsk-[A-Za-z0-9_-]{24,}\b',rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}']
    rows=[];omitted=[];total=0
    for p in sorted(R.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        rel=p.relative_to(R)
        if p.name in ['server-config.json','SHARING.json'] or p.suffix in ['.o','.a','.lock']:
            omitted.append(str(rel));continue
        raw=p.read_bytes()
        if raw.startswith(b'\x7fELF'):omitted.append(str(rel));continue
        assert len(raw)<50*1024**2 and p.suffix not in ['.gguf','.safetensors','.pem','.key'],rel
        assert key not in raw and not any(re.search(x,raw) for x in pats),'credential pattern: '+str(rel)
        shared=rel.with_name('guard-health.json') if p.name=='health.json' else rel
        out=D/shared;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw);total+=len(raw)
        rows.append(dict(original=str(p),shared=str(shared),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),byte_identical=True))
    assert total<512*1024**2
    (D/'RAW_MAPPING.json').write_text(json.dumps(rows,indent=2)+'\n');(D/'EXPORT_SCOPE.json').write_text(json.dumps(dict(omitted=omitted,secret_scan_passed=True,scope='No binary/model/private guard config. guard-health is a byte-identical rename; replay reconstructs per-run config from LAUNCH command/environment in a disposable directory.'),indent=2)+'\n')
    m={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'FILES_SHA256.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(files=len(m)+1,bytes=sum(x['bytes'] for x in m.values()),secret_scan_passed=True,directory=str(D))))
if __name__=='__main__':main()
