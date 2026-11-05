#!/usr/bin/env python3
"""Check archive integrity and likely credential leaks without printing matched values."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',
 'huggingface_token':rb'\bhf_[A-Za-z0-9]{24,}\b',
 'api_token':rb'\bsk-[A-Za-z0-9_-]{24,}\b',
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'literal_bearer':rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}',
}
def files():
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in ['.git','__pycache__'] for x in rel.parts) or rel.parts[0] in ['work','dist','runtime-local']:continue
        if p.is_file() or p.is_symlink():yield p,rel
def main():
    findings=[];total=0;count=0
    for p,rel in files():
        count+=1
        if p.is_symlink():findings.append({'file':str(rel),'reason':'unexpected source symlink'});continue
        raw=p.read_bytes();total+=len(raw)
        if len(raw)>=50*1024**2:findings.append({'file':str(rel),'reason':'large Git blob'})
        if p.suffix.lower() in ['.gguf','.safetensors','.pem','.key','.p12','.pfx','.so','.o','.a'] or p.name in ['hosts.yml','.env','id_rsa','id_ed25519','server-config.json','health.json']:
            findings.append({'file':str(rel),'reason':'excluded file type/name'})
        for kind,pattern in PATTERNS.items():
            for match in re.finditer(pattern,raw):
                findings.append({'file':str(rel),'line':raw[:match.start()].count(b'\n')+1,'reason':kind})
    m=json.loads((ROOT/'provenance/EXPORT_MANIFEST.json').read_text());matched=0
    for row in m['files']:
        p=ROOT/row['archive']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:findings.append({'file':row['archive'],'reason':'copied content hash mismatch'})
        else:matched+=1
    result={'files_checked':count,'bytes':total,'copied_source_hashes_verified':matched,'findings':findings,
            'note':'Pattern and manifest checks; actual local API key comparison is performed separately before upload. No matched secret values are printed.'}
    print(json.dumps(result,indent=2));return bool(findings)
if __name__=='__main__':sys.exit(main())
