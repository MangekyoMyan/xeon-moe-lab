"""Publish the review evidence. The model, authentication material and ELF files stay local."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CHECKOUT=ROOT.parent.parent/"xeon-moe-review-20261116"
TARGET=CHECKOUT/"archive/review-handoffs-not-for-readers/2026-11-19-runtime-simplification"
CURRENT=CHECKOUT/"current"
PRIVATE_NAMES={"api-key.txt","server-config.json","README.template","PUBLIC_REPLAY.json","SHARING.json"}
PRIVATE_DIRS={"__pycache__","private-profiles"}
PATTERNS=[rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',rb'\bhf_[A-Za-z0-9]{24,}\b',rb'\bsk-[A-Za-z0-9_-]{24,}\b',rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}']

def scan(raw, key, path):
    if raw.startswith(b"\x7fELF") or path.suffix in {".a",".o",".so",".gguf",".safetensors",".key",".pem"}:
        return False
    assert len(raw)<50*1024**2 and key not in raw and not any(re.search(p,raw) for p in PATTERNS),path
    return True

def publish():
    assert not TARGET.exists() and not CURRENT.exists()
    assert json.loads((ROOT/"STATE.json").read_text())["completed"]
    key=(ROOT.parent/"api-server-20261030/api-key.txt").read_bytes().strip()
    assert len(key)>16
    mapping=[]
    for p in sorted(ROOT.rglob("*")):
        rel=p.relative_to(ROOT)
        if not p.is_file() or p.is_symlink() or PRIVATE_DIRS.intersection(rel.parts) or p.name in PRIVATE_NAMES or p.suffix==".lock":continue
        if rel.parts[0]=="received":
            # Keep the two new recommendations and decisions. Integrity of the entire input ZIP was verified on intake.
            if len(rel.parts)!=2 or p.name not in {"MAIN_SIMPLIFICATION_ECC9_JA.txt","CPU_UPGRADE_E52697V2_ADDENDUM_JA.txt","CURRENT_DECISIONS_ECC9.json","USER_REPORT_ECC9_JA.txt"}:continue
        if rel.parts[:2]==("build","restore-experiment"):continue
        raw=p.read_bytes()
        if not scan(raw,key,p):continue
        dest=rel.with_name("guard-health.json") if p.name=="health.json" else rel
        if p.name=="switch_runtime.py":dest=Path("unapplied/switch_runtime.py.txt")
        target=TARGET/dest
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        mapping.append({"file":str(dest),"local":str(rel),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
        if rel.parts[0]=="current":
            second=CURRENT/Path(*rel.parts[1:])
            second.parent.mkdir(parents=True,exist_ok=True)
            second.write_bytes(raw)
    (TARGET/"EXPORT_SCOPE.json").write_text(json.dumps({"entries":mapping,"prior_evidence_commit":"ecc9af41d42c031856b7dfcf52012c5450a52563","omissions":"Incoming duplicate historical evidence, ELF, GGUF, private per-run configs and deployment configs. Run configs reconstruct from LAUNCH. Model fingerprints are retained.","secret_scan_passed":True},indent=2)+"\n")
    manifest={}
    for p in sorted(TARGET.rglob("*")):
        if p.is_file():
            b=p.read_bytes()
            manifest[str(p.relative_to(TARGET))]={"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
    (TARGET/"FILES_SHA256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    assert sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file()) <512_000_000
    print(json.dumps({"archive_files":len(manifest)+1,"current_files":sum(p.is_file() for p in CURRENT.rglob("*")),"manifest_sha256":hashlib.sha256((TARGET/"FILES_SHA256.json").read_bytes()).hexdigest()}))
if __name__=="__main__":publish()
