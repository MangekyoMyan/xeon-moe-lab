"""Prepare a ZIP and an identical, browsable tree of the comparison evidence."""
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CHECKOUT=ROOT.parent.parent/"xeon-moe-review-20261116"
ARCHIVE="2026-11-23-preupgrade-baseline"
TARGET=CHECKOUT/"archive/review-handoffs-not-for-readers"/ARCHIVE
DELIVER=ROOT/"deliverables"
EXCLUDES={"local-frozen","deliverables","__pycache__",".git","runtime"}
PRIVATE_NAMES={"server-config.json","worker.local.data","quality-host-pcs.txt","quality-host-pcs-with-symbols.local.txt","SHARING.json","PUBLIC_REPLAY.json",".study.lock"}
SECRET_PATTERNS=[rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',rb'\bhf_[A-Za-z0-9]{24,}\b',rb'\bsk-[A-Za-z0-9_-]{24,}\b',
                 rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',rb'Bearer[ \t]+[A-Za-z0-9._~-]{24,}']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_text())

def main():
    assert not TARGET.exists()
    assert read(ROOT/"RESULT_INDEX.json")["restored_normal_api"]
    key=(ROOT.parent/"api-server-20261030/api-key.txt").read_bytes().strip()
    assert len(key)>16
    DELIVER.mkdir(exist_ok=True)
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TARGET.parent) as tmp:
        stage=Path(tmp)/ARCHIVE;stage.mkdir()
        manifest={}
        for p in sorted(ROOT.rglob("*")):
            relative=p.relative_to(ROOT)
            if p.is_symlink() or not p.is_file() or EXCLUDES.intersection(relative.parts):continue
            if p.name in PRIVATE_NAMES or p.suffix in (".pyc",".so",".o",".a",".key",".pem",".gguf",".zip"):continue
            if relative.parts[:2]==("intake","tests") or relative.parts[:2]==("intake","reviews"):continue
            raw=p.read_bytes()
            if raw.startswith(b"\x7fELF"):continue
            assert len(raw)<49*1024**2,relative
            assert key not in raw and not any(re.search(pattern,raw) for pattern in SECRET_PATTERNS),f"Credential pattern in {relative}"
            shared=relative.with_name("guard-health.json") if p.name=="health.json" else relative
            if shared==Path("intake/MANIFEST_SHA256.json"):shared=Path("intake/ORIGINAL_KIT_MANIFEST.json")
            dest=stage/shared;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            manifest[shared.as_posix()]={"bytes":len(raw),"sha256":sha(raw)}
        (stage/"FILES_SHA256.json").write_text(json.dumps(manifest,indent=2)+"\n")
        stage.rename(TARGET)
    entries=[p for p in sorted(TARGET.rglob("*")) if p.is_file()]
    output=DELIVER/"xeon-moe-preupgrade-baseline-20261123.zip"
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as archive:
        for p in entries:
            archive.write(p,str(Path(ARCHIVE)/p.relative_to(TARGET)))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist()==[str(Path(ARCHIVE)/p.relative_to(TARGET)) for p in entries]
        for p in entries:
            assert archive.read(str(Path(ARCHIVE)/p.relative_to(TARGET)))==p.read_bytes()
    assert output.stat().st_size<512_000_000
    h=hashlib.sha256(output.read_bytes()).hexdigest()
    (DELIVER/(output.name+".sha256")).write_text(f"{h}  {output.name}\n")
    record={"zip":str(output),"sha256":h,"zip_bytes":output.stat().st_size,"files":len(entries),
            "uncompressed_bytes":sum(p.stat().st_size for p in entries),"archive_content_verified":True,"secret_screen_passed":True,
            "manifest_sha256":sha((TARGET/"FILES_SHA256.json").read_bytes()),"browsable_directory":str(TARGET)}
    (DELIVER/"PACKAGE.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))
if __name__=="__main__":main()
