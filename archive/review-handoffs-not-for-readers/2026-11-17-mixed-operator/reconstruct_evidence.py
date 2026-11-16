"""Restore exact saved mode0 expected arrays only; no model, code build or API call."""
from pathlib import Path
import gzip,hashlib,json
R=Path(__file__).resolve().parent
for digest,entry in json.loads((R/'correctness/INDEX.json').read_text()).items():
    data=gzip.decompress((R/entry['gzip']).read_bytes())
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==digest
    for name in entry['targets']:
        p=(R/name).resolve();assert p.is_relative_to(R.resolve())
        if p.exists():assert p.read_bytes()==data
        else:p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(data)
print('Restored all 24 exact expected output arrays; no model weights included')
