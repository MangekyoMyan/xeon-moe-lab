#!/usr/bin/env python3
"""Warm the existing model's file cache sequentially; no model inference."""
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent
files=sorted((root.parent/'models/ggml-org--GLM-4.5-Air-GGUF').glob('GLM-4.5-Air-Q8_0-*.gguf'))
assert len(files)==2
buffer=bytearray(8*1024**2)
start=time.monotonic(); total=0; last=start
record={'files':[{"path":str(p),"bytes":p.stat().st_size} for p in files],'samples':[]}
for path in files:
    with path.open('rb',buffering=0) as f:
        while True:
            n=f.readinto(buffer)
            if not n:break
            total+=n; now=time.monotonic()
            if now-last>=30:
                row={'seconds':now-start,'bytes_read':total}
                record['samples'].append(row)
                (root/'preload-state.json').write_text(json.dumps(record,indent=2)+'\n')
                print('Preload GiB',round(total/2**30,2),'elapsed_s',round(now-start),flush=True)
                last=now
record.update(completed=True,bytes_read=total,elapsed_s=time.monotonic()-start)
(root/'preload-state.json').write_text(json.dumps(record,indent=2)+'\n')
print('Preload complete',total,flush=True)
