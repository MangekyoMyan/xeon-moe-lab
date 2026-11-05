#!/usr/bin/env python3
"""Read-only direct I/O samples from an existing model. Does not drop global caches."""
import fcntl
import json
import mmap
import os
from pathlib import Path
import shutil
import time

LAB=Path(__file__).resolve().parent.parent
OUT=LAB/'results/server-study-20261023'
MODEL_DIRECTORY=LAB/'models/ggml-org--GLM-4.5-Air-GGUF'
def main():
    with (LAB/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=max(MODEL_DIRECTORY.glob('*.gguf'),key=lambda p:p.stat().st_size)
        before=model.stat();chunk=8*1024*1024;span=1024**3
        assert before.st_size>3*span
        offsets=[0,((before.st_size-span)//2//chunk)*chunk,((before.st_size-span)//chunk)*chunk]
        results=[];fd=os.open(model,os.O_RDONLY|os.O_DIRECT|os.O_NOATIME)
        try:
            buffer=mmap.mmap(-1,chunk);view=memoryview(buffer)
            for label,offset in zip(['begin','middle','end'],offsets):
                print('Direct read sample: '+label,flush=True)
                started=time.monotonic();count=0
                for pos in range(offset,offset+span,chunk):
                    got=os.preadv(fd,[view],pos);assert got==chunk;count+=got
                elapsed=time.monotonic()-started
                row={'position':label,'offset_bytes':offset,'read_bytes':count,'seconds':elapsed,'MB_s':count/elapsed/1e6}
                results.append(row);print(json.dumps(row),flush=True)
            del view;buffer.close()
        finally:os.close(fd)
        after=model.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        usage=shutil.disk_usage(LAB);fs=os.statvfs(LAB)
        payload={'model':str(model),'file_size':before.st_size,'results':results,'read_only':True,
                 'O_DIRECT':True,'global_cache_drop':False,'mtime_unchanged':True,
                 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'filesystem':{'total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free,'available_unprivileged_bytes':fs.f_bavail*fs.f_frsize},
                 'scope':'Sequential reads within three existing file regions; not a whole-disk maximum, random-I/O test, or download benchmark.'}
        (OUT/'storage-read.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
