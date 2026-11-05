#!/usr/bin/env python3
"""Inspect, or explicitly redistribute, only the retained GGUF's file-cache pages.

No bytes are written to the model, no system NUMA policy is changed. Migration
uses per-page targets because mbind INTERLEAVE does not restripe resident pages.
Linux reference: https://man7.org/linux/man-pages/man2/move_pages.2.html
"""
import argparse,collections,ctypes,json,mmap,os,time
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--stripe',action='store_true',help='Explicitly migrate resident pages, only when no model inference runs')
ap.add_argument('--label',required=True)
args=ap.parse_args()
assert os.geteuid()==0,'Run as file owner/root; unprivileged mincore can report misleading residency'
assert '/' not in args.label and '..' not in args.label
out=R/(args.label+'.json');assert not out.exists()
P=R.parent/'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00002-of-00002.gguf'
assert P.stat().st_size==113602942912
page=os.sysconf('SC_PAGESIZE');assert page==4096 and os.uname().machine=='x86_64'
lib=ctypes.CDLL(None,use_errno=True)
lib.syscall.restype=ctypes.c_long
policy=ctypes.c_int();mask=ctypes.c_ulong()
assert lib.syscall(ctypes.c_long(239),ctypes.byref(policy),ctypes.byref(mask),ctypes.c_ulong(64),None,ctypes.c_ulong(0))==0
if args.stripe:
    assert policy.value==3 and mask.value==3,'Use numactl --interleave=0,1; default automatic NUMA balancing can undo explicit migration while this helper scans'
record={'pid':os.getpid(),'path':str(P),'stripe':args.stripe,'started_unix':time.time(),
        'file_stat_before':list(P.stat()),'chunks':[],'task_policy':policy.value,'task_nodemask':mask.value}
def query(addresses,nodes=None):
    count=len(addresses);ptrs=(ctypes.c_void_p*count)(*addresses);status=(ctypes.c_int*count)(*([-999]*count))
    targets=None if nodes is None else (ctypes.c_int*count)(*nodes)
    # Linux x86_64 SYS_move_pages=279. A NULL nodes array queries only.
    rc=lib.syscall(ctypes.c_long(279),ctypes.c_int(0),ctypes.c_ulong(count),ptrs,targets,status,ctypes.c_int(4 if nodes is not None else 0))
    if rc<0: raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return list(status),int(rc)
with P.open('rb') as f:
    m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY)
    base=ctypes.addressof(ctypes.c_char.from_buffer(m))
    size=len(m);count=(size+page-1)//page
    resident=(ctypes.c_ubyte*count)();assert lib.mincore(ctypes.c_void_p(base),ctypes.c_size_t(size),resident)==0
    record['resident_before_percent']=100*bytes(resident).count(1)/count
    assert record['resident_before_percent']>=99.9,'Do not mix migration with cold HDD loading'
    # Fault sample PTEs by reading; do not modify shared model bytes.
    sample_offsets=list(range(0,size,4*1024**2))
    checksum=sum(m[offset] for offset in sample_offsets)
    status,_=query([base+x for x in sample_offsets])
    groups=collections.defaultdict(collections.Counter)
    for offset,node in zip(sample_offsets,status):groups[offset//2**30][node]+=1
    record['before_samples_per_GiB']={str(k):dict(v) for k,v in groups.items()}
    record['sample_checksum']=checksum
    out.write_text(json.dumps(record,indent=2)+'\n')
    print('before sampled node counts',dict(collections.Counter(status)),flush=True)
    if args.stripe:
        started=time.monotonic();last=started
        # Bound scratch arrays and report/stop between64MiB batches.
        for first in range(0,count,16384):
            assert not (R/'STOP_REQUESTED').exists()
            n=min(16384,count-first);offsets=[(first+i)*page for i in range(n)]
            for offset in offsets: _=m[offset]
            addresses=[base+x for x in offsets]
            old,_=query(addresses)
            need=[i for i,node in enumerate(old) if node>=0 and node!=((first+i)&1)]
            before_errors=collections.Counter(x for x in old if x<0)
            if need:
                moved,rc=query([addresses[i] for i in need],[((first+i)&1) for i in need])
                errors=collections.Counter(x for x in moved if x<0)
                correct=sum(x==((first+i)&1) for i,x in zip(need,moved))
            else: errors={};rc=0;correct=0
            record['chunks'].append({'page':first,'pages':n,'requested':len(need),'migrated':correct,'nonmigrated':rc,'errors':dict(errors),'before_errors':dict(before_errors)})
            assert not before_errors,('Some pages were not resident',before_errors)
            assert not errors,('Migration incomplete; stop and inspect rather than silently claim striping',errors)
            if time.monotonic()-last>=10:
                print('striped GiB',round((first+n)*page/2**30,2),'elapsed_s',round(time.monotonic()-started),flush=True)
                out.write_text(json.dumps(record,indent=2)+'\n');last=time.monotonic()
        after,_=query([base+x for x in sample_offsets])
        record['after_sample_counts']=dict(collections.Counter(after))
        # Our4MiB-spaced sample offsets are all even pages; query their adjacent
        # pages as well, so success requires both target nodes, not just totals.
        verify_offsets=[x+y for x in sample_offsets for y in [0,page] if x+y<size]
        for x in verify_offsets: _=m[x]
        verification,_=query([base+x for x in verify_offsets])
        record['stripe_sample_mismatches']=sum(node!=((offset//page)&1) for offset,node in zip(verify_offsets,verification))
        record['verification_nodes']=dict(collections.Counter(verification))
        out.write_text(json.dumps(record,indent=2)+'\n')
        assert record['stripe_sample_mismatches']==0
    m.close()
record['file_stat_after']=list(P.stat())
assert record['file_stat_before'][6]==record['file_stat_after'][6] # size
assert record['file_stat_before'][8]==record['file_stat_after'][8] # mtime
record['completed']=True;record['elapsed_s']=time.time()-record['started_unix']
out.write_text(json.dumps(record,indent=2)+'\n')
owner=R.stat();os.chown(out,owner.st_uid,owner.st_gid)
print('Completed',out,flush=True)
