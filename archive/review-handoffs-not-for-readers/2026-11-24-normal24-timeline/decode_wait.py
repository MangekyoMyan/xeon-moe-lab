"""Resolve stripped OpenMP PCs against the exact library's Debian debug ELF."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from decimal import Decimal

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"runs/observe"
SO=Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")
DEBUG=ROOT/"private/libgomp.debug"
BUILD="abac9c46bfe169c31d97018db81b5386bdf38a17"


def id_for(p):
    out=subprocess.check_output(["readelf","-n",str(p)],text=True,stderr=subprocess.DEVNULL)
    return re.search(r"Build ID: (\w+)",out)[1]


def decode():
    assert id_for(SO)==id_for(DEBUG)==BUILD
    st=SO.stat()
    all_offsets=set();retained=[]
    for c in (0,1):
        data=RUN/f"pc-{c}.local.data"
        raw=subprocess.check_output(["/usr/bin/perf","script","--show-mmap-events","--ns","-i",str(data),
                    "-F","pid,tid,cpu,time,ip,sym,dso"],text=True,timeout=90)
        rx=r"PERF_RECORD_MMAP2 (\d+)/\d+: \[(0x[\da-f]+)\((0x[\da-f]+)\) @ (0x[\da-f]+) ([0-9a-f]+):([0-9a-f]+) (\d+) [^\]]+\]: r-xp "+re.escape(str(SO))
        mappings=set(tuple(int(x,b) for x,b in zip(m,(10,16,16,16,16,16,10))) for m in re.findall(rx,raw))
        assert len(mappings)==1,("mapping",mappings)
        pid,base,length,off,major,minor,ino=mappings.pop()
        assert (major,minor,ino)==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)
        # Translate the sample's file offset through the executable ELF segment.
        ph=subprocess.check_output(["readelf","-lW",str(SO)],text=True).splitlines()
        segments=[s.split() for s in ph if s.lstrip().startswith("LOAD ") and " E " in s]
        assert len(segments)==1
        seg_file,seg_va,seg_len=(int(segments[0][i],16) for i in (1,2,4))
        rx=re.compile(r"^\s*(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):\s+([0-9a-fA-F]+)\s+(.*?)\s+\((.*)\)\s*$")
        sample=[]
        for line in (RUN/f"pc-{c}.local.txt").read_text().splitlines():
            if not line.strip():continue
            m=rx.fullmatch(line);assert m and int(m[1])==pid
            _,tid,cpu,t,addr,sym,dso=m.groups();rel=None
            if dso==str(SO):
                actual=int(addr,16);assert base<=actual<base+length
                fileoff=actual-base+off
                assert seg_file<=fileoff<seg_file+seg_len
                rel=fileoff-seg_file+seg_va
                all_offsets.add(rel)
            sample.append([int(Decimal(t)*1_000_000_000),int(tid),int(cpu),sym,Path(dso).name,rel])
        retained.append(sample)
    order=sorted(all_offsets)
    cmd=["/usr/bin/addr2line","-a","-f","-C","-i","-e",str(DEBUG)]
    result=subprocess.run(cmd,input="".join(f"{x:x}\n" for x in order),text=True,capture_output=True,check=True).stdout
    groups=re.split(r"(?m)^0x([0-9a-f]+)\n",result);assert not groups[0].strip()
    symbols={}
    for raw,frames in zip(groups[1::2],groups[2::2]):
        lines=frames.rstrip("\n").splitlines();assert len(lines)%2==0
        symbols[int(raw,16)]=list(zip(lines[::2],lines[1::2]))
    assert set(symbols)==set(order)
    study=json.loads((RUN/"RUN.json").read_text());cases={}
    for c,(b,sample) in enumerate(zip(study["blocks"],retained)):
        clocks=list(zip(b["start_ns"][4:],b["output_ready_ns"][4:]))
        counts=collections.Counter();bycpu=collections.defaultdict(collections.Counter)
        byoff=collections.Counter();numeric_tids=set();total=0
        with (RUN/f"gomp-relative-{c}.tsv").open("w") as f:
            relative=csv.writer(f,delimiter="\t",lineterminator="\n")
            relative.writerow(("monotonic_ns","tid","cpu","relative_elf_va"))
            for t,tid,cpu,sym,dso,off in sample:
                if not any(a<=t<z for a,z in clocks):continue
                total+=1
                if off is not None:
                    relative.writerow((t,tid,cpu,hex(off)))
                    stack=[p[0].lower() for p in symbols[off]];byoff[hex(off)]+=1
                    if any("spin" in s for s in stack) and any("barrier" in s for s in stack):label="OpenMP barrier spin"
                    elif any("barrier" in s for s in stack) and any("wait" in s for s in stack):label="OpenMP barrier wait code (outside named spin)"
                    elif any("wait" in s or "spin" in s for s in stack):label="OpenMP other wait/spin code"
                    else:label="OpenMP other"
                elif dso.startswith("libggml-cpu") and any(k in sym for k in ("compact_dot","mx_rhs","mxfp4_rhs","vec_dot","tiled_run","quantize_row")):
                    label="CPU numerical kernels";numeric_tids.add(tid)
                elif dso.startswith("libggml-cpu"):label="other CPU backend"
                else:label="other libraries"
                counts[label]+=1;bycpu[cpu][label]+=1
        cases[b["case_id"]]={"total_within_T1_clocks":total,"categories":counts,"percent":{k:100*v/total for k,v in counts.items()},
                 "numeric_worker_tids":len(numeric_tids),"per_cpu":bycpu,"gomp_elf_va_counts":byoff}
    data={"scope":"79 Hz CPU-clock user-mode PC samples in calls 5–64 only. On-CPU thread samples, not elapsed T1, sleep or directly removable critical-path time.",
          "build_id":BUILD,"mapped_device_inode_verified":True,"debug_sha256":hashlib.sha256(DEBUG.read_bytes()).hexdigest(),
          "library_sha256":hashlib.sha256(SO.read_bytes()).hexdigest(),"debug_source":"https://debuginfod.debian.net/buildid/"+BUILD+"/debuginfo",
          "resolved_relative_elf_offsets":{hex(k):v for k,v in symbols.items()},"cases":cases}
    (ROOT/"CPU_SAMPLES.json").write_text(json.dumps(data,indent=2)+"\n")
    print([(k,v["percent"],v["numeric_worker_tids"]) for k,v in cases.items()])


if __name__=="__main__":decode()
