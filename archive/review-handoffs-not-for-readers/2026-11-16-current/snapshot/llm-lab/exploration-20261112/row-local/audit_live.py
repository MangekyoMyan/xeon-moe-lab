"""One-shot read-only physical-page query for the live private compact cache."""
from pathlib import Path
import bisect
import ctypes
import hashlib
import importlib.util
import json
import os
import time

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('helpers',ROOT.parent/'postcompact-20261103/live_probe.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main():
    pid=helpers.api_pid();cfg=ROOT.parent/'deepseek41-bringup-20261101/server-config.json';raw=cfg.read_bytes();config=json.loads(raw)
    start_ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    plan_lines=Path(config['environment']['GGML_CPU_IQ_PLAN']).read_text().splitlines()[1:]
    plan={parts[0]:tuple(map(int,parts[2:6])) for line in plan_lines if (parts:=line.split())}
    candidates=list((ROOT/'row-local/audit').glob(f'placement-{pid}-*'))
    assert len(candidates)==1,'Exactly one audit owner expected'
    entries=[json.loads(x) for x in candidates[0].read_text().splitlines()]
    assert len(entries)==80 and len({x['name'] for x in entries})==80
    assert sum(x['slots'] for x in entries)==19200 and sum(x['bytes'] for x in entries)==127401984000
    libc=ctypes.CDLL('libnuma.so.1',use_errno=True)
    call=libc.move_pages;call.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int];call.restype=ctypes.c_long
    expected_local=config['environment'].get('GGML_CPU_IQ_ROW_LOCAL')=='1'
    loaded=Path(f'/proc/{pid}/maps').read_text()
    expected_cpu=[x for x in config['study_libraries'] if 'libggml-cpu.' in x['file']]
    assert len(expected_cpu)==1 and expected_cpu[0]['file'] in loaded
    assert hashlib.sha256(Path(expected_cpu[0]['file']).read_bytes()).hexdigest()==expected_cpu[0]['sha256']
    policies={int(line.split()[0],16):line.split()[1] for line in Path(f'/proc/{pid}/numa_maps').read_text().splitlines()}
    starts=sorted(policies)
    counts={};checked=0;bound=0;expected_bound=0
    for e in entries:
        assert e['pid']==pid and (e['k'],e['rows'],384,e['slots'])==plan[e['name']]
        stride=e['bytes']//e['slots'];assert stride==6635520
        offsets=[0,4096,stride//2-4096,stride//2,stride//2+4096,stride-4096]
        addresses=[e['address']+slot*stride+off for slot in range(e['slots']) for off in offsets]
        ptrs=(ctypes.c_void_p*len(addresses))(*addresses);nodes=(ctypes.c_int*len(addresses))(*([-9999]*len(addresses)))
        result=call(pid,len(addresses),ptrs,None,nodes,0)
        if result!=0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
        eligible=e['k']==5120 and e['rows']==2304 and e['name'].endswith(('ffn_up_exps.weight','ffn_gate_exps.weight'))
        local=expected_local and eligible
        assert e['policy']==('row-halves-0-1' if local else 'inherit'),'Audit policy does not match configured eligible tensors'
        if local:expected_bound+=len(addresses)
        for i,address in enumerate(addresses):
            policy=policies[starts[bisect.bisect_right(starts,address)-1]]
            assert policy==('bind:'+str(0 if i%6<3 else 1) if local else 'interleave:0-1'),'Wrong live VMA policy'
        if not local:
            for i,node in enumerate(nodes):assert node==nodes[(i//6)*6]^((offsets[i%6]//4096)%2),'Control/inherited physical pages are not interleaved' 
        for i,node in enumerate(nodes):
            assert node in [0,1],f'Nonresident/error page {node}'
            counts[str(node)]=counts.get(str(node),0)+1
            if local:assert node==(0 if i%6<3 else 1),'Wrong physical NUMA node'
        checked+=len(addresses)
        if local:bound+=len(addresses)
    assert bound==expected_bound and (bound>0)==expected_local
    assert cfg.read_bytes()==raw
    assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==start_ticks
    status=Path(f'/proc/{pid}/status').read_text();assert 'THP_enabled:\t0' in status
    maps=Path(f'/proc/{pid}/maps').read_text().splitlines()
    smaps=Path(f'/proc/{pid}/smaps').read_text().splitlines()
    huge_kb=sum(int(x.split()[1]) for x in smaps if x.startswith('AnonHugePages:'))
    assert huge_kb==0
    out={'pid':pid,'config_sha256':hashlib.sha256(raw).hexdigest(),'profile':config['optimization_study']['profile'],
         'source_audit':str(candidates[0]),'entries':len(entries),'selected_projections':19200,'payload_bytes':127401984000,
         'row_local_expected':expected_local,'physical_pages_queried':checked,'bound_page_queries':bound,'node_sample_counts':counts,'anon_huge_kib':huge_kb,
         'vmas':len(maps),'vm_max_map_count':int(Path('/proc/sys/vm/max_map_count').read_text()),
         'mem_available_kib':next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
         'pass':True,'unix':time.time()}
    folder=ROOT/'row-local/audit-results';folder.mkdir(exist_ok=True,mode=0o700)
    target=folder/f'{time.strftime("%Y%m%d-%H%M%S")}-{out["profile"]}.json'
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    if os.geteuid()==0:os.fchown(fd,ROOT.stat().st_uid,ROOT.stat().st_gid)
    with os.fdopen(fd,'w') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
