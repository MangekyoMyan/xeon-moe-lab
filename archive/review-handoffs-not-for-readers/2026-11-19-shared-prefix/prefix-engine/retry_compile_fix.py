from pathlib import Path
import hashlib,json,subprocess
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=json.loads((P/'BUILD.json').read_text())
assert j['arms']['control']['bit_identical_deployed'] and j['arms']['control']['bit_identical_object']
assert sha(j['arms']['control']['library'])==j['original_base']['sha256']
for p,h in j['original_link_inputs'].items():assert sha(p)==h
subprocess.run(['python3',str(P/'prepare.py')],check=True)
arm=j['arms']['candidate'];out=P/'candidate'
arm['result']=[]
for label,cmd in zip(['compile','link'],arm['commands']):
    with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=j['cwd'],stdout=f,stderr=subprocess.STDOUT,timeout=240)
    arm['result'].append(r.returncode)
    if r.returncode:break
for p in j['source_sha256']:j['source_sha256'][p]=sha(p)
j['compile_fix_receipt']=dict(prior='compile-fix-checkpoint/BUILD.json',script_sha256=sha(__file__),control_not_rebuilt=True)
if arm['result']==[0,0]:
    lib=out/'libggml-base.so.0.28.0';arm.update(library=str(lib),library_sha256=sha(lib),object_sha256=sha(out/'ggml-backend.cpp.o'))
    syms=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True);arm['stats_exported']=any(l.split()[-1]=='xeon_shared_prefix_stats_v1' for l in syms.splitlines() if l.split());assert arm['stats_exported']
(P/'BUILD.json').write_text(json.dumps(j,indent=2)+'\n')
print(json.dumps({a:{k:v for k,v in x.items() if k not in ['commands']} for a,x in j['arms'].items()}))
assert arm['result']==[0,0]
