"""Prove reused objects/headers reproduce the deployed baseline before candidate use."""
from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent;OLD=R.parent/'iq-cache-next-20261103';OUT=R/'cache120/control';OUT.mkdir(parents=True,exist_ok=True)
prior=json.loads((OLD/'BUILD.json').read_text());rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['iq3-cache.cpp','llama-model.cpp']:
    cmd=next(c.copy() for c in prior['commands'] if isinstance(c,list) and '-c' in c and c[c.index('-c')+1]==str(OLD/name))
    original=cmd[cmd.index('-o')+1];control=OUT/(name+'.o');cmd[cmd.index('-o')+1]=str(control)
    subprocess.run(cmd,check=True,timeout=240)
    rows.append({'kind':'unchanged_source_recompiled','file':name,'baseline':sha(original),'control':sha(control),'same':sha(original)==sha(control)})
for name in ['libggml-cpu.so.0.28.0','libllama.so.0.8.0']:
    link=next(c for c in prior['commands'] if isinstance(c,dict) and '-o' in c['argv'] and Path(c['argv'][c['argv'].index('-o')+1]).name.startswith(name))
    argv=link['argv'].copy();argv[argv.index('-o')+1]=str(OUT/name)
    argv=[('-Wl,--dependency-file='+str(OUT/(name+'.d'))) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
    subprocess.run(argv,cwd=link['cwd'],check=True,timeout=90)
    rows.append({'kind':'baseline_relinked','file':name,'baseline':sha(OLD/'runtime'/name),'control':sha(OUT/name),'same':sha(OLD/'runtime'/name)==sha(OUT/name)})
result={'checks':rows,'passed':all(x['same'] for x in rows)}
(OUT/'CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit('Control differs: inspect before deployment')
