#!/usr/bin/env python3
"""Raise an isolated cache budget ceiling; keep arithmetic and 64GiB reserve unchanged."""
from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent;OLD=LAB/'iq-cache-next-20261103'
OUT=R/'cache120';OUT.mkdir(exist_ok=True);RT=OUT/'runtime';RT.mkdir(exist_ok=True)
prior=json.loads((OLD/'BUILD.json').read_text());commands=[]
for row in prior['libraries']:
    assert hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()==row['sha256']
replacements={
 'iq3-cache.cpp': [('constexpr size_t cap = size_t(96) << 30;','constexpr size_t cap = size_t(128) << 30;')],
 'llama-model.cpp': [('mib > 98304','mib > 131072'),('budget must be 1..98304 MiB','budget must be 1..131072 MiB')],
}
objects={}
for name,changes in replacements.items():
    text=(OLD/name).read_text()
    for before,after in changes:
        assert before in text,(name,before);text=text.replace(before,after)
    new=OUT/name;new.write_text(text)
    matches=[cmd for cmd in prior['commands'] if isinstance(cmd,list) and '-c' in cmd and cmd[cmd.index('-c')+1]==str(OLD/name)]
    assert len(matches)==1,name
    cmd=matches[0].copy();cmd[cmd.index('-c')+1]=str(new);obj=OUT/(name+'.o');cmd[cmd.index('-o')+1]=str(obj)
    commands.append(cmd);subprocess.run(cmd,check=True,timeout=240)
    objects[str(OLD/(name+'.o'))]=str(obj)
for unit,target in [('ggml-cpu','libggml-cpu.so.0.28.0'),('llama','libllama.so.0.8.0')]:
    match=[cmd for cmd in prior['commands'] if isinstance(cmd,dict) and '-o' in cmd['argv']
           and Path(cmd['argv'][cmd['argv'].index('-o')+1]).name.startswith(target)]
    assert len(match)==1,(unit,len(match))
    link=match[0];argv=[objects.get(x,x) for x in link['argv']]
    argv[argv.index('-o')+1]=str(RT/target)
    argv=[('-Wl,--dependency-file='+str(OUT/(unit+'-link.d'))) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
    commands.append({'cwd':link['cwd'],'argv':argv});subprocess.run(argv,cwd=link['cwd'],check=True,timeout=90)
    for alias in ['lib'+unit+'.so','lib'+unit+'.so.0']:
        p=RT/alias
        if not p.exists():p.symlink_to(target)
record={'commands':commands,'changes':'Ceiling96->128GiB; trial requests120GiB; same arithmetic and64GiB reserve',
 'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.cpp')},
 'libraries':[{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in RT.glob('*.so.*') if not p.is_symlink()]}
(OUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n');print('Isolated cache budget overlay built')
