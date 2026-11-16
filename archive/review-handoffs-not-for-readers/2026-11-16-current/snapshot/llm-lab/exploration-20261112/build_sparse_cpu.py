from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'iq-cache-next-20261103'
OUT = ROOT/'sparse-cache'
RUNTIME = OUT/'runtime'
RUNTIME.mkdir(exist_ok=True)
prior = json.loads((OLD/'BUILD.json').read_text())
commands = []
objects = {}
for name in ['ggml-cpu.c','ggml-cpu.cpp','iq3-cache.cpp']:
    matches = [cmd for cmd in prior['commands'] if isinstance(cmd,list) and '-c' in cmd and cmd[cmd.index('-c')+1] == str(OLD/name)]
    assert len(matches) == 1
    cmd = matches[0].copy()
    cmd[cmd.index('-c')+1] = str(OUT/name)
    cmd[cmd.index('-o')+1] = str(OUT/(name+'.o'))
    cmd.insert(1,'-I'+str(OUT))
    commands.append(cmd);subprocess.run(cmd,check=True,timeout=180)
    objects[str(OLD/(name+'.o'))] = str(OUT/(name+'.o'))
links = [cmd for cmd in prior['commands'] if isinstance(cmd,dict) and '-o' in cmd['argv'] and Path(cmd['argv'][cmd['argv'].index('-o')+1]).name.startswith('libggml-cpu.so.0.28.0')]
assert len(links) == 1
link = links[0];argv = [objects.get(x,x) for x in link['argv']]
argv[argv.index('-o')+1] = str(RUNTIME/'libggml-cpu.so.0.28.0')
argv = [('-Wl,--dependency-file='+str(OUT/'cpu-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
commands.append({'cwd':link['cwd'],'argv':argv});subprocess.run(argv,cwd=link['cwd'],check=True,timeout=120)
for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
    if not (RUNTIME/alias).exists():(RUNTIME/alias).symlink_to('libggml-cpu.so.0.28.0')
(OUT/'CPU_BUILD.json').write_text(json.dumps({'commands':commands,
    'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.suffix in ['.c','.cpp','.h','.inc']},
    'library':{'file':str(RUNTIME/'libggml-cpu.so.0.28.0'),'sha256':hashlib.sha256((RUNTIME/'libggml-cpu.so.0.28.0').read_bytes()).hexdigest()}},indent=2)+'\n')
print('Sparse CPU overlay built; no model-loader or production changes yet.')
