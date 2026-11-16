from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parent;manifest=json.loads((R/'BUILD.json').read_text());out=R/'san';out.mkdir(exist_ok=True)
replacements={}
for cmd in manifest['commands'][:3]:
    cmd=list(cmd);old=cmd[cmd.index('-o')+1];new=str(out/Path(old).name);replacements[old]=new;cmd[cmd.index('-o')+1]=new
    cmd=[v for v in cmd if v!='-O3'];cmd+=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    subprocess.run(cmd,check=True,timeout=180)
rec=manifest['commands'][3];cmd=[replacements.get(v,v) for v in rec['argv']]
cmd[cmd.index('-o')+1]=str(out/'libggml-cpu.so.0.28.0');cmd+=['-fsanitize=address,undefined']
cmd=[('-Wl,--dependency-file='+str(out/'link.d')) if v.startswith('-Wl,--dependency-file=') else v for v in cmd]
subprocess.run(cmd,cwd=rec['cwd'],check=True,timeout=60)
for name in ['libggml-cpu.so','libggml-cpu.so.0']:
    p=out/name
    if not p.exists():p.symlink_to('libggml-cpu.so.0.28.0')
cmd=json.loads((R/'TEST_BUILD.json').read_text());cmd=[v for v in cmd if v!='-O3']
cmd=[v.replace(str(R/'runtime'),str(out)) for v in cmd];cmd[cmd.index('-o')+1]=str(out/'test-cache');cmd+=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
subprocess.run(cmd,check=True,timeout=90)
print('Sanitized modified CPU code and graph test built',flush=True)
