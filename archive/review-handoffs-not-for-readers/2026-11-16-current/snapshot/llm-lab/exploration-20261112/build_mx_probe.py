from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'mx-prepack'
old=json.loads((ROOT.parent/'optimization-20261112/cpu-bias/BUILD.json').read_text())['command']
cmd=[]
for part in old:
    if part.endswith('/cpu-bias/test-bias.cpp'):part=str(OUT/'probe.cpp')
    elif part.endswith('/cpu-bias/test-bias'):part=str(OUT/'probe')
    cmd.append(part)
idx=cmd.index('-lggml-cpu');cmd[idx:idx]=['-L'+str(ROOT/'sparse-cache/release-v2-a'),'-Wl,-rpath,'+str(ROOT/'sparse-cache/release-v2-a')]
# Ensure the frozen study CPU library has precedence over the original base build.
cmd=[x for x in cmd if not x.startswith('-Wl,-rpath,')]+['-Wl,-rpath,'+str(ROOT/'sparse-cache/release-v2-a')+':'+str(ROOT.parent/'gpu-stream-20261103/runtime')+':'+str(ROOT.parent/'deepseek41-bringup-20261101/build/bin'),'-ldl']
subprocess.run(cmd,check=True,timeout=180)
record={'command':cmd,'sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['probe.cpp','kernel.inc']},'binary_sha256':hashlib.sha256((OUT/'probe').read_bytes()).hexdigest(),'reference_library':str(ROOT/'sparse-cache/release-v2-a/libggml-cpu.so.0.28.0'),'reference_sha256':hashlib.sha256((ROOT/'sparse-cache/release-v2-a/libggml-cpu.so.0.28.0').read_bytes()).hexdigest()}
(OUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
print('Built standalone MXFP4 probe; no API or source engine changes')
