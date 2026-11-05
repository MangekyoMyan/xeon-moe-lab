#!/usr/bin/env python3
"""Build isolated experiment overlays without modifying accepted objects/libraries."""
import hashlib, json, shlex, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent
E = R.parent / 'xeon-volta-engine'
B = E / 'build-cpu/ggml/src'
io = '--io' in sys.argv[1:]
gateup = '--gateup' in sys.argv[1:] or io
pair = '--pair' in sys.argv[1:] or gateup
mlp = '--mlp' in sys.argv[1:] or pair
runtime = R / ('runtime-io' if io else 'runtime-gateup' if gateup else 'runtime-pair' if pair else 'runtime-mlp' if mlp else 'runtime')
source = R / ('io-src' if io else 'gateup-src' if gateup else 'pair-src' if pair else 'mlp-src' if mlp else 'src')
build = R / ('build-io' if io else 'build-gateup' if gateup else 'build-pair' if pair else 'build-mlp' if mlp else 'build')
runtime.mkdir(exist_ok=True)
build.mkdir(exist_ok=True)
for src in (E/'runtime').iterdir():
    if src.name.startswith('libggml-cpu'):
        continue
    dst = runtime/src.name
    if not dst.exists(): dst.symlink_to(src)
commands = []
def run(cmd, cwd=R):
    commands.append({'argv': list(map(str,cmd)), 'cwd': str(cwd)})
    print(shlex.join(list(map(str,cmd))), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

flags = {}
for line in (B/'CMakeFiles/ggml-cpu.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1); flags[k]=shlex.split(v)
obj=build/'ggml-cpu.c.o'
run(['/usr/bin/cc',*flags['C_DEFINES'],*flags['C_INCLUDES'],*flags['C_FLAGS'],'-c',source/'ggml-cpu.c','-o',obj])
if mlp:
    run(['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',source/'xv-bridge.cpp','-o',build/'xv-bridge.cpp.o'])
cmd=shlex.split((B/'CMakeFiles/ggml-cpu.dir/link.txt').read_text())
cmd[cmd.index('-o')+1]=str(runtime/'libggml-cpu.so.0.25.3')
cmd=[str(obj) if x=='CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o' else
     str(build/'xv-bridge.cpp.o') if mlp and x=='CMakeFiles/ggml-cpu.dir/ggml-cpu/xv-bridge.cpp.o' else
     '-Wl,--dependency-file='+str(build/'link.d') if x.startswith('-Wl,--dependency-file=') else x for x in cmd]
run(cmd,B)
for name,target in [('libggml-cpu.so','libggml-cpu.so.0'),('libggml-cpu.so.0','libggml-cpu.so.0.25.3')]:
    p=runtime/name
    if not p.exists(): p.symlink_to(target)
run(['g++','-std=c++17','-O2','-march=ivybridge','-fPIC','-shared','-Wall','-Wextra','-Werror','-pthread',
     '-I'+str(E/'source/ggml/include'),'-I'+str(source),source/'expert-cache.cpp',
     '-L'+str(runtime),'-lggml-cuda','-lggml-base','-o',runtime/'libxv-experiment.so'])
manifest={'commands':commands,'sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [source/'ggml-cpu.c',source/'expert-cache.cpp',source/'xv-bridge.cpp',source/'xv-bridge.h',obj,runtime/'libggml-cpu.so.0.25.3',runtime/'libxv-experiment.so']}}
(build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
