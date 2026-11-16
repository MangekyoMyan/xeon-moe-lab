from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OLD = LAB / 'iq-cache-next-20261103'
OUT = ROOT / 'census'
RUNTIME = OUT / 'runtime'
RUNTIME.mkdir(exist_ok=True)
if (OUT/'BUILD.json').exists():
    previous = (OUT/'BUILD.json').read_bytes()
    archive = OUT/('BUILD-'+hashlib.sha256(previous).hexdigest()[:12]+'.json')
    if not archive.exists():archive.write_bytes(previous)
prior = json.loads((OLD/'BUILD.json').read_text())
source = (OLD/'ggml-cpu.c').read_text()
needle = '''                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk'''
replace = '''                matrix_row_counts[i02] += 1;
            }
        }
        xeon_moe_census_v1(src0, ids, matrix_row_counts, n_as);
    }

    // reset current_chunk'''
assert source.count(needle) == 1
source = '#include "census.h"\n' + source.replace(needle, replace)
(OUT/'ggml-cpu.c').write_text(source)
commands = []
objects = {}
for name in ['ggml-cpu.c']:
    matches = [cmd for cmd in prior['commands'] if isinstance(cmd,list) and '-c' in cmd and cmd[cmd.index('-c')+1] == str(OLD/name)]
    assert len(matches) == 1
    cmd = matches[0].copy()
    cmd[cmd.index('-c')+1] = str(OUT/name)
    cmd[cmd.index('-o')+1] = str(OUT/(name+'.o'))
    commands.append(cmd)
    subprocess.run(cmd,check=True,timeout=180)
    objects[str(OLD/(name+'.o'))] = str(OUT/(name+'.o'))
include = LAB/'deepseek41-bringup-20261101/source/ggml/include'
cmd = ['g++','-O3','-std=c++17','-fPIC','-I'+str(include),'-c',str(OUT/'census.cpp'),'-o',str(OUT/'census.cpp.o')]
commands.append(cmd);subprocess.run(cmd,check=True,timeout=120)
objects[str(OLD/'iq3-cache.cpp.o')] = str(LAB/'optimization-20261112/cache120/iq3-cache.cpp.o')
links = [cmd for cmd in prior['commands'] if isinstance(cmd,dict) and '-o' in cmd['argv'] and Path(cmd['argv'][cmd['argv'].index('-o')+1]).name.startswith('libggml-cpu.so.0.28.0')]
assert len(links) == 1
link = links[0];argv = [objects.get(x,x) for x in link['argv']]
argv[argv.index('-o')+1] = str(RUNTIME/'libggml-cpu.so.0.28.0')
argv = [('-Wl,--dependency-file='+str(OUT/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
argv.append(str(OUT/'census.cpp.o'))
commands.append({'cwd':link['cwd'],'argv':argv});subprocess.run(argv,cwd=link['cwd'],check=True,timeout=120)
for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
    target = RUNTIME/alias
    if not target.exists():target.symlink_to('libggml-cpu.so.0.28.0')
cmd = ['g++','-O2','-std=c++17','-pthread','-I'+str(include),str(OUT/'test.cpp'),str(OUT/'census.cpp'),'-o',str(OUT/'test')]
commands.append(cmd);subprocess.run(cmd,check=True,timeout=120)
revision = hashlib.sha256((OUT/'census.cpp').read_bytes()).hexdigest()[:12]
testfile = OUT/('unit-counters-'+revision+'.bin')
env = dict(os.environ,XEON_MOE_CENSUS_PATH=str(testfile))
assert not testfile.exists(), 'Use a new output path for a second build/test; preserve existing evidence'
subprocess.run([str(OUT/'test')],env=env,check=True,timeout=30)
spec = importlib.util.spec_from_file_location('reader',OUT/'read_census.py');reader = importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
result = reader.read(testfile)
assert len(result['records']) == 1
r = result['records'][0]
assert r['calls'] == 800 and r['assignments'] == 4800
assert r['counts'][0] == 1600 and r['counts'][17] == 800 and r['counts'][383] == 2400
(OUT/('UNIT_RESULT-'+revision+'.json')).write_text(json.dumps(result,indent=2)+'\n')
record = {'commands':commands,'unit_test_passed':True,'library':{'file':str(RUNTIME/'libggml-cpu.so.0.28.0'),'sha256':hashlib.sha256((RUNTIME/'libggml-cpu.so.0.28.0').read_bytes()).hexdigest()},'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.suffix in ['.c','.cpp','.h','.py']}}
(OUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
print('Built optional counter overlay and passed concurrent aggregate-counter test.')
