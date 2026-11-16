from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parent;E=R.parent/'deepseek41-bringup-20261101';B=E/'build/tools/fit-params';O=R/'gpu-fit'
flags={}
for line in (B/'CMakeFiles/llama-fit-params.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_') and ' = 'in line:
  key,value=line.split(' = ',1);flags[key]=shlex.split(value)
cmd=['g++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-std=c++17','-c',str(O/'estimate.cpp'),'-o',str(O/'estimate.o')]
subprocess.run(cmd,check=True,timeout=180)
link=shlex.split((B/'CMakeFiles/llama-fit-params.dir/link.txt').read_text());link[link.index('CMakeFiles/llama-fit-params.dir/main.cpp.o')]=str(O/'estimate.o');link[link.index('-o')+1]=str(O/'estimate');link.remove('../../bin/libllama-fit-params-impl.so');link=[('-Wl,--dependency-file='+str(O/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
subprocess.run(link,cwd=B,check=True,timeout=120)
(O/'BUILD.json').write_text(json.dumps({'compile':cmd,'link':link,'cwd':str(B),'source_sha256':hashlib.sha256((O/'estimate.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((O/'estimate').read_bytes()).hexdigest()},indent=2)+'\n');print('Built metadata-only text memory estimator')
