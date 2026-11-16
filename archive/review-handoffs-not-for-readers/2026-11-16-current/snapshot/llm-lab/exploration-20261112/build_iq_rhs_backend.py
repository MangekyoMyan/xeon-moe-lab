from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent;BASE=R/'sparse-cache';O=R/'iq-rhs';MX=R/'mx-prepack/backend'
def main():
 p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');a=p.parse_args();suffix='-san' if a.sanitize else '';extra=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1'] if a.sanitize else []
 prior=json.loads((BASE/'CPU_BUILD.json').read_text())['commands'];cpp=next(x.copy() for x in prior if isinstance(x,list) and '-c'in x and x[x.index('-c')+1]==str(BASE/'iq3-cache.cpp'));cpp.insert(1,'-I'+str(O));cpp[cpp.index('-c')+1]=str(O/'iq-rhs.cpp');cpp[cpp.index('-o')+1]=str(O/('iq-rhs'+suffix+'.o'));cpp+=extra;subprocess.run(cpp,check=True,timeout=180)
 variants={}
 for name,enable in [('control',0),('candidate',1)]:
  folder=O/(name+suffix);folder.mkdir(exist_ok=True);cmd=next(x.copy() for x in prior if isinstance(x,list) and '-c'in x and x[x.index('-c')+1]==str(BASE/'ggml-cpu.c'));cmd[1:1]=['-I'+str(O),'-I'+str(MX),'-DXEON_IQ_PREPACK='+str(enable)];cmd[cmd.index('-c')+1]=str(O/'ggml-cpu.c');cmd[cmd.index('-o')+1]=str(folder/'ggml-cpu.c.o');cmd+=extra;subprocess.run(cmd,check=True,timeout=180)
  info=json.loads((R/'row-local/BUILD.json').read_text())['link'];link=[str(folder/'ggml-cpu.c.o') if x==str(BASE/'ggml-cpu.c.o') else x for x in info['argv']];runtime=folder/'runtime';runtime.mkdir(exist_ok=True);lib=runtime/'libggml-cpu.so.0.28.0';temp=lib.with_suffix('.new');link[link.index('-o')+1]=str(temp);link=[('-Wl,--dependency-file='+str(folder/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link];link.extend([str(O/('iq-rhs'+suffix+'.o')),str(MX/('mx-rhs'+suffix+'.o')),*extra]);subprocess.run(link,cwd=info['cwd'],check=True,timeout=120);temp.replace(lib)
  for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
   if not(runtime/alias).exists():(runtime/alias).symlink_to(lib.name)
  variants[name]={'compile':cmd,'link':{'argv':link,'cwd':info['cwd']},'library':str(lib),'sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
 test=json.loads((BASE/'TEST_BUILD.json').read_text());test=[str(O/'test-graph.cpp') if x==str(BASE/'test-sparse.cpp') else str(O/('test-graph'+suffix)) if x==str(BASE/'test-sparse') else x for x in test];test.insert(1,'-I'+str(O));test+=['-ldl',*extra];subprocess.run(test,check=True,timeout=180)
 subprocess.run(['g++','-O2','-std=c++17','-fPIC','-shared',str(O/'trace-calls.cpp'),'-ldl','-o',str(O/'trace-calls.so')],check=True,timeout=120)
 record={'cpp':cpp,'test':test,'variants':variants,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.suffix in ['.h','.cpp','.inc','.c']}};(O/('BUILD'+suffix+'.json')).write_text(json.dumps(record,indent=2)+'\n');print('Built isolated IQ RHS control/candidate'+suffix)
if __name__=='__main__':main()
