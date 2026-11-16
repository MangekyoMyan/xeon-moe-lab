from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent;BASE=R/'sparse-cache';OUT=R/'mx-prepack/backend'
def main():
 p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');a=p.parse_args();suffix='-san' if a.sanitize else ''
 prior=json.loads((BASE/'CPU_BUILD.json').read_text())['commands']
 flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1'] if a.sanitize else []
 cpp=next(x.copy() for x in prior if isinstance(x,list) and '-c'in x and x[x.index('-c')+1]==str(BASE/'iq3-cache.cpp'))
 cpp.insert(1,'-I'+str(OUT));cpp[cpp.index('-c')+1]=str(OUT/'mx-rhs.cpp');cpp[cpp.index('-o')+1]=str(OUT/('mx-rhs'+suffix+'.o'));cpp+=flags
 subprocess.run(cpp,check=True,timeout=180)
 builds={}
 for label,enabled in [('control',0),('candidate',1)]:
  variant=OUT/(label+suffix);variant.mkdir(exist_ok=True)
  cmd=next(x.copy() for x in prior if isinstance(x,list) and '-c'in x and x[x.index('-c')+1]==str(BASE/'ggml-cpu.c'));cmd[1:1]=['-I'+str(OUT),'-DXEON_MX_PREPACK='+str(enabled)]
  cmd[cmd.index('-c')+1]=str(OUT/'ggml-cpu.c');cmd[cmd.index('-o')+1]=str(variant/'ggml-cpu.c.o');cmd+=flags;subprocess.run(cmd,check=True,timeout=180)
  info=json.loads((R/'row-local/BUILD.json').read_text())['link'];link=[str(variant/'ggml-cpu.c.o') if x==str(BASE/'ggml-cpu.c.o') else x for x in info['argv']]
  runtime=variant/'runtime';runtime.mkdir(exist_ok=True);lib=runtime/'libggml-cpu.so.0.28.0';temp=lib.with_suffix('.new');link[link.index('-o')+1]=str(temp);link=[('-Wl,--dependency-file='+str(variant/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link];link+=[str(OUT/('mx-rhs'+suffix+'.o')),*flags]
  subprocess.run(link,cwd=info['cwd'],check=True,timeout=120);temp.replace(lib)
  for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
   if not(runtime/alias).exists():(runtime/alias).symlink_to(lib.name)
  builds[label]={'compile':cmd,'link':{'argv':link,'cwd':info['cwd']},'library':str(lib),'sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
 (OUT/('BUILD'+suffix+'.json')).write_text(json.dumps({'cpp':cpp,'sources':json.loads((OUT/'SOURCE.json').read_text()),'variants':builds},indent=2)+'\n');print('Built isolated MXFP4 backend control/candidate'+suffix)
if __name__=='__main__':main()
