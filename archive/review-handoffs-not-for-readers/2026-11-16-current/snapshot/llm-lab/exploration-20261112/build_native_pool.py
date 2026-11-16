"""Use ggml's pthread graph executor; preserve parallel cache loading and math objects."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent;OUT=R/'native-pool';BASE=R/'sparse-cache';MX=R/'mx-prepack/backend'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sanitize',action='store_true');a=parser.parse_args()
    suffix='-san' if a.sanitize else '';flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1'] if a.sanitize else []
    OUT.mkdir(exist_ok=True);out=OUT/('san' if a.sanitize else 'normal');out.mkdir(exist_ok=True)
    prior=json.loads((MX/'BUILD.json').read_text())['variants']['candidate'];commands=[]
    cpp=next(c.copy() for c in json.loads((BASE/'CPU_BUILD.json').read_text())['commands'] if isinstance(c,list) and '-c' in c and c[c.index('-c')+1]==str(BASE/'ggml-cpu.cpp'))
    inputs=[prior['compile'].copy(),cpp];objects={}
    for command in inputs:
        original=Path(command[command.index('-c')+1]);source=OUT/original.name;source.write_bytes(original.read_bytes())
        old_object=command[command.index('-o')+1];new_object=out/(original.name+'.o');objects[old_object]=str(new_object)
        assert '-DGGML_USE_OPENMP' in command and '-fopenmp' in command
        command=[x for x in command if x not in ['-DGGML_USE_OPENMP','-fopenmp']]
        command[command.index('-c')+1]=str(source);command[command.index('-o')+1]=str(new_object);command+=['-pthread',*flags]
        subprocess.run(command,check=True,timeout=180);commands.append(command)
    linkinfo=prior['link'];link=[objects.get(x,x) for x in linkinfo['argv']]
    runtime=out/'runtime';runtime.mkdir(exist_ok=True);library=runtime/'libggml-cpu.so.0.28.0';temporary=library.with_suffix('.new')
    link[link.index('-o')+1]=str(temporary)
    link=[('-Wl,--dependency-file='+str(out/'link.d')) if x.startswith('-Wl,--dependency-file=') else str(MX/('mx-rhs'+suffix+'.o')) if x==str(MX/'mx-rhs.o') else x for x in link]
    link+=['-pthread',*flags];subprocess.run(link,cwd=linkinfo['cwd'],check=True,timeout=120);temporary.replace(library)
    for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
        if not (runtime/alias).exists():(runtime/alias).symlink_to(library.name)
    test=json.loads((BASE/'TEST_BUILD.json').read_text())
    test=[str(OUT/'test-pool.cpp') if x==str(BASE/'test-sparse.cpp') else str(OUT/('test-pool'+suffix)) if x==str(BASE/'test-sparse') else x for x in test]
    test+=['-pthread','-ldl','-UNDEBUG',*flags];subprocess.run(test,check=True,timeout=180)
    result={'scope':'Pthread graph execution; OpenMP retained for cache/quantizer initialization; native affinity places master last; row-local forbidden',
            'compile':commands,'link':{'argv':link,'cwd':linkinfo['cwd']},'test_compile':test,'library':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'ggml-cpu.c',OUT/'ggml-cpu.cpp',OUT/'test-pool.cpp']}}
    (out/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print('Built isolated native graph executor'+suffix,flush=True)
if __name__=='__main__':main()
