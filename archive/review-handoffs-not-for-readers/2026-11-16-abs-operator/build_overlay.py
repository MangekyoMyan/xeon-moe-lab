from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
E=R.parents[1]/'exploration-20261112'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base_path=E/'mx-prepack/release-candidate-a/BUILD.json';base=json.loads(base_path.read_text())
    runtime=R/'runtime';runtime.mkdir(exist_ok=True)
    commands=[]
    cpp=base['cpp'].copy();cpp[1:1]=['-I'+str(R),'-I'+str(E/'row-local')]
    cpp[cpp.index('-c')+1]=str(R/'iq-abs.cpp');cpp[cpp.index('-o')+1]=str(R/'iq-abs.o')
    subprocess.run(cpp,check=True,timeout=180);commands.append(cpp)
    original=base['variants']['candidate'];cc=original['compile'].copy()
    old_object=cc[cc.index('-o')+1];cc.insert(1,'-I'+str(R))
    cc[cc.index('-c')+1]=str(R/'ggml-cpu.c');cc[cc.index('-o')+1]=str(R/'ggml-cpu.c.o')
    subprocess.run(cc,check=True,timeout=180);commands.append(cc)
    info=original['link'];link=[str(R/'ggml-cpu.c.o') if x==old_object else x for x in info['argv']]
    library=runtime/'libggml-cpu.so.0.28.0';link[link.index('-o')+1]=str(library.with_suffix('.new'))
    link=[('-Wl,--dependency-file='+str(R/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
    link += [str(R/'iq-abs.o'),'-Wl,--no-undefined']
    subprocess.run(link,cwd=info['cwd'],check=True,timeout=120);library.with_suffix('.new').replace(library)
    for name in ['libggml-cpu.so.0','libggml-cpu.so']:
        p=runtime/name
        if not p.exists():p.symlink_to(library.name)
    result={'scope':'Experimental operator library only; production configuration not changed','baseline_build':str(base_path),'baseline_build_sha256':sha(base_path),'commands':commands,'link':{'cwd':info['cwd'],'argv':link},'library':str(library),'sha256':sha(library),'source_hashes':{p.name:sha(p) for p in [R/'ggml-cpu.c',R/'iq-abs.cpp',R/'iq-abs.h']}}
    (R/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print('Experimental CPU overlay built',result['sha256'])
if __name__=='__main__':main()
