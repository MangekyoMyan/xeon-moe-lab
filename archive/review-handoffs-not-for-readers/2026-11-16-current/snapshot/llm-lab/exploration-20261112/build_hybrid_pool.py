"""Compile a private mixed executor without replacing a mapped library."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent;OUT=R/'hybrid-pool';MX=R/'mx-prepack/backend'
def main():
    p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');a=p.parse_args()
    suffix='-san' if a.sanitize else '';flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1'] if a.sanitize else []
    out=OUT/('san' if a.sanitize else 'normal');out.mkdir(exist_ok=True)
    prior=json.loads((MX/'BUILD.json').read_text())['variants']['candidate']
    cmd=prior['compile'].copy();source=OUT/'ggml-cpu.c';obj=out/'ggml-cpu.c.o';old_obj=cmd[cmd.index('-o')+1]
    cmd[cmd.index('-c')+1]=str(source);cmd[cmd.index('-o')+1]=str(obj);cmd+=['-pthread',*flags]
    assert '-DGGML_USE_OPENMP' in cmd and '-fopenmp' in cmd
    subprocess.run(cmd,check=True,timeout=180)
    runtime=out/'runtime';runtime.mkdir(exist_ok=True);library=runtime/'libggml-cpu.so.0.28.0'
    linkinfo=prior['link'];link=[str(obj) if x==old_obj else x for x in linkinfo['argv']]
    link[link.index('-o')+1]=str(library.with_suffix('.new'))
    link=[('-Wl,--dependency-file='+str(out/'link.d')) if x.startswith('-Wl,--dependency-file=') else str(MX/('mx-rhs'+suffix+'.o')) if x==str(MX/'mx-rhs.o') else x for x in link]
    link+=['-pthread',*flags];subprocess.run(link,cwd=linkinfo['cwd'],check=True,timeout=120);library.with_suffix('.new').replace(library)
    for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
        if not (runtime/alias).exists():(runtime/alias).symlink_to(library.name)
    test=json.loads((R/'sparse-cache/TEST_BUILD.json').read_text())
    test=[str(OUT/'test-hybrid.cpp') if x==str(R/'sparse-cache/test-sparse.cpp') else str(OUT/('test-hybrid'+suffix)) if x==str(R/'sparse-cache/test-sparse') else x for x in test]
    test+=['-pthread','-ldl','-UNDEBUG',*flags];subprocess.run(test,check=True,timeout=180)
    iq_source=(R/'sparse-cache/test-sparse.cpp').read_text()
    assert iq_source.count('for (int threads : {1,12})')==1
    iq_source=iq_source.replace('for (int threads : {1,12})','for (int threads : {24})')
    (OUT/'test-iq-native.cpp').write_text(iq_source)
    iqcmd=json.loads((R/'sparse-cache/TEST_BUILD.json').read_text())
    iqcmd=[str(OUT/'test-iq-native.cpp') if x==str(R/'sparse-cache/test-sparse.cpp') else str(OUT/('test-iq-native'+suffix)) if x==str(R/'sparse-cache/test-sparse') else x for x in iqcmd]
    iqcmd+=['-pthread','-UNDEBUG',*flags];subprocess.run(iqcmd,check=True,timeout=180)
    (out/'BUILD.json').write_text(json.dumps({'compile':cmd,'link':{'argv':link,'cwd':linkinfo['cwd']},'test_compile':test,'IQ_test_compile':iqcmd,'library':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'sources':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['ggml-cpu.c','packed-generation.h','test-hybrid.cpp','test-iq-native.cpp']}},indent=2)+'\n')
    print('Built isolated mixed executor'+suffix,flush=True)
if __name__=='__main__':main()
