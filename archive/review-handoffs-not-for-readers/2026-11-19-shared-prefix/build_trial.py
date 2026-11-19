from pathlib import Path
import hashlib,json,subprocess,os
R=Path(__file__).resolve().parent;LAB=R.parent;M=R/'trial';S=LAB/'deepseek41-bringup-20261101/source';B=S.parent/'build/bin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def main():
    assert not (M/'BUILD.json').exists()
    libs=json.loads((R/'INITIAL_RUNTIME.json').read_text())['mapped_runtime_hashes'];rt=M/'runtime';rt.mkdir(exist_ok=True)
    for name,digest in libs.items():
        p=Path(name);assert sha(p)==digest
        for n in [p.name,p.name.rsplit('.',2)[0] if p.name.startswith('libllama') else p.name.rsplit('.',2)[0],p.name.split('.so')[0]+'.so']:
            link=rt/n
            if not link.exists():link.symlink_to(p)
    cmd=['g++','-std=c++17','-O2','-g','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-fopenmp','-pthread']
    for inc in [M,S/'include',S/'ggml/include',S/'ggml/src',S/'common',S/'src',S/'tools/mtmd',S/'vendor/nlohmann']:cmd+=['-I'+str(inc)]
    cmd += [str(M/'driver.cpp'),'-L'+str(rt),'-L'+str(B),'-Wl,-rpath,'+str(rt)+':'+str(B),'-Wl,--no-as-needed','-lllama','-lggml-cpu','-lggml-base','-lggml','-lllama-common','-lmtmd','-ldl','-Wl,--no-undefined','-o',str(M/'driver')]
    with (M/'BUILD.stdout.log').open('w') as out,(M/'BUILD.stderr.log').open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,timeout=180)
    receipt=dict(command=cmd,returncode=r.returncode,source_sha256={p.name:sha(p) for p in [M/'driver.cpp',M/'CASES.json']},libraries={Path(p).name:dict(path=p,sha256=h) for p,h in libs.items()})
    if not r.returncode:receipt['driver_sha256']=sha(M/'driver')
    write(M/'BUILD.json',receipt);print(json.dumps({'returncode':r.returncode,'receipt':str(M/'BUILD.json')}));raise SystemExit(r.returncode)
if __name__=='__main__':main()
